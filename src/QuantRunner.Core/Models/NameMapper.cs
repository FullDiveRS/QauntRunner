using System.Text.RegularExpressions;

namespace QuantRunner.Core.Models;

/// <summary>
/// Maps external tensor names (GGUF / HuggingFace) to QuantRunner canonical names.
/// Canonical scheme:
///   tok_emb, lm_head?, final_norm
///   l{i}.attn_norm, l{i}.wq, l{i}.wk, l{i}.wv, l{i}.wo, l{i}.ffn_norm
///   l{i}.w_gate, l{i}.w_up, l{i}.w_down            (dense FFN)
///   l{i}.router                                    (MoE gate)
///   l{i}.e{j}.w_gate, l{i}.e{j}.w_up, l{i}.e{j}.w_down   (experts)
///
/// Fused projections are returned as pseudo names that the converter splits into several
/// canonical tensors once the architecture config is known:
///   QKV:{i}      fused q/k/v, contiguous rows  (q rows | k rows | v rows)
///   QKVGLM:{i}   fused q/k/v, GLM/BLOOM interleaved layout [head][3][head_dim]
///   GU:{i}       fused gate/up, contiguous halves
///   EGU:{i}:{j}  fused expert gate/up
/// </summary>
public static class NameMapper
{
    public enum Source { Gguf, HuggingFace }

    /// <summary>
    /// Per-layer attention families detected from the checkpoint's own tensor names
    /// (self_attn.A_log / conv1d => KDA, self_attn.q_a_proj / kv_b_proj => MLA).
    /// The mapper uses this to disambiguate shared names like self_attn.q_proj, which are
    /// GQA projections on normal layers but KDA projections on linear-attention layers.
    /// </summary>
    public sealed class LayerKinds
    {
        public HashSet<int> Kda = new();
        public HashSet<int> Mla = new();
    }

    /// <summary>Map a tensor name. Returns canonical name, or null to skip (with reason).</summary>
    public static string? Map(Source src, string name, long[] dims, int nLayers, out string note) =>
        Map(src, name, dims, nLayers, null, out note);

    /// <summary>Map a tensor name with layer-kind context (GLM-5.x KDA/MLA checkpoints).</summary>
    public static string? Map(Source src, string name, long[] dims, int nLayers, LayerKinds? kinds, out string note)
    {
        note = "";
        if (src == Source.Gguf) return MapGguf(name, dims, out note);
        return MapHf(name, nLayers, kinds, out note);
    }

    /// <summary>Detect per-layer attention kinds from raw checkpoint names (call before Map).</summary>
    public static LayerKinds? DetectLayerKinds(IEnumerable<string> allNames)
    {
        var kinds = new LayerKinds();
        foreach (var n in allNames)
        {
            var mk = Regex.Match(n, @"(?:^|\.)layers\.(\d+)\.self_attn\.(?:A_log|dt_bias|(?:q|k|v)_conv1d)\b");
            if (mk.Success) kinds.Kda.Add(int.Parse(mk.Groups[1].Value));
            var mm = Regex.Match(n, @"(?:^|\.)layers\.(\d+)\.self_attn\.(?:q_a_proj|q_b_proj|kv_a_proj|kv_b_proj|q_a_layernorm|kv_a_layernorm)\b");
            if (mm.Success) kinds.Mla.Add(int.Parse(mm.Groups[1].Value));
        }
        return kinds.Kda.Count + kinds.Mla.Count > 0 ? kinds : null;
    }

    private static string? MapHf(string name, int nLayers, LayerKinds? kinds, out string note)
    {
        note = "";
        string n = name;

        // transformers-5 multimodal containers nest the language model one level deeper
        // (GLM-5.x: "model.language_model.layers.3..."; legacy VLMs: "language_model.model.layers...")
        if (n.StartsWith("model.language_model.", StringComparison.Ordinal))
            n = "model." + n["model.language_model.".Length..];
        else if (n.StartsWith("language_model.model.", StringComparison.Ordinal))
            n = "model." + n["language_model.model.".Length..];
        else if (n.StartsWith("language_model.", StringComparison.Ordinal))
            n = "model." + n["language_model.".Length..];

        if (n.EndsWith(".bias", StringComparison.Ordinal) &&
            !n.Contains("e_score_correction_bias", StringComparison.Ordinal))
        { note = "bias tensor (not used)"; return null; }
        if (n.EndsWith(".weight_scale_inv", StringComparison.Ordinal)) return null;   // fp8 scale companion — consumed by the reader, never planned
        if (n.Contains("rotary_emb", StringComparison.Ordinal)) { note = "rotary_emb buffer (not used)"; return null; }

        switch (n)
        {
            case "model.embed_tokens.weight":
            case "model.tok_embeddings.weight":
            case "transformer.embedding.word_embeddings.weight":
            case "transformer.word_embeddings.weight":
                return "tok_emb";
            case "lm_head.weight":
            case "output.weight":
                return "lm_head";
            case "model.norm.weight":
            case "model.final_layernorm.weight":
            case "transformer.encoder.final_layernorm.weight":
                return "final_norm";
        }

        // ---- model.layers.{i}.* (Llama-family) / transformer.encoder.layers.{i}.* (ChatGLM) / layers.{i}.* (original-Llama) ----
        var m = Regex.Match(n, @"^(?:model|transformer\.encoder)\.layers\.(\d+)\.(.+)$");
        if (!m.Success)
            m = Regex.Match(n, @"^layers\.(\d+)\.(.+)$");
        if (m.Success)
        {
            int i = int.Parse(m.Groups[1].Value);
            if (i >= nLayers) { note = "MTP / next-layer weights (beyond num_hidden_layers)"; return null; }
            string rest = m.Groups[2].Value;

            // ---- mHC hyper-connection params (GLM-5.x, bare names without .weight) ----
            switch (rest)
            {
                case "hc_attn_fn": return $"l{i}.hc_attn_fn";
                case "hc_attn_base": return $"l{i}.hc_attn_base";
                case "hc_attn_scale": return $"l{i}.hc_attn_scale";
                case "hc_ffn_fn": return $"l{i}.hc_ffn_fn";
                case "hc_ffn_base": return $"l{i}.hc_ffn_base";
                case "hc_ffn_scale": return $"l{i}.hc_ffn_scale";
            }

            bool isKda = kinds?.Kda.Contains(i) == true;
            bool isMla = kinds?.Mla.Contains(i) == true;

            // ---- MLA (multi-latent) attention — GLM-5.3-Flash / DeepSeek-V3 style ----
            if (isMla)
            {
                switch (rest)
                {
                    case "self_attn.q_a_proj.weight": return $"l{i}.mla.qa";
                    case "self_attn.q_a_layernorm.weight": return $"l{i}.mla.qanorm";
                    case "self_attn.q_b_proj.weight": return $"l{i}.mla.qb";
                    case "self_attn.kv_a_proj_with_mqa.weight": return $"l{i}.mla.kva";
                    case "self_attn.kv_a_layernorm.weight": return $"l{i}.mla.kvanorm";
                    case "self_attn.kv_b_proj.weight": return $"l{i}.mla.kvb";
                    case "self_attn.o_proj.weight": return $"l{i}.mla.o";
                }
                if (rest.StartsWith("self_attn.indexer", StringComparison.Ordinal))
                {
                    note = $"'{name}': DSA indexer (deepseek sparse attention) — exact for context <= index_topk," +
                           " engine runs full attention over the latent cache instead";
                    return null;
                }
            }

            // ---- KDA (Kimi delta / gated-delta linear attention) — GLM-5.3-Flash linear layers ----
            if (isKda)
            {
                switch (rest)
                {
                    case "self_attn.q_proj.weight": return $"l{i}.kda.q";
                    case "self_attn.k_proj.weight": return $"l{i}.kda.k";
                    case "self_attn.v_proj.weight": return $"l{i}.kda.v";
                    case "self_attn.o_proj.weight": return $"l{i}.kda.o";
                    case "self_attn.q_conv1d.weight": return $"l{i}.kda.qconv";
                    case "self_attn.k_conv1d.weight": return $"l{i}.kda.kconv";
                    case "self_attn.v_conv1d.weight": return $"l{i}.kda.vconv";
                    case "self_attn.A_log": return $"l{i}.kda.alog";
                    case "self_attn.dt_bias": return $"l{i}.kda.dt";
                    case "self_attn.b_proj.weight": return $"l{i}.kda.b";
                    case "self_attn.f_a_proj.weight": return $"l{i}.kda.fa";
                    case "self_attn.f_b_proj.weight": return $"l{i}.kda.fb";
                    case "self_attn.g_a_proj.weight": return $"l{i}.kda.ga";
                    case "self_attn.g_b_proj.weight": return $"l{i}.kda.gb";
                    case "self_attn.o_norm.weight": return $"l{i}.kda.onorm";
                }
            }

            // explicit unsupported attention families when no kind context is available — explain WHY
            if (!isMla && (rest.StartsWith("self_attn.q_a_proj") || rest.StartsWith("self_attn.q_b_proj") ||
                           rest.StartsWith("self_attn.kv_a_proj") || rest.StartsWith("self_attn.kv_b_proj")))
            {
                note = $"'{name}': MLA (multi-latent / DeepSeek-style) attention — not supported by the engine";
                return null;
            }
            if (!isKda && (rest.StartsWith("linear_attn") || rest.StartsWith("self_attn.linear_attn") ||
                           rest.Contains("conv1d", StringComparison.Ordinal) || rest.StartsWith("in_proj_") ||
                           rest.StartsWith("self_attn.beta") || rest.StartsWith("self_attn.A_log")))
            {
                note = $"'{name}': linear-attention (KDA / gated-delta) — not supported by the engine";
                return null;
            }
            if (rest.EndsWith("q_norm.weight", StringComparison.Ordinal) ||
                rest.EndsWith("k_norm.weight", StringComparison.Ordinal))
            {
                note = $"'{name}': QK-norm not wired into forward — skipped";
                return null;
            }
            if (rest.Contains("shared_expert", StringComparison.Ordinal) ||
                rest.Contains("share_expert", StringComparison.Ordinal))
            {
                // GLM-5.x shared expert is wired into the MoE forward (sigmoid-router models)
                var se = Regex.Match(rest, @"^mlp\.shared_experts\.(gate|up|down)_proj\.weight$");
                if (se.Success) return $"l{i}.sw_{se.Groups[1].Value}";
                note = $"'{name}': shared-expert tensor skipped (not yet wired into forward)";
                return null;
            }
            if (rest.StartsWith("post_self_attn_layernorm") || rest.StartsWith("post_mlp_layernorm"))
                return null;   // extra sandwich norms (GLM-4.5 style) — engine uses the standard pair

            switch (rest)
            {
                case "input_layernorm.weight":
                case "attention_norm.weight":
                    return $"l{i}.attn_norm";
                case "post_attention_layernorm.weight":
                case "ffn_norm.weight":
                    return $"l{i}.ffn_norm";

                case "self_attn.q_proj.weight":
                case "attention.wq.weight":
                    return $"l{i}.wq";
                case "self_attn.k_proj.weight":
                case "attention.wk.weight":
                    return $"l{i}.wk";
                case "self_attn.v_proj.weight":
                case "attention.wv.weight":
                    return $"l{i}.wv";
                case "self_attn.o_proj.weight":
                case "attention.wo.weight":
                    return $"l{i}.wo";

                // fused attention projections — split into wq/wk/wv once the config is known
                case "self_attn.qkv_proj.weight":
                case "self_attn.W_pack.weight":
                    return $"QKV:{i}";
                case "self_attention.query_key_value.weight":   // ChatGLM/BLOOM interleaved layout
                case "self_attn.query_key_value.weight":
                    return $"QKVGLM:{i}";

                case "mlp.gate_proj.weight":
                case "feed_forward.w1.weight":
                    return $"l{i}.w_gate";
                case "mlp.up_proj.weight":
                case "feed_forward.w3.weight":
                    return $"l{i}.w_up";
                case "mlp.down_proj.weight":
                case "feed_forward.w2.weight":
                    return $"l{i}.w_down";
                case "mlp.gate_up_proj.weight":
                case "feed_forward.gate_up_proj.weight":
                    return $"GU:{i}";

                case "mlp.gate.weight":
                case "block_sparse_moe.gate.weight":
                case "mlp.router.weight":
                case "feed_forward.router.weight":
                    return $"l{i}.router";
                case "mlp.gate.e_score_correction_bias":        // GLM/DeepSeek noaux_tc router bias
                    return $"l{i}.router_bias";
            }

            // experts: mlp.experts.{j}.{gate,up,down}_proj.weight (DeepSeek style)
            var me = Regex.Match(rest, @"^mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$");
            if (me.Success)
                return $"l{i}.e{me.Groups[1].Value}.{Sub(me.Groups[2].Value)}";
            // fused expert projections
            var meu = Regex.Match(rest, @"^(?:mlp|feed_forward)\.experts\.(\d+)\.gate_up_proj\.weight$");
            if (meu.Success)
                return $"EGU:{i}:{meu.Groups[1].Value}";
            // Mixtral: block_sparse_moe.experts.{j}.{w1,w3,w2}.weight
            var mm2 = Regex.Match(rest, @"^block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$");
            if (mm2.Success)
            {
                string part = mm2.Groups[2].Value switch { "w1" => "gate", "w3" => "up", _ => "down" };
                return $"l{i}.e{mm2.Groups[1].Value}.w_{part}";
            }
            // Llama-4 style: feed_forward.experts.{j}.{w1,w3,w2}.weight
            var m4 = Regex.Match(rest, @"^feed_forward\.experts\.(\d+)\.(w1|w2|w3)\.weight$");
            if (m4.Success)
            {
                string part = m4.Groups[2].Value switch { "w1" => "gate", "w3" => "up", _ => "down" };
                return $"l{i}.e{m4.Groups[1].Value}.w_{part}";
            }
        }
        return null;
    }

    private static string Sub(string p) => p switch { "gate" => "w_gate", "up" => "w_up", _ => "w_down" };

    private static string? MapGguf(string name, long[] dims, out string note)
    {
        note = "";
        if (name == "token_embd.weight") return "tok_emb";
        if (name == "output.weight") return "lm_head";
        if (name == "output_norm.weight") return "final_norm";

        var m = Regex.Match(name, @"^blk\.(\d+)\.(.+)$");
        if (!m.Success) return null;
        int i = int.Parse(m.Groups[1].Value);
        string rest = m.Groups[2].Value;
        switch (rest)
        {
            case "attn_norm": return $"l{i}.attn_norm";
            case "attn_norm.weight": return $"l{i}.attn_norm";
            case "attn_q": case "attn_q.weight": return $"l{i}.wq";
            case "attn_k": case "attn_k.weight": return $"l{i}.wk";
            case "attn_v": case "attn_v.weight": return $"l{i}.wv";
            case "attn_o": case "attn_o.weight": return $"l{i}.wo";
            case "ffn_norm": case "ffn_norm.weight": return $"l{i}.ffn_norm";
            case "ffn_gate": case "ffn_gate.weight": return $"l{i}.w_gate";
            case "ffn_up": case "ffn_up.weight": return $"l{i}.w_up";
            case "ffn_down": case "ffn_down.weight": return $"l{i}.w_down";
            case "ffn_gate_inp": case "ffn_gate_inp.weight": return $"l{i}.router";
        }
        // 3-D packed experts (ggml MoE layout)
        if (name.Contains("ffn_gate_exps")) return $"EXPSPLIT:{i}:gate";
        if (name.Contains("ffn_up_exps")) return $"EXPSPLIT:{i}:up";
        if (name.Contains("ffn_down_exps")) return $"EXPSPLIT:{i}:down";
        return null;
    }
}
