using System.Text.RegularExpressions;

namespace QuantRunner.Core.Models;

/// <summary>
/// Parameter counting and classification by tensor name. Works across the three naming
/// schemes QuantRunner reads: canonical .qw names (l0.wq, tok_emb, l0.e3.w_gate),
/// GGUF names (blk.0.attn_q, token_embd, blk.0.ffn_gate_exps) and HuggingFace names
/// (model.layers.0.self_attn.q_proj, model.embed_tokens, mlp.experts.3.gate_proj).
/// Pure name/shape math — no file access — so it is unit-testable and reused by `info`
/// for .qw, .gguf, .safetensors and HF directories alike.
/// </summary>
public static class ParamReport
{
    public enum B { Embedding, Head, Attention, LinearAttn, Ffn, RoutedExperts, SharedExperts, Router, Norms, Misc }

    public sealed class Stats
    {
        public long Total;
        public int TensorCount;
        public long Embedding;       // token (+ position) embeddings
        public long Head;            // lm_head / untied output projection
        public long Attention;       // GQA + MLA projections (q/k/v/o, fused qkv, MLA lora up/down)
        public long LinearAttn;      // KDA linear-attention projections, convs, decays, gates
        public long Ffn;             // dense FFN gate/up/down
        public long RoutedExperts;   // routed expert weights, summed over ALL experts
        public long SharedExperts;   // shared experts (active for every token)
        public long Router;          // MoE gate / router
        public long Norms;           // all norms (RMS/LayerNorm, q/k norms, MLA/KDA norms)
        public long Misc;            // biases of unknown modules, mHC hyper-connections, caches

        public long NonEmbedding => Total - Embedding - Head;

        public IEnumerable<(string Label, long N)> NonZeroBuckets()
        {
            if (Embedding > 0) yield return ("embedding", Embedding);
            if (Head > 0) yield return ("lm_head (untied)", Head);
            if (Attention > 0) yield return ("attention", Attention);
            if (LinearAttn > 0) yield return ("linear attn (KDA)", LinearAttn);
            if (Ffn > 0) yield return ("ffn (dense)", Ffn);
            if (RoutedExperts > 0) yield return ("experts (routed)", RoutedExperts);
            if (SharedExperts > 0) yield return ("experts (shared)", SharedExperts);
            if (Router > 0) yield return ("router", Router);
            if (Norms > 0) yield return ("norms", Norms);
            if (Misc > 0) yield return ("other", Misc);
        }
    }

    private static readonly Regex QwExpert = new(@"\.e\d+\.", RegexOptions.Compiled);
    private static readonly Regex LayerIdx = new(@"(?:^|\.)(?:l|blk|layer|layers|h)\.?(\d+)\.", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    /// <summary>Sum tensor element counts into per-component buckets.</summary>
    public static Stats Count(IEnumerable<(string Name, long[] Dims)> tensors)
    {
        var s = new Stats();
        foreach (var (name, dims) in tensors)
        {
            long n = 1;
            foreach (var d in dims) n *= d;
            s.Total += n;
            s.TensorCount++;
            switch (Bucket(name))
            {
                case B.Embedding: s.Embedding += n; break;
                case B.Head: s.Head += n; break;
                case B.Attention: s.Attention += n; break;
                case B.LinearAttn: s.LinearAttn += n; break;
                case B.Ffn: s.Ffn += n; break;
                case B.RoutedExperts: s.RoutedExperts += n; break;
                case B.SharedExperts: s.SharedExperts += n; break;
                case B.Router: s.Router += n; break;
                case B.Norms: s.Norms += n; break;
                default: s.Misc += n; break;
            }
        }
        return s;
    }

    /// <summary>Number of distinct transformer layer indexes seen in the names (0 if none).</summary>
    public static int LayersSeen(IEnumerable<string> names)
    {
        var set = new HashSet<int>();
        foreach (var n in names)
        {
            var m = LayerIdx.Match(n);
            if (m.Success) set.Add(int.Parse(m.Groups[1].Value));
        }
        return set.Count;
    }

    // order matters: norms first (mla.qanorm / output_norm / ffn_norm), then structural
    // buckets, then the broad attention/ffn name families last
    public static B Bucket(string rawName)
    {
        string n = rawName.ToLowerInvariant();
        bool Has(params string[] pats) { foreach (var p in pats) if (n.Contains(p)) return true; return false; }

        if (Has("norm", ".ln", "layernorm")) return B.Norms;              // incl. mla.qanorm, kda.onorm, output_norm
        if (Has("tok_emb", "embed_tokens", "token_embd", "word_emb", "embedding", "wte", "wpe", "pos_emb")) return B.Embedding;
        if (Has("lm_head") || n.EndsWith("output.weight") || n.EndsWith("output.bias")) return B.Head;

        if (Has("shexp", "shared_expert", ".sw_")) return B.SharedExperts; // gguf *_shexp | HF shared_expert | qw sw_gate
        if (Has("exps", "experts") || QwExpert.IsMatch(n)) return B.RoutedExperts;  // gguf ffn_*_exps | HF mlp.experts.N | qw l0.e3.w_gate
        if (Has("gate_inp", "exp_probs_b", "e_score_correction", ".router", "gate.classifier")) return B.Router;
        if (Has(".gate.weight") || n.EndsWith(".gate.bias") || n == "gate.weight") return B.Router;  // HF mlp.gate.weight

        if (Has(".kda.", "linear_attn", "conv1d", "in_proj_qkvz", "in_proj_ba", "a_log", "dt_bias")) return B.LinearAttn;
        if (Has("hc_")) return B.Misc;                                    // mHC hyper-connection params
        if (Has("attn", "attention", ".mla.", "query_key_value",
               ".wq", ".wk", ".wv", ".wo", ".wqkv", "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj",
               "qkv_proj", "w_pack", "rope")) return B.Attention;

        if (Has("ffn", "mlp", "feed_forward", "w_gate", "w_up", "w_down", "gate_proj", "up_proj", "down_proj")) return B.Ffn;
        return B.Misc;
    }

    /// <summary>
    /// Parameters actually touched per token for a MoE model: everything except the routed
    /// expert pool, plus nActive/nExperts of that pool. Shared experts are always active.
    /// Returns Total for dense models.
    /// </summary>
    public static long ActivePerToken(Stats s, int nExperts, int nActive)
    {
        if (nExperts <= 0 || nActive <= 0 || s.RoutedExperts <= 0) return s.Total;
        return s.Total - s.RoutedExperts + s.RoutedExperts * nActive / nExperts;
    }

    /// <summary>Human-readable parameter count: 671032832512 -> "671B", 7618378752 -> "7.62B".</summary>
    public static string Human(long n)
    {
        double a = Math.Abs(n);
        string S(double v, string u)
        {
            string t = Math.Abs(v) >= 100 ? v.ToString("F1") : v.ToString("F2");
            if (t.Contains('.')) t = t.TrimEnd('0').TrimEnd('.');
            return t + u;
        }
        return a >= 1e12 ? S(n / 1e12, "T")
             : a >= 1e9 ? S(n / 1e9, "B")
             : a >= 1e6 ? S(n / 1e6, "M")
             : a >= 1e3 ? S(n / 1e3, "K")
             : n.ToString();
    }
}
