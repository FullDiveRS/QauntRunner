using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using QuantRunner.Core;
using QuantRunner.Core.Importers;
using QuantRunner.Core.Memory;
using QuantRunner.Core.Models;
using QuantRunner.Core.Quantization;

namespace QuantRunner.Cli;

/// <summary>Builds tiny synthetic .qw models and tokenizer.json files for tests/benches.</summary>
internal static class Synth
{
    public static void Write(string path, int dim, int layers, int heads, int ffn, int vocab,
        int experts, int active, int seed = 5, QuantFormat fmt = QuantFormat.Q4_K)
    {
        int headDim = dim / heads;
        int kvHeads = heads; // dense K/V in synthetic models (GQA covered by real converts)
        var cfg = new ModelConfig
        {
            Arch = experts > 0 ? "moe" : "llama",
            HiddenSize = dim, NumLayers = layers, NumHeads = heads, NumKvHeads = kvHeads,
            HeadDim = headDim, FfnDim = ffn, VocabSize = vocab,
            NumExperts = experts, NumActiveExperts = active, ExpertFfnDim = experts > 0 ? ffn : -1,
            MaxCtx = 2048, SourceDescription = "synthetic"
        };
        var rng = new Random(seed);

        using var b = new ModelStoreBuilder(path, ConfigWithKvs(cfg));
        var plans = new List<(TensorPlan P, float Scale)>();
        void Plan(string name, long[] dims, QuantFormat f, float scale)
            => plans.Add((b.Plan(name, dims, f), scale));

        Plan("tok_emb", new long[] { vocab, dim }, QuantFormat.F16, 0.02f);
        Plan("final_norm", new long[] { dim }, QuantFormat.F32, 1f);
        for (int l = 0; l < layers; l++)
        {
            Plan($"l{l}.attn_norm", new long[] { dim }, QuantFormat.F32, 1f);
            Plan($"l{l}.wq", new long[] { dim, dim }, fmt, 0.02f);
            Plan($"l{l}.wk", new long[] { kvHeads * headDim, dim }, fmt, 0.02f);
            Plan($"l{l}.wv", new long[] { kvHeads * headDim, dim }, fmt, 0.02f);
            Plan($"l{l}.wo", new long[] { dim, dim }, fmt, 0.02f);
            Plan($"l{l}.ffn_norm", new long[] { dim }, QuantFormat.F32, 1f);
            if (experts > 0)
            {
                Plan($"l{l}.router", new long[] { experts, dim }, QuantFormat.F32, 1f);
                for (int e = 0; e < experts; e++)
                {
                    Plan($"l{l}.e{e}.w_gate", new long[] { ffn, dim }, fmt, 0.02f);
                    Plan($"l{l}.e{e}.w_up", new long[] { ffn, dim }, fmt, 0.02f);
                    Plan($"l{l}.e{e}.w_down", new long[] { dim, ffn }, fmt, 0.02f);
                }
            }
            else
            {
                Plan($"l{l}.w_gate", new long[] { ffn, dim }, fmt, 0.02f);
                Plan($"l{l}.w_up", new long[] { ffn, dim }, fmt, 0.02f);
                Plan($"l{l}.w_down", new long[] { dim, ffn }, fmt, 0.02f);
            }
        }
        Plan("lm_head", new long[] { vocab, dim }, QuantFormat.F16, 0.02f);

        b.Commit((plan, stream) =>
        {
            var (_, scale) = plans.First(p => p.P.Name == plan.Name);
            long n = plan.NumElements;
            var buf = new float[n];
            bool isNorm = plan.Name.Contains("norm") || plan.Name.EndsWith("router");
            for (int i = 0; i < n; i++)
                buf[i] = isNorm ? 1f : NextGaussian(rng) * scale;

            // quantize in one shot (test tensors are small)
            var bytes = new byte[plan.SizeBytes];
            if (plan.Format.IsQuantized())
                BlockQuantizer.Quantize(plan.Format, buf, bytes.AsSpan(0, bytes.Length));
            else
            {
                // fp formats: emit raw little-endian
                for (int i = 0; i < n; i++)
                {
                    switch (plan.Format)
                    {
                        case QuantFormat.F32:
                            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), buf[i]);
                            break;
                        case QuantFormat.F16:
                            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2), Dtypes.FloatToF16(buf[i]));
                            break;
                        default:
                            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2), Dtypes.FloatToBf16(buf[i]));
                            break;
                    }
                }
            }
            stream.Write(bytes, 0, bytes.Length);
        });
    }

    /// <summary>
    /// Builds a GLM-5.x-shaped hybrid test model — the GLM-5.3-Flash machinery at tiny scale:
    /// per-layer attention kinds in a 3:1 KDA:MLA pattern, mHC hyper-connections (default 2
    /// streams), MLA without RoPE (GLM-5.3-Flash uses NoPE), KDA with the bounded-sigmoid
    /// forget gate (lower bound -5), and when experts > 0 the GLM sigmoid-MoE (choice bias,
    /// routed scaling, shared expert, limited SwiGLU). Weights are random: this exercises the
    /// runtime machinery and shapes, not model quality. The mini BPE tokenizer is embedded, so
    /// `run` works with no external files.
    /// </summary>
    public static void WriteGlm5(string path, int dim, int layers, int heads, int ffn, int vocab,
        int experts, int active, int hcMult = 2, int seed = 7, QuantFormat fmt = QuantFormat.Q4_K)
    {
        // fixed small MLA/KDA geometry (256-aligned so K-quants quantize without fallbacks)
        const int qLora = 256, kvLora = 256, nope = 64, vHead = 64, rope = 0;   // NoPE
        const int dk = 128, linHeads = 2, kernel = 4;                           // qkv = 256
        int qkDim = nope + rope, qkv = linHeads * dk, hcW = (2 + hcMult) * hcMult;

        var kinds = new string[layers];
        for (int l = 0; l < layers; l++) kinds[l] = l % 4 == 3 ? "mla" : "kda"; // 3 KDA : 1 MLA
        var cfg = new ModelConfig
        {
            Arch = experts > 0 ? "moe" : "llama",
            HiddenSize = dim, NumLayers = layers, NumHeads = heads, HeadDim = 64,
            FfnDim = ffn, VocabSize = vocab,
            NumExperts = experts, NumActiveExperts = active, ExpertFfnDim = experts > 0 ? ffn : -1,
            NumSharedExperts = experts > 0 ? 1 : 0, MoeSigmoid = experts > 0, RoutedScale = 2.5f,
            SwigluLimit = 10f,
            AttnKinds = kinds, HcMult = hcMult,
            QLoraRank = qLora, KvLoraRank = kvLora,
            QkNopeHeadDim = nope, QkRopeHeadDim = rope, VHeadDim = vHead,
            LinearHeads = linHeads, LinearHeadDim = dk, LinearConvKernel = kernel,
            LinearLowerBound = -5f,
            MaxCtx = 2048, SourceDescription = $"synthetic-glm5(seed={seed})"
        };
        var rng = new Random(seed);
        using var b = new ModelStoreBuilder(path, ConfigWithKvs(cfg));
        void Plan(string name, long[] dims, QuantFormat f) => b.Plan(name, dims, f);

        Plan("tok_emb", new long[] { vocab, dim }, QuantFormat.F16);
        Plan("final_norm", new long[] { dim }, QuantFormat.F32);
        for (int l = 0; l < layers; l++)
        {
            Plan($"l{l}.attn_norm", new long[] { dim }, QuantFormat.F32);
            Plan($"l{l}.ffn_norm", new long[] { dim }, QuantFormat.F32);
            // mHC entry/exit weights (tiny tensors: F32 keeps row handling trivial)
            Plan($"l{l}.hc_attn_fn", new long[] { hcW, (long)hcMult * dim }, QuantFormat.F32);
            Plan($"l{l}.hc_attn_base", new long[] { hcW }, QuantFormat.F32);
            Plan($"l{l}.hc_attn_scale", new long[] { 3 }, QuantFormat.F32);
            Plan($"l{l}.hc_ffn_fn", new long[] { hcW, (long)hcMult * dim }, QuantFormat.F32);
            Plan($"l{l}.hc_ffn_base", new long[] { hcW }, QuantFormat.F32);
            Plan($"l{l}.hc_ffn_scale", new long[] { 3 }, QuantFormat.F32);

            if (kinds[l] == "mla")
            {
                Plan($"l{l}.mla.qa", new long[] { qLora, dim }, fmt);
                Plan($"l{l}.mla.qanorm", new long[] { qLora }, QuantFormat.F32);
                Plan($"l{l}.mla.qb", new long[] { (long)heads * qkDim, qLora }, fmt);
                Plan($"l{l}.mla.kva", new long[] { kvLora + rope, dim }, fmt);
                Plan($"l{l}.mla.kvanorm", new long[] { kvLora }, QuantFormat.F32);
                Plan($"l{l}.mla.kvb", new long[] { (long)heads * (nope + vHead), kvLora }, fmt);
                Plan($"l{l}.mla.o", new long[] { dim, (long)heads * vHead }, fmt);
            }
            else
            {
                Plan($"l{l}.kda.q", new long[] { qkv, dim }, fmt);
                Plan($"l{l}.kda.k", new long[] { qkv, dim }, fmt);
                Plan($"l{l}.kda.v", new long[] { qkv, dim }, fmt);
                Plan($"l{l}.kda.o", new long[] { dim, qkv }, fmt);
                Plan($"l{l}.kda.qconv", new long[] { qkv, kernel }, QuantFormat.F32);   // depthwise taps
                Plan($"l{l}.kda.kconv", new long[] { qkv, kernel }, QuantFormat.F32);
                Plan($"l{l}.kda.vconv", new long[] { qkv, kernel }, QuantFormat.F32);
                Plan($"l{l}.kda.fa", new long[] { dk, dim }, fmt);
                Plan($"l{l}.kda.fb", new long[] { qkv, dk }, fmt);
                Plan($"l{l}.kda.ga", new long[] { dk, dim }, fmt);
                Plan($"l{l}.kda.gb", new long[] { qkv, dk }, fmt);
                Plan($"l{l}.kda.b", new long[] { linHeads, dim }, fmt);
                Plan($"l{l}.kda.alog", new long[] { linHeads }, QuantFormat.F32);
                Plan($"l{l}.kda.dt", new long[] { qkv }, QuantFormat.F32);
                Plan($"l{l}.kda.onorm", new long[] { dk }, QuantFormat.F32);
            }

            if (experts > 0)
            {
                Plan($"l{l}.router", new long[] { experts, dim }, QuantFormat.F32);
                Plan($"l{l}.router_bias", new long[] { experts }, QuantFormat.F32);
                for (int e = 0; e < experts; e++)
                {
                    Plan($"l{l}.e{e}.w_gate", new long[] { ffn, dim }, fmt);
                    Plan($"l{l}.e{e}.w_up", new long[] { ffn, dim }, fmt);
                    Plan($"l{l}.e{e}.w_down", new long[] { dim, ffn }, fmt);
                }
                Plan($"l{l}.sw_gate", new long[] { ffn, dim }, fmt);
                Plan($"l{l}.sw_up", new long[] { ffn, dim }, fmt);
                Plan($"l{l}.sw_down", new long[] { dim, ffn }, fmt);
            }
            else
            {
                Plan($"l{l}.w_gate", new long[] { ffn, dim }, fmt);
                Plan($"l{l}.w_up", new long[] { ffn, dim }, fmt);
                Plan($"l{l}.w_down", new long[] { dim, ffn }, fmt);
            }
        }
        Plan("lm_head", new long[] { vocab, dim }, QuantFormat.F16);

        b.Commit((plan, stream) =>
        {
            long n = plan.NumElements;
            var buf = new float[n];
            bool isOne = plan.Name.Contains("norm") || plan.Name.EndsWith("router")
                      || plan.Name.EndsWith("_base") || plan.Name.EndsWith("_scale");
            bool isTiny = plan.Name.EndsWith("router_bias") || plan.Name.Contains("conv")
                       || plan.Name.EndsWith("alog") || plan.Name.EndsWith(".dt");
            for (int i = 0; i < n; i++)
                buf[i] = isOne ? 1f : isTiny ? NextGaussian(rng) * 0.05f : NextGaussian(rng) * 0.02f;

            var bytes = new byte[plan.SizeBytes];
            if (plan.Format.IsQuantized())
                BlockQuantizer.Quantize(plan.Format, buf, bytes.AsSpan(0, bytes.Length));
            else if (plan.Format == QuantFormat.F32)
                Buffer.BlockCopy(buf, 0, bytes, 0, bytes.Length);
            else
            {
                for (int i = 0; i < n; i++)
                    BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2),
                        Dtypes.FloatToF16(buf[i]));
            }
            stream.Write(bytes, 0, bytes.Length);
        });

        int kdaN = kinds.Count(k => k == "kda"), mlaN = kinds.Length - kdaN;
        Console.WriteLine($"  GLM-5.x hybrid: {layers} layers ({kdaN} KDA / {mlaN} MLA, 3:1), " +
                          $"mHC x{hcMult}, MLA q_lora {qLora}/kv_lora {kvLora} head {nope}(+rope {rope}) v {vHead}, " +
                          $"KDA {linHeads}x{dk} conv {kernel} lb {cfg.LinearLowerBound}" +
                          (experts > 0 ? $", GLM-MoE {experts}/{active}+1 shared" : "") +
                          $", quant={fmt}");
        Console.WriteLine($"  wrote {path} ({Program.Fmt(new FileInfo(path).Length)})");
    }

    internal static float NextGaussian(Random r)
    {
        double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
    }

    /// <summary>
    /// Minimal GPT-2-style byte-level BPE tokenizer.json: all 256 byte chars as base tokens
    /// (ids 0..255 in bytes-to-unicode order), plus merges h+e and l+l.
    /// </summary>
    public static string MiniTokenizerJson()
    {
        // GPT-2 bytes_to_unicode table
        var b2c = new Dictionary<int, char>();
        var bs = new List<int>();
        for (int b = '!'; b <= '~'; b++) bs.Add(b);
        for (int b = 0xA1; b <= 0xAC; b++) bs.Add(b);
        for (int b = 0xAE; b <= 0xFF; b++) bs.Add(b);
        int n = 0;
        for (int b = 0; b < 256; b++) if (!bs.Contains(b)) bs.Add(b);
        foreach (int b in bs)
        {
            char c = (char)b;
            if (c < '!' || c > '~') c = (char)(256 + n++);
            b2c[b] = c;
        }

        var vocab = new StringBuilder();
        vocab.Append('{');
        for (int i = 0; i < 256; i++)
        {
            char c = b2c[i];
            vocab.Append($"{JsonEncoder(c.ToString())}:{i}");
            if (i < 255) vocab.Append(',');
        }
        vocab.Append($",{JsonEncoder("he")}:256,{JsonEncoder("ll")}:257");
        vocab.Append('}');

        string h = b2c['h'].ToString(), e = b2c['e'].ToString(), l = b2c['l'].ToString();
        return $$"""
        {
          "version": "1.0",
          "added_tokens": [],
          "eos_token": "{{b2c[0]}}",
          "bos_token": "{{b2c[1]}}",
          "model": {
            "type": "BPE",
            "vocab": {{vocab}},
            "merges": [
              "{{h}} {{e}}",
              "{{l}} {{l}}"
            ]
          }
        }
        """;
    }

    private static string JsonEncoder(string s) => System.Text.Json.JsonSerializer.Serialize(s);

    /// <summary>ModelConfig JSON with the mini GGUF metadata (arch + tokenizer) embedded under
    /// gguf_kvs — synthesized models are self-contained: `run` reads the tokenizer from the file
    /// itself and convert qw -> gguf carries it over verbatim.</summary>
    public static JsonElement ConfigWithKvs(ModelConfig cfg)
    {
        var node = System.Text.Json.Nodes.JsonNode.Parse(cfg.ToJson())!.AsObject();
        node["gguf_kvs"] = System.Text.Json.Nodes.JsonNode.Parse(MiniGgufKvsJson(cfg))!;
        return JsonDocument.Parse(node.ToJsonString()).RootElement;
    }

    /// <summary>
    /// GGUF KV section (wire bytes) describing this synthetic model's architecture and the mini
    /// byte-level BPE tokenizer (256 byte chars + "he" + "ll", merges h+e and l+l). Embedded
    /// into the .qw header as gguf_kvs so synthesized models are self-contained: `run` finds the
    /// tokenizer inside the file, and convert qw -> gguf restores the same metadata verbatim.
    /// </summary>
    public static string MiniGgufKvsJson(ModelConfig cfg)
    {
        int dim = cfg.HiddenSize, layers = cfg.NumLayers, ffn = (int)cfg.FfnDim,
            heads = cfg.NumHeads, kvHeads = cfg.EffectiveKvHeads, vocab = cfg.VocabSize;
        var bs = new List<int>();
        for (int b = '!'; b <= '~'; b++) bs.Add(b);
        for (int b = 0xA1; b <= 0xAC; b++) bs.Add(b);
        for (int b = 0xAE; b <= 0xFF; b++) bs.Add(b);
        for (int b = 0; b < 256; b++) if (!bs.Contains(b)) bs.Add(b);
        var b2c = new Dictionary<int, char>();
        int n = 0;
        foreach (int b in bs)
        {
            char c = (char)b;
            if (c < '!' || c > '~') c = (char)(256 + n++);
            b2c[b] = c;
        }

        var tokens = new string[258];
        for (int i = 0; i < 256; i++) tokens[i] = b2c[i].ToString();
        tokens[256] = "he";
        tokens[257] = "ll";
        string h = b2c['h'].ToString(), e = b2c['e'].ToString(), l = b2c['l'].ToString();
        var types = new long[258];
        types[0] = 2; types[1] = 2;                    // byte 0/1 chars marked CONTROL (specials)

        var kvs = new List<(string, GgufFile.GgufValueType, object)>
        {
            ("general.architecture", GgufFile.GgufValueType.String, "llama"),
            ("general.name", GgufFile.GgufValueType.String, "synthetic"),
            ("llama.context_length", GgufFile.GgufValueType.U32, 2048u),
            ("llama.embedding_length", GgufFile.GgufValueType.U32, (uint)dim),
            ("llama.block_count", GgufFile.GgufValueType.U32, (uint)layers),
            ("llama.feed_forward_length", GgufFile.GgufValueType.U32, (uint)ffn),
            ("llama.attention.head_count", GgufFile.GgufValueType.U32, (uint)heads),
            ("llama.attention.head_count_kv", GgufFile.GgufValueType.U32, (uint)kvHeads),
            ("llama.attention.layer_norm_rms_epsilon", GgufFile.GgufValueType.F32, 1e-5f),
            ("llama.rope.freq_base", GgufFile.GgufValueType.F32, 10000f),
            ("llama.vocab_size", GgufFile.GgufValueType.U32, (uint)vocab),
            ("tokenizer.ggml.model", GgufFile.GgufValueType.String, "gpt2"),
            ("tokenizer.ggml.tokens", GgufFile.GgufValueType.Array, tokens),
            ("tokenizer.ggml.merges", GgufFile.GgufValueType.Array, new[] { $"{h} {e}", $"{l} {l}" }),
            ("tokenizer.ggml.token_type", GgufFile.GgufValueType.Array, types),
            ("tokenizer.ggml.bos_token_id", GgufFile.GgufValueType.U32, 1u),
            ("tokenizer.ggml.eos_token_id", GgufFile.GgufValueType.U32, 0u),
        };
        if (cfg.IsMoe)
        {
            kvs.Add(("llama.expert_count", GgufFile.GgufValueType.U32, (uint)cfg.NumExperts));
            kvs.Add(("llama.expert_used_count", GgufFile.GgufValueType.U32, (uint)cfg.NumActiveExperts));
        }

        using var ms = new MemoryStream();
        using (var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            foreach (var (key, type, value) in kvs)
            {
                WriteGgufString(bw, key);
                bw.Write((uint)type);
                switch (type)
                {
                    case GgufFile.GgufValueType.String: WriteGgufString(bw, (string)value); break;
                    case GgufFile.GgufValueType.U32: bw.Write(Convert.ToUInt32(value)); break;
                    case GgufFile.GgufValueType.F32: bw.Write(Convert.ToSingle(value)); break;
                    case GgufFile.GgufValueType.Array when value is string[] a:
                        bw.Write((uint)GgufFile.GgufValueType.String);
                        bw.Write((ulong)a.Length);
                        foreach (var s in a) WriteGgufString(bw, s);
                        break;
                    case GgufFile.GgufValueType.Array when value is long[] a:
                        bw.Write((uint)GgufFile.GgufValueType.I32);
                        bw.Write((ulong)a.Length);
                        foreach (var v in a) bw.Write((int)v);
                        break;
                    default: throw new ArgumentException($"mini gguf kv {key}");
                }
            }
        }
        return System.Text.Json.JsonSerializer.Serialize(new Dictionary<string, object>
        {
            ["v"] = 3,
            ["n_kv"] = kvs.Count,
            ["b64"] = Convert.ToBase64String(ms.ToArray())
        });
    }

    private static void WriteGgufString(BinaryWriter bw, string s)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        bw.Write((ulong)bytes.Length);
        bw.Write(bytes);
    }
}
