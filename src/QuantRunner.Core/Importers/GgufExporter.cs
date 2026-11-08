using System.Text;
using System.Text.Json;
using QuantRunner.Core.Memory;
using QuantRunner.Core.Models;
using QuantRunner.Core.Quantization;

namespace QuantRunner.Core.Importers;

/// <summary>Options for exporting a .qw model to GGUF.</summary>
public sealed class GgufExportOptions
{
    /// <summary>"auto" (default): layout-identical formats pass through byte-exact, our Q4_0 is
    /// re-encoded as a true ggml Q4_0, TRI/sub-TRI/I8_BLK fall back to F16.
    /// "f16"/"f32"/"q8_0": every quantized 2-D weight is dequantized and re-encoded to that type.</summary>
    public string GgmlType = "auto";
    public string? ModelName;
}

/// <summary>
/// Exports a QuantRunner .qw model to a GGUF (v3) file that llama.cpp can load.
///
/// What survives:
///   • layout-identical formats (F32/F16/BF16, Q8_0, Q2_K..Q6_K) are copied byte-exact —
///     llama.cpp dequantizes them with the same formula (verified against ggml-quants.c),
///     so there is no quality change at all;
///   • our 20-byte Q4_0 is re-encoded as a true 18-byte ggml Q4_0 (keeps 4-bit size);
///   • TRI / B1 / B1X / T1E / T1E_LO / I8_BLK have no ggml counterpart and export as F16
///     (or F32/Q8_0 via --ggml-type);
///   • MoE experts are reassembled from l{i}.e{j}.* into the 3-D gguf layout;
///   • if the .qw was imported from GGUF, its full metadata (tokenizer, chat template, rope
///     scaling, arch details) was embedded in the .qw header and is restored verbatim — the
///     output tokenizes exactly like the source. A .qw quantized from safetensors carries no
///     tokenizer, so the output is weights-only for llama.cpp.
/// </summary>
public static class GgufExporter
{
    private const int Align = 32;   // general.alignment written by this exporter

    public static void Export(string qwPath, string outGgufPath, GgufExportOptions? options = null)
    {
        var opt = options ?? new GgufExportOptions();
        using var store = ModelStore.Open(qwPath);
        var cfg = ModelConfig.FromJson(store.ConfigJson);

        var (sourceKvs, ggufArch) = ParseEmbeddedKvs(store.ConfigJson);
        string? srcName = null;
        foreach (var (k, _, v) in sourceKvs) if (k == "general.name" && v is string s) { srcName = s; break; }

        Console.WriteLine($"QuantRunner convert: {Path.GetFileName(qwPath)} -> {Path.GetFileName(outGgufPath)} (GGUF v3)");
        var sw = System.Diagnostics.Stopwatch.StartNew();

        var skipped = new List<(string, string)>();
        var typeCounts = new Dictionary<string, int>();
        var outs = BuildTensorPlan(store, opt, skipped, typeCounts);

        // ---------------------------------------------------------------- metadata
        string arch = ggufArch ?? "llama";
        var finalKvs = new List<(string Key, GgufFile.GgufValueType Type, object Value)>();
        var overrideKeys = new HashSet<string>();
        void Emit(string key, GgufFile.GgufValueType type, object value)
        {
            finalKvs.Add((key, type, value));
            overrideKeys.Add(key);
        }

        Emit("general.architecture", GgufFile.GgufValueType.String, arch);
        Emit("general.name", GgufFile.GgufValueType.String,
            opt.ModelName ?? srcName ?? Path.GetFileNameWithoutExtension(qwPath));
        Emit("general.file_type", GgufFile.GgufValueType.U32, (uint)ComputeFileType(outs));
        Emit("general.quantization_version", GgufFile.GgufValueType.U32, 2u);
        Emit("general.alignment", GgufFile.GgufValueType.U32, (uint)Align);
        Emit($"{arch}.vocab_size", GgufFile.GgufValueType.U32, (uint)cfg.VocabSize);
        Emit($"{arch}.context_length", GgufFile.GgufValueType.U32, (uint)cfg.MaxCtx);
        Emit($"{arch}.embedding_length", GgufFile.GgufValueType.U32, (uint)cfg.HiddenSize);
        Emit($"{arch}.block_count", GgufFile.GgufValueType.U32, (uint)cfg.NumLayers);
        Emit($"{arch}.feed_forward_length", GgufFile.GgufValueType.U32, (uint)cfg.FfnDim);
        Emit($"{arch}.attention.head_count", GgufFile.GgufValueType.U32, (uint)cfg.NumHeads);
        Emit($"{arch}.attention.head_count_kv", GgufFile.GgufValueType.U32, (uint)cfg.EffectiveKvHeads);
        Emit($"{arch}.attention.layer_norm_rms_epsilon", GgufFile.GgufValueType.F32, cfg.RmsEps);
        Emit($"{arch}.rope.freq_base", GgufFile.GgufValueType.F32, cfg.RopeTheta);
        if (cfg.IsMoe)
        {
            Emit($"{arch}.expert_count", GgufFile.GgufValueType.U32, (uint)cfg.NumExperts);
            Emit($"{arch}.expert_used_count", GgufFile.GgufValueType.U32, (uint)cfg.NumActiveExperts);
        }

        // preserve every other KV of the source gguf (tokenizer, chat template, rope scaling, ...)
        int restored = 0;
        foreach (var (key, type, value) in sourceKvs)
            if (!overrideKeys.Contains(key)) { finalKvs.Add((key, type, value)); restored++; }

        // ---------------------------------------------------------------- write file
        using (var fs = new FileStream(outGgufPath, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20))
        using (var bw = new BinaryWriter(fs))
        {
            bw.Write('G'); bw.Write('G'); bw.Write('U'); bw.Write('F');
            bw.Write(3u);                                    // version
            bw.Write((ulong)outs.Count);
            bw.Write((ulong)finalKvs.Count);

            foreach (var (key, type, value) in finalKvs)
                WriteKv(bw, key, type, value);

            foreach (var t in outs)
            {
                WriteGgufString(bw, t.Name);
                bw.Write(t.Ne.Length);
                foreach (var d in t.Ne) bw.Write((ulong)d);
                bw.Write((uint)t.GgmlType);
                bw.Write((ulong)t.Offset);
            }

            long dataStart = (fs.Position + Align - 1) / Align * Align;
            while (fs.Position < dataStart) bw.Write((byte)0);

            foreach (var t in outs)
            {
                while (fs.Position - dataStart < t.Offset) bw.Write((byte)0);   // per-tensor 32 B alignment
                WriteTensorData(store, t, bw);
            }
            bw.Flush();
        }
        sw.Stop();

        long total = new FileInfo(outGgufPath).Length;
        Console.WriteLine($"  tensors      : {outs.Count} ({string.Join(", ", typeCounts.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key} x{kv.Value}"))})");
        Console.WriteLine($"  output size  : {Fmt(total)}");
        Console.WriteLine($"  time         : {sw.Elapsed.TotalSeconds:F1}s  ({Fmt((long)(total / Math.Max(0.1, sw.Elapsed.TotalSeconds)))}/s)");
        if (restored > 0) Console.WriteLine($"  metadata     : {restored} KV(s) restored from the .qw header (tokenizer + arch)");
        else Console.WriteLine("  metadata     : weights only — the source .qw has no embedded tokenizer (was not imported from GGUF)");
        foreach (var (name, why) in skipped.Take(8)) Console.WriteLine($"  skipped      : {name} — {why}");
        if (skipped.Count > 8) Console.WriteLine($"                 ... and {skipped.Count - 8} more");
    }

    // ------------------------------------------------------------------ plan

    private sealed class GgufOutTensor
    {
        public required string Name;
        public required long[] Ne;
        public required int GgmlType;
        public required long Offset;
        public required (string QwName, int ExpertIdx)[] Parts;   // ExpertIdx = -1 for plain tensors
        public long NumElements => Ne.Aggregate(1L, (acc, d) => acc * d);
        public required long DataBytes;
        public required string Kind;                              // pass | f16 | f32 | q8_0 | ggml_q4_0
    }

    private static List<GgufOutTensor> BuildTensorPlan(ModelStore store, GgufExportOptions opt,
        List<(string, string)> skipped, Dictionary<string, int> typeCounts)
    {
        QuantFormat? forced = opt.GgmlType?.ToLowerInvariant() switch
        {
            "auto" or "" or null => null,
            "f16" or "fp16" => QuantFormat.F16,
            "f32" or "fp32" => QuantFormat.F32,
            "q8_0" or "q8" => QuantFormat.Q8_0,
            _ => throw new ArgumentException($"unknown --ggml-type '{opt.GgmlType}' (auto|f16|f32|q8_0)")
        };

        // group experts: l{i}.e{j}.w_{gate,up,down} -> one 3-D gguf tensor per (layer, matrix)
        var experts = new Dictionary<(int layer, string which), List<(int j, TensorView t)>>();
        var used = new HashSet<string>();
        foreach (var t in store.Tensors.Values)
        {
            var m = System.Text.RegularExpressions.Regex.Match(t.Name, @"^l(\d+)\.e(\d+)\.w_(gate|up|down)$");
            if (m.Success)
            {
                var key = (int.Parse(m.Groups[1].Value), m.Groups[3].Value);
                if (!experts.TryGetValue(key, out var list)) experts[key] = list = new();
                list.Add((int.Parse(m.Groups[2].Value), t));
                used.Add(t.Name);
            }
        }

        var plan = new List<GgufOutTensor>();
        long offset = 0;

        (int ggmlType, string kind) Decide(TensorView t)
        {
            if (forced != null && t.Dims.Length == 2 && t.Format.IsQuantized())
                return forced.Value switch
                {
                    QuantFormat.F32 => (0, "f32"),
                    QuantFormat.Q8_0 => (8, "q8_0"),
                    _ => (1, "f16")
                };
            if (t.Dims.Length == 1 || !t.Format.IsQuantized())
                return t.Format switch { QuantFormat.F16 => (1, "pass"), QuantFormat.BF16 => (30, "pass"), _ => (0, "pass") };
            return t.Format switch
            {
                QuantFormat.F32 => (0, "pass"),
                QuantFormat.F16 => (1, "pass"),
                QuantFormat.BF16 => (30, "pass"),
                QuantFormat.Q8_0 => (8, "pass"),
                QuantFormat.Q2_K => (10, "pass"),
                QuantFormat.Q3_K => (11, "pass"),
                QuantFormat.Q4_K => (12, "pass"),
                QuantFormat.Q5_K => (13, "pass"),
                QuantFormat.Q6_K => (14, "pass"),
                QuantFormat.Q4_0 => (2, "ggml_q4_0"),   // re-encode to the true ggml 18 B layout
                _ => (1, "f16")                          // TRI / B1 / B1X / T1E / T1E_LO / I8_BLK
            };
        }

        void Add(string ggufName, long[] ne, int ggmlType, string kind, (string, int)[] parts)
        {
            long elems = ne.Aggregate(1L, (acc, d) => acc * d);
            long dataBytes = kind switch
            {
                "pass" => parts.Sum(p => Dtypes.TensorBytes(TensorElems(store.Get(p.Item1)), store.Get(p.Item1).Format)),
                "f16" => elems * 2,
                "f32" => elems * 4,
                "q8_0" => Dtypes.TensorBytes(elems, QuantFormat.Q8_0),
                "ggml_q4_0" => elems / 32 * 18,
                _ => throw new InvalidOperationException(kind)
            };
            long off = (offset + Align - 1) / Align * Align;
            plan.Add(new GgufOutTensor
            {
                Name = ggufName, Ne = ne, GgmlType = ggmlType, Offset = off,
                Parts = parts, DataBytes = dataBytes, Kind = kind
            });
            offset = off + dataBytes;
            string label = GgufFile.GgmlTypeName(ggmlType) + (kind != "pass" ? "*" : "");
            typeCounts[label] = (typeCounts.TryGetValue(label, out var c) ? c : 0) + 1;
        }

        foreach (var t in OrderedTensors(store))
        {
            if (used.Contains(t.Name)) continue;
            if (TryGgufName(t.Name, t.Dims, out string? ggufName, out long[]? ne, out string? why))
            {
                var (gt, kind) = Decide(t);
                // ggml row constraints: K-quants need ne[0] % 256; byte formats ne[0] % 32
                if ((gt is >= 10 and <= 14 && ne[0] % 256 != 0) || (gt is 2 or 8 && ne[0] % 32 != 0))
                {
                    skipped.Add((ggufName, $"{GgufFile.GgmlTypeName(gt)} needs aligned rows — exporting as F16 instead"));
                    (gt, kind) = (1, "f16");
                }
                Add(ggufName, ne, gt, kind, new[] { (t.Name, -1) });
            }
            else
            {
                skipped.Add((t.Name, why ?? "no gguf mapping"));
            }
        }

        foreach (var ((layer, which), list) in experts.OrderBy(kv => kv.Key.layer))
        {
            list.Sort((x, y) => x.j.CompareTo(y.j));
            var first = list[0].t;
            // per-expert .qw dims are [rows, cols]; gguf 3-D ne = {cols, rows, n_experts}
            long[] ne = first.Dims.Length == 2
                ? new[] { first.Dims[1], first.Dims[0], list.Count }
                : new[] { first.Dims[0], list.Count };
            var (gt, kind) = Decide(first);
            if ((gt is >= 10 and <= 14 && ne[0] % 256 != 0) || (gt is 2 or 8 && ne[0] % 32 != 0)) (gt, kind) = (1, "f16");
            Add($"blk.{layer}.ffn_{which}_exps.weight", ne, gt, kind, list.Select(e => (e.t.Name, e.j)).ToArray());
        }

        return plan;
    }

    private static long TensorElems(TensorView t) { long n = 1; foreach (var d in t.Dims) n *= d; return n; }

    /// <summary>Deterministic tensor order: tok_emb, layers 0..N (attn -> ffn), final_norm, lm_head.</summary>
    private static IEnumerable<TensorView> OrderedTensors(ModelStore store)
    {
        int Rank(string name)
        {
            if (name == "tok_emb") return 0;
            if (name == "final_norm") return 1_000_000_000;
            if (name == "lm_head") return 1_000_000_001;
            var m = System.Text.RegularExpressions.Regex.Match(name, @"^l(\d+)\.(.+)$");
            if (!m.Success) return 2_000_000_000;
            int layer = int.Parse(m.Groups[1].Value);
            string rest = m.Groups[2].Value;
            int sub = rest switch
            {
                "attn_norm" => 0,
                "mla.qanorm" => 1, "mla.qa" => 2, "mla.qb" => 3, "mla.kvanorm" => 4, "mla.kva" => 5, "mla.kvb" => 6, "mla.o" => 7,
                "kda.qconv" => 1, "kda.kconv" => 2, "kda.vconv" => 3, "kda.q" => 4, "kda.k" => 5, "kda.v" => 6, "kda.o" => 7,
                "kda.b" => 8, "kda.fa" => 9, "kda.fb" => 10, "kda.ga" => 11, "kda.gb" => 12, "kda.alog" => 13, "kda.dt" => 14, "kda.onorm" => 15,
                "wq" => 20, "wk" => 21, "wv" => 22, "wo" => 23,
                "ffn_norm" => 30,
                "router" => 40, "router_bias" => 41,
                "sw_gate" => 42, "sw_up" => 43, "sw_down" => 44,
                "w_gate" => 45, "w_up" => 46, "w_down" => 47,
                _ => 50
            };
            if (System.Text.RegularExpressions.Regex.Match(rest, @"^e(\d+)\.w_(gate|up|down)$").Success) sub = 60;   // grouped separately
            if (rest.StartsWith("hc_", StringComparison.Ordinal)) sub = 70;
            return 100_000_000 + layer * 1000 + sub;
        }
        return store.Tensors.Values.OrderBy(t => Rank(t.Name)).ThenBy(t => t.Name, StringComparer.Ordinal);
    }

    /// <summary>Canonical .qw name -> gguf tensor name + gguf ne dims (ne[0] = fastest dim = cols).
    /// Returns false with a reason for tensors without a llama.cpp-side equivalent.</summary>
    private static bool TryGgufName(string name, long[] dims, out string? ggufName, out long[]? ne, out string? why)
    {
        ggufName = null; ne = null; why = null;
        if (name == "tok_emb") { ggufName = "token_embd.weight"; ne = DimsNe(dims); return true; }
        if (name == "lm_head") { ggufName = "output.weight"; ne = DimsNe(dims); return true; }
        if (name == "final_norm") { ggufName = "output_norm.weight"; ne = dims; return true; }

        var m = System.Text.RegularExpressions.Regex.Match(name, @"^l(\d+)\.(.+)$");
        if (!m.Success) { why = "not a layer tensor"; return false; }
        int i = int.Parse(m.Groups[1].Value);
        string b = $"blk.{i}.";
        switch (m.Groups[2].Value)
        {
            case "attn_norm": ggufName = b + "attn_norm.weight"; ne = dims; return true;
            case "ffn_norm": ggufName = b + "ffn_norm.weight"; ne = dims; return true;
            case "wq": ggufName = b + "attn_q.weight"; ne = DimsNe(dims); return true;
            case "wk": ggufName = b + "attn_k.weight"; ne = DimsNe(dims); return true;
            case "wv": ggufName = b + "attn_v.weight"; ne = DimsNe(dims); return true;
            case "wo": ggufName = b + "attn_o.weight"; ne = DimsNe(dims); return true;
            case "router": ggufName = b + "ffn_gate_inp.weight"; ne = DimsNe(dims); return true;
            case "router_bias": ggufName = b + "exp_probs_b.bias"; ne = dims; return true;
            case "sw_gate": ggufName = b + "ffn_gate_shexp.weight"; ne = DimsNe(dims); return true;
            case "sw_up": ggufName = b + "ffn_up_shexp.weight"; ne = DimsNe(dims); return true;
            case "sw_down": ggufName = b + "ffn_down_shexp.weight"; ne = DimsNe(dims); return true;
            case "w_gate": ggufName = b + "ffn_gate.weight"; ne = DimsNe(dims); return true;
            case "w_up": ggufName = b + "ffn_up.weight"; ne = DimsNe(dims); return true;
            case "w_down": ggufName = b + "ffn_down.weight"; ne = DimsNe(dims); return true;
            // MLA (DeepSeek-V3 / GLM-5.3 style) — standard gguf deepseek2 tensor names
            case "mla.qa": ggufName = b + "attn_q_a.weight"; ne = DimsNe(dims); return true;
            case "mla.qanorm": ggufName = b + "attn_q_a_norm.weight"; ne = dims; return true;
            case "mla.qb": ggufName = b + "attn_q_b.weight"; ne = DimsNe(dims); return true;
            case "mla.kva": ggufName = b + "attn_kv_a_mqa.weight"; ne = DimsNe(dims); return true;
            case "mla.kvanorm": ggufName = b + "attn_kv_a_norm.weight"; ne = dims; return true;
            case "mla.kvb": ggufName = b + "attn_kv_b.weight"; ne = DimsNe(dims); return true;
            case "mla.o": ggufName = b + "attn_o.weight"; ne = DimsNe(dims); return true;
            default:
                if (m.Groups[2].Value.StartsWith("kda.", StringComparison.Ordinal))
                    why = "KDA (linear attention) has no ggml/llama.cpp counterpart";
                else if (m.Groups[2].Value.StartsWith("hc_", StringComparison.Ordinal))
                    why = "mHC hyper-connection params have no ggml/llama.cpp counterpart";
                else
                    why = "no gguf mapping";
                return false;
            }

        static long[] DimsNe(long[] d) => d.Length == 2 ? new[] { d[1], d[0] } : d;
    }

    // ------------------------------------------------------------------ data writing

    private static void WriteTensorData(ModelStore store, GgufOutTensor t, BinaryWriter bw)
    {
        const long CHUNK = 1 << 20;
        if (t.Kind == "pass")
        {
            foreach (var (qwName, _) in t.Parts)
                bw.Write(store.Get(qwName).Bytes);
            return;
        }

        long chunkElems = CHUNK;
        var f32 = new float[CHUNK];
        foreach (var (qwName, _) in t.Parts)
        {
            var view = store.Get(qwName);
            long n = TensorElems(view);
            var (bs, bb) = Dtypes.BlockSpec(view.Format);
            long chunk = chunkElems - chunkElems % bs;
            long done = 0;
            while (done < n)
            {
                long elems = Math.Min(chunk, n - done);
                long blocks = elems / bs;
                var src = view.Bytes.Slice((int)(done / bs * bb), (int)(blocks * bb));
                BlockDequantizer.Dequantize(view.Format, src, f32.AsSpan(0, (int)elems));
                switch (t.Kind)
                {
                    case "f16":
                    {
                        var bytes = new byte[elems * 2];
                        unsafe
                        {
                            fixed (float* pf = f32)
                            fixed (byte* pd = bytes)
                            {
                                ushort* o = (ushort*)pd;
                                for (int i = 0; i < elems; i++) o[i] = Dtypes.FloatToF16(pf[i]);
                            }
                        }
                        bw.Write(bytes);
                        break;
                    }
                    case "f32":
                    {
                        var bytes = new byte[elems * 4];
                        Buffer.BlockCopy(f32, 0, bytes, 0, bytes.Length);
                        bw.Write(bytes);
                        break;
                    }
                    case "q8_0":
                    {
                        var bytes = new byte[Dtypes.TensorBytes(elems, QuantFormat.Q8_0)];
                        BlockQuantizer.Quantize(QuantFormat.Q8_0, f32.AsSpan(0, (int)elems), bytes);
                        bw.Write(bytes);
                        break;
                    }
                    case "ggml_q4_0":
                    {
                        var bytes = new byte[elems / 32 * 18];
                        GgmlDequant.QuantizeGgmlQ4_0(f32.AsSpan(0, (int)elems), bytes);
                        bw.Write(bytes);
                        break;
                    }
                    default: throw new InvalidOperationException(t.Kind);
                }
                done += elems;
            }
        }
    }

    // ------------------------------------------------------------------ metadata

    private static long ComputeFileType(List<GgufOutTensor> plan)
    {
        // dominant 2-D weight type -> LLAMA_FTYPE_* label (advisory metadata only)
        var hist = new Dictionary<int, int>();
        foreach (var t in plan)
            if (t.Name.EndsWith(".weight", StringComparison.Ordinal) && t.Ne.Length == 2)
                hist[t.GgmlType] = (hist.TryGetValue(t.GgmlType, out var c) ? c : 0) + 1;
        int dom = hist.OrderByDescending(kv => kv.Value).FirstOrDefault().Key;
        return dom switch
        {
            0 => 0, 1 => 1, 2 => 2, 3 => 3, 6 => 8, 7 => 9, 8 => 7,
            10 => 10, 11 => 12, 12 => 15, 13 => 17, 14 => 18, 30 => 1,
            _ => 1
        };
    }

    /// <summary>Ordered KV list embedded at gguf -> qw time (GGUF wire format preserved in the header).
    /// Also returns the source general.architecture when present.</summary>
    internal static (List<(string Key, GgufFile.GgufValueType Type, object Value)> Ordered, string? Arch)
        ParseEmbeddedKvs(JsonElement config)
    {
        var ordered = new List<(string, GgufFile.GgufValueType, object)>();
        string? arch = null;
        if (config.ValueKind == JsonValueKind.Object && config.TryGetProperty("gguf_kvs", out var el) &&
            el.ValueKind == JsonValueKind.Object && el.TryGetProperty("b64", out var b64) && b64.GetString() is string b64s)
        {
            long nKv = el.TryGetProperty("n_kv", out var n) ? n.GetInt64() : 0;
            byte[] blob = Convert.FromBase64String(b64s);
            using var ms = new MemoryStream(blob);
            using var br = new BinaryReader(ms, Encoding.UTF8, leaveOpen: true);
            for (long i = 0; i < nKv && ms.Position < ms.Length; i++)
            {
                string key = GgufFile.ReadString(br);
                var type = (GgufFile.GgufValueType)br.ReadUInt32();
                object value = GgufFile.ReadValue(br, type);
                ordered.Add((key, type, value));
                if (key == "general.architecture" && value is string s) arch = s;
            }
        }
        return (ordered, arch);
    }

    private static void WriteKv(BinaryWriter bw, string key, GgufFile.GgufValueType type, object value)
    {
        WriteGgufString(bw, key);
        bw.Write((uint)type);
        switch (type)
        {
            case GgufFile.GgufValueType.U8: bw.Write(Convert.ToByte(value)); break;
            case GgufFile.GgufValueType.I8: bw.Write(Convert.ToSByte(value)); break;
            case GgufFile.GgufValueType.U16: bw.Write(Convert.ToUInt16(value)); break;
            case GgufFile.GgufValueType.I16: bw.Write(Convert.ToInt16(value)); break;
            case GgufFile.GgufValueType.U32: bw.Write(Convert.ToUInt32(value)); break;
            case GgufFile.GgufValueType.I32: bw.Write(Convert.ToInt32(value)); break;
            case GgufFile.GgufValueType.U64: bw.Write(Convert.ToUInt64(value)); break;
            case GgufFile.GgufValueType.I64: bw.Write(Convert.ToInt64(value)); break;
            case GgufFile.GgufValueType.F32: bw.Write(Convert.ToSingle(value)); break;
            case GgufFile.GgufValueType.F64: bw.Write(Convert.ToDouble(value)); break;
            case GgufFile.GgufValueType.Bool: bw.Write(Convert.ToBoolean(value)); break;
            case GgufFile.GgufValueType.String: WriteGgufString(bw, (string)value); break;
            case GgufFile.GgufValueType.Array:
                WriteArrayValue(bw, value);
                break;
            default: throw new InvalidDataException($"cannot write gguf kv type {type}");
        }
    }

    private static void WriteArrayValue(BinaryWriter bw, object value)
    {
        switch (value)
        {
            case string[] a:
                bw.Write((uint)GgufFile.GgufValueType.String);
                bw.Write((ulong)a.Length);
                foreach (var s in a) WriteGgufString(bw, s);
                break;
            case float[] a:
                bw.Write((uint)GgufFile.GgufValueType.F32);
                bw.Write((ulong)a.Length);
                foreach (var f in a) bw.Write(f);
                break;
            case double[] a:
                bw.Write((uint)GgufFile.GgufValueType.F64);
                bw.Write((ulong)a.Length);
                foreach (var f in a) bw.Write(f);
                break;
            case int[] a:
                bw.Write((uint)GgufFile.GgufValueType.I32);
                bw.Write((ulong)a.Length);
                foreach (var f in a) bw.Write(f);
                break;
            case uint[] a:
                bw.Write((uint)GgufFile.GgufValueType.U32);
                bw.Write((ulong)a.Length);
                foreach (var f in a) bw.Write(f);
                break;
            case ulong[] a:
                bw.Write((uint)GgufFile.GgufValueType.U64);
                bw.Write((ulong)a.Length);
                foreach (var f in a) bw.Write(f);
                break;
            case long[] a:
                bw.Write((uint)GgufFile.GgufValueType.I64);
                bw.Write((ulong)a.Length);
                foreach (var f in a) bw.Write(f);
                break;
            default:   // u8/i8/u16/i16/bool arrays read back as long[]
                bw.Write((uint)GgufFile.GgufValueType.I32);
                var arr = (long[])value;
                bw.Write((ulong)arr.Length);
                foreach (var f in arr) bw.Write((int)f);
                break;
        }
    }

    private static void WriteGgufString(BinaryWriter bw, string s)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        bw.Write((ulong)bytes.Length);
        bw.Write(bytes);
    }

    private static string Fmt(long b) => b switch
    {
        >= 1L << 40 => $"{b / (double)(1L << 40):F2} TiB",
        >= 1L << 30 => $"{b / (double)(1L << 30):F2} GiB",
        >= 1L << 20 => $"{b / (double)(1L << 20):F1} MiB",
        >= 1024 => $"{b / 1024.0:F1} KiB",
        _ => $"{b} B"
    };
}
