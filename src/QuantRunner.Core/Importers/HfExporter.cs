using System.Text;
using System.Text.Json;
using QuantRunner.Core.Memory;
using QuantRunner.Core.Models;
using QuantRunner.Core.Quantization;

namespace QuantRunner.Core.Importers;

/// <summary>Options for exporting a .qw / .gguf model to a HuggingFace-style directory.</summary>
public sealed class HfExportOptions
{
    /// <summary>Split into N roughly equal shards (tensor-granular, never split a tensor).</summary>
    public int Shards = 1;
    /// <summary>Alternative to <see cref="Shards"/>: group transformer layers M per shard
    /// (embeddings in shard 0, final norm + lm_head in the last shard). &gt; 0 wins over Shards.</summary>
    public int LayersPerShard;
    /// <summary>f16 (default) | f32 | bf16 — dequantized output precision.</summary>
    public string DType = "f16";
}

/// <summary>
/// Exports a QuantRunner .qw model (or a GGUF file) to a HuggingFace-ready directory:
///
///   outdir/
///     model-00001-of-000NN.safetensors ...   (dequantized weights, __metadata__ format=pt)
///     model.safetensors.index.json           (weight_map + total_size, when &gt; 1 shard)
///     config.json                            (architecture + dims, ready for transformers)
///     generation_config.json                 (bos/eos when known)
///     tokenizer.json / tokenizer_config.json / special_tokens_map.json   (when a tokenizer
///                                            is available: GGUF source, or a .qw imported
///                                            from GGUF which embeds the original metadata)
///
/// Weights are DEQUANTIZED to the requested dtype — a .qw is a lossy container, so the
/// export restores portability (fine-tuning, training, transformers/vLLM), not the original
/// fp16 checkpoint. 2-D tensors keep the HF [out, in] layout the .qw already stores, so no
/// transpose is needed. KDA / mHC tensors have no HF counterpart and are skipped with a note.
/// </summary>
public static class HfExporter
{
    private const long CHUNK = 1 << 20;   // elements per dequant chunk

    public static void FromQw(string qwPath, string outDir, HfExportOptions? options = null)
    {
        var opt = options ?? new HfExportOptions();
        using var store = ModelStore.Open(qwPath);
        var cfg = ModelConfig.FromJson(store.ConfigJson);
        var (_, srcArch) = GgufExporter.ParseEmbeddedKvs(store.ConfigJson);

        var dtype = ParseDtype(opt.DType);
        Console.WriteLine($"QuantRunner convert: {Path.GetFileName(qwPath)} -> {outDir}/ (safetensors, {opt.DType})");
        var sw = System.Diagnostics.Stopwatch.StartNew();

        var (kvs, _) = GgufExporter.ParseEmbeddedKvs(store.ConfigJson);
        ExportCore(EntriesFromQw(store), cfg, srcArch, KvsToDict(kvs), outDir, opt, dtype, sw);
    }

    public static void FromGguf(string ggufPath, string outDir, HfExportOptions? options = null)
    {
        var opt = options ?? new HfExportOptions();
        using var gguf = GgufFile.Open(ggufPath);
        var cfg = ModelConverter.ConfigFromGguf(gguf, new ConvertOptions(), ggufPath);

        var dtype = ParseDtype(opt.DType);
        Console.WriteLine($"QuantRunner convert: {Path.GetFileName(ggufPath)} -> {outDir}/ (safetensors, {opt.DType})");
        var sw = System.Diagnostics.Stopwatch.StartNew();

        ExportCore(EntriesFromGguf(gguf), cfg, gguf.Kv("general.architecture") as string, gguf.Kvs, outDir, opt, dtype, sw);
    }

    // ------------------------------------------------------------------ entries

    private sealed class HfEntry
    {
        public required string Name;          // HF tensor name
        public required long[] Shape;
        public required long NumElements;
        public TensorView? Qw;                // .qw source
        public GgufFile? Gguf;                // gguf source
        public GgufFile.GgufTensorInfo? Info;
        public long GgufElemOffset;           // expert split offset (elements)
        public int Shard;
    }

    private static List<HfEntry> EntriesFromQw(ModelStore store)
    {
        var entries = new List<HfEntry>();
        var skipped = new List<(string, string)>();
        foreach (var t in OrderedQwTensors(store))
        {
            if (!ReverseHfName(t.Name, out string? hfName, out string? why))
            {
                skipped.Add((t.Name, why!));
                continue;
            }
            long n = 1; foreach (var d in t.Dims) n *= d;
            entries.Add(new HfEntry { Name = hfName!, Shape = t.Dims, NumElements = n, Qw = t });
        }
        ReportSkipped(skipped);
        return entries;
    }

    private static List<HfEntry> EntriesFromGguf(GgufFile gguf)
    {
        var entries = new List<HfEntry>();
        var skipped = new List<(string, string)>();
        var notes = new Dictionary<string, int>();

        foreach (var t in gguf.Tensors)
        {
            var mapped = NameMapper.Map(NameMapper.Source.Gguf, t.Name, t.Dims, int.MaxValue, out string note);
            if (note != "") notes[note] = (notes.TryGetValue(note, out var c) ? c : 0) + 1;
            if (mapped == null) { skipped.Add((t.Name, note == "" ? "not recognized" : note)); continue; }

            if (mapped.StartsWith("EXPSPLIT:", StringComparison.Ordinal))
            {
                var parts = mapped.Split(':');
                int layer = int.Parse(parts[1]);
                string which = parts[2];                    // gate | up | down
                if (t.Dims.Length != 3) throw new InvalidDataException($"expert tensor {t.Name} expected 3-D");
                long perExpert = t.Dims[0] * t.Dims[1];
                for (int e = 0; e < t.Dims[2]; e++)
                {
                    if (!ReverseHfName($"l{layer}.e{e}.w_{which}", out string? hfName, out _)) continue;
                    entries.Add(new HfEntry
                    {
                        Name = hfName!, Shape = new[] { t.Dims[1], t.Dims[0] },
                        NumElements = perExpert, Gguf = gguf, Info = t, GgufElemOffset = e * perExpert
                    });
                }
                continue;
            }

            if (!ReverseHfName(mapped, out string? name2, out _)) { skipped.Add((mapped, "no HF equivalent")); continue; }
            // gguf ne[0] is the fastest dim: 2-D [K, M] -> HF [M, K]; data order is unchanged
            var shape = t.Dims.Length == 2 ? new[] { t.Dims[1], t.Dims[0] } : t.Dims.ToArray();
            entries.Add(new HfEntry { Name = name2!, Shape = shape, NumElements = t.NumElements, Gguf = gguf, Info = t });
        }
        ReportSkipped(skipped);
        foreach (var kv in notes.Take(4)) Console.WriteLine($"  note     : {kv.Value}x {kv.Key}");
        return entries;
    }

    private static IEnumerable<TensorView> OrderedQwTensors(ModelStore store)
    {
        int Rank(string name)
        {
            if (name == "tok_emb") return 0;
            if (name == "final_norm") return 1_000_000_000;
            if (name == "lm_head") return 1_000_000_001;
            var m = System.Text.RegularExpressions.Regex.Match(name, @"^l(\d+)\.");
            if (!m.Success) return 2_000_000_000;
            return 100_000_000 + int.Parse(m.Groups[1].Value) * 1000;
        }
        return store.Tensors.Values.OrderBy(t => Rank(t.Name)).ThenBy(t => t.Name, StringComparer.Ordinal);
    }

    /// <summary>Canonical .qw name -> HuggingFace name (reverse of NameMapper).</summary>
    internal static bool ReverseHfName(string canonical, out string? hfName, out string? why)
    {
        hfName = null; why = null;
        switch (canonical)
        {
            case "tok_emb": hfName = "model.embed_tokens.weight"; return true;
            case "lm_head": hfName = "lm_head.weight"; return true;
            case "final_norm": hfName = "model.norm.weight"; return true;
        }
        var m = System.Text.RegularExpressions.Regex.Match(canonical, @"^l(\d+)\.(.+)$");
        if (!m.Success) { why = "no HF mapping"; return false; }
        int i = int.Parse(m.Groups[1].Value);
        string p = $"model.layers.{i}.";
        switch (m.Groups[2].Value)
        {
            case "attn_norm": hfName = p + "input_layernorm.weight"; return true;
            case "ffn_norm": hfName = p + "post_attention_layernorm.weight"; return true;
            case "wq": hfName = p + "self_attn.q_proj.weight"; return true;
            case "wk": hfName = p + "self_attn.k_proj.weight"; return true;
            case "wv": hfName = p + "self_attn.v_proj.weight"; return true;
            case "wo": hfName = p + "self_attn.o_proj.weight"; return true;
            case "w_gate": hfName = p + "mlp.gate_proj.weight"; return true;
            case "w_up": hfName = p + "mlp.up_proj.weight"; return true;
            case "w_down": hfName = p + "mlp.down_proj.weight"; return true;
            case "router": hfName = p + "mlp.gate.weight"; return true;
            case "router_bias": hfName = p + "mlp.gate.e_score_correction_bias"; return true;
            case "sw_gate": hfName = p + "mlp.shared_experts.gate_proj.weight"; return true;
            case "sw_up": hfName = p + "mlp.shared_experts.up_proj.weight"; return true;
            case "sw_down": hfName = p + "mlp.shared_experts.down_proj.weight"; return true;
            // MLA — DeepSeek-V3 style names round-trip exactly
            case "mla.qa": hfName = p + "self_attn.q_a_proj.weight"; return true;
            case "mla.qanorm": hfName = p + "self_attn.q_a_layernorm.weight"; return true;
            case "mla.qb": hfName = p + "self_attn.q_b_proj.weight"; return true;
            case "mla.kva": hfName = p + "self_attn.kv_a_proj_with_mqa.weight"; return true;
            case "mla.kvanorm": hfName = p + "self_attn.kv_a_layernorm.weight"; return true;
            case "mla.kvb": hfName = p + "self_attn.kv_b_proj.weight"; return true;
            case "mla.o": hfName = p + "self_attn.o_proj.weight"; return true;
        }
        var e = System.Text.RegularExpressions.Regex.Match(m.Groups[2].Value, @"^e(\d+)\.w_(gate|up|down)$");
        if (e.Success)
        {
            string w = e.Groups[2].Value switch { "gate" => "gate", "up" => "up", _ => "down" };
            hfName = $"{p}mlp.experts.{e.Groups[1].Value}.{w}_proj.weight";
            return true;
        }
        if (m.Groups[2].Value.StartsWith("kda.", StringComparison.Ordinal)) { why = "KDA (linear attention) has no standard HF counterpart"; return false; }
        if (m.Groups[2].Value.StartsWith("hc_", StringComparison.Ordinal)) { why = "mHC hyper-connection params have no standard HF counterpart"; return false; }
        why = "no HF mapping";
        return false;
    }

    // ------------------------------------------------------------------ sharding + write

    private static void ExportCore(List<HfEntry> entries, ModelConfig cfg, string? srcArch,
        Dictionary<string, object> kvs, string outDir, HfExportOptions opt,
        (string dtype, string torchDtype, int elemSize) dtypeInfo, System.Diagnostics.Stopwatch sw)
    {
        Directory.CreateDirectory(outDir);
        AssignShards(entries, opt);

        int nShards = entries.Max(e => e.Shard) + 1;
        var weightMap = new Dictionary<string, string>();
        long totalSize = 0;

        for (int s = 0; s < nShards; s++)
        {
            var shardEntries = entries.Where(e => e.Shard == s).ToList();
            string fileName = nShards == 1 ? "model.safetensors" : $"model-{s + 1:D5}-of-{nShards:D5}.safetensors";
            string filePath = Path.Combine(outDir, fileName);

            // header with byte offsets (safetensors packs tensors back-to-back, no padding)
            long offset = 0;
            var header = new Dictionary<string, object>
            {
                ["__metadata__"] = new Dictionary<string, string> { ["format"] = "pt" }
            };
            foreach (var e in shardEntries)
            {
                long bytes = e.NumElements * dtypeInfo.elemSize;
                header[e.Name] = new Dictionary<string, object>
                {
                    ["dtype"] = dtypeInfo.dtype,
                    ["shape"] = e.Shape,
                    ["data_offsets"] = new[] { offset, offset + bytes }
                };
                offset += bytes;
                weightMap[e.Name] = fileName;
            }
            totalSize += offset;

            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20);
            using var bw = new BinaryWriter(fs);
            var headerBytes = JsonSerializer.SerializeToUtf8Bytes(header);
            bw.Write((ulong)headerBytes.Length);
            bw.Write(headerBytes);

            foreach (var e in shardEntries)
                WriteTensorData(e, dtypeInfo, bw);
            bw.Flush();
        }

        if (nShards > 1)
            File.WriteAllText(Path.Combine(outDir, "model.safetensors.index.json"),
                JsonSerializer.Serialize(new Dictionary<string, object>
                {
                    ["metadata"] = new Dictionary<string, object> { ["total_size"] = totalSize },
                    ["weight_map"] = weightMap
                }, new JsonSerializerOptions { WriteIndented = true }));

        WriteConfigFiles(outDir, cfg, srcArch, kvs, dtypeInfo, entries);
        sw.Stop();

        Console.WriteLine($"  tensors      : {entries.Count} in {nShards} shard(s)");
        Console.WriteLine($"  output size  : {Fmt(totalSize)}  ({Path.GetFileName(outDir.TrimEnd('/'))}/)");
        Console.WriteLine($"  time         : {sw.Elapsed.TotalSeconds:F1}s  ({Fmt((long)(totalSize / Math.Max(0.1, sw.Elapsed.TotalSeconds)))}/s)");
        bool hasTokenizer = WriteTokenizer(outDir, kvs, cfg, srcArch);
        if (!hasTokenizer)
            Console.WriteLine("  note       : no tokenizer available (source has no embedded GGUF metadata) — copy tokenizer files from the original HF repo for training");
    }

    private static void AssignShards(List<HfEntry> entries, HfExportOptions opt)
    {
        if (opt.LayersPerShard > 0)
        {
            int maxLayer = -1;
            foreach (var e in entries)
            {
                var m = System.Text.RegularExpressions.Regex.Match(e.Name, @"^model\.layers\.(\d+)\.");
                if (m.Success) maxLayer = Math.Max(maxLayer, int.Parse(m.Groups[1].Value));
            }
            int lastGroup = Math.Max(0, maxLayer / opt.LayersPerShard);
            foreach (var e in entries)
            {
                var m = System.Text.RegularExpressions.Regex.Match(e.Name, @"^model\.layers\.(\d+)\.");
                e.Shard = m.Success
                    ? int.Parse(m.Groups[1].Value) / opt.LayersPerShard
                    : e.Name.StartsWith("model.embed_tokens", StringComparison.Ordinal) ? 0 : lastGroup;
            }
            return;
        }
        if (opt.Shards <= 1) return;

        long total = entries.Sum(e => e.NumElements) * (opt.DType.ToLowerInvariant() switch { "f32" => 4, _ => 2 });
        long target = total / opt.Shards + 1;
        int shard = 0;
        long acc = 0;
        foreach (var e in entries)
        {
            e.Shard = shard;
            acc += e.NumElements * (opt.DType.ToLowerInvariant() switch { "f32" => 4, _ => 2 });
            if (acc >= target && shard < opt.Shards - 1) { shard++; acc = 0; }
        }
    }

    private static unsafe void WriteTensorData(HfEntry e, (string dtype, string torchDtype, int elemSize) dt, BinaryWriter bw)
    {
        if (e.Qw != null)
        {
            var view = e.Qw;
            long n = e.NumElements;
            var (bs, bb) = Dtypes.BlockSpec(view.Format);
            var f32 = new float[Math.Min(CHUNK, n)];
            long done = 0;
            while (done < n)
            {
                long elems = Math.Min(CHUNK - CHUNK % bs, n - done);
                long blocks = elems / bs;
                var src = view.Bytes.Slice((int)(done / bs * bb), (int)(blocks * bb));
                BlockDequantizer.Dequantize(view.Format, src, f32.AsSpan(0, (int)elems));
                EmitChunk(f32, (int)elems, dt, bw);
                done += elems;
            }
            return;
        }

        // gguf source (optionally a per-expert element range of a 3-D tensor)
        var info = e.Info!;
        var gguf = e.Gguf!;
        var (bs2, bb2) = GgufFile.GgmlBlockSpec(info.GgmlType);
        var buf = new byte[Math.Min(CHUNK, e.NumElements) / bs2 * bb2 + 8];
        var f32b = new float[Math.Min(CHUNK, e.NumElements)];
        long done2 = 0;
        while (done2 < e.NumElements)
        {
            long elems = Math.Min(CHUNK - CHUNK % bs2, e.NumElements - done2);
            long blocks = elems / bs2;
            long blockOff = (e.GgufElemOffset + done2) / bs2;
            gguf.Stream.Position = gguf.DataStart + info.Offset + blockOff * bb2;
            int total = (int)(blocks * bb2), got = 0;
            while (got < total)
            {
                int r = gguf.Stream.Read(buf, got, total - got);
                if (r <= 0) throw new EndOfStreamException($"gguf tensor {info.Name}");
                got += r;
            }
            GgmlDequant.Dequant(info.GgmlType, buf.AsSpan(0, total), f32b.AsSpan(0, (int)elems));
            EmitChunk(f32b, (int)elems, dt, bw);
            done2 += elems;
        }
    }

    private static void EmitChunk(float[] f32, int n, (string dtype, string torchDtype, int elemSize) dt, BinaryWriter bw)
    {
        switch (dt.dtype)
        {
            case "F32":
            {
                var bytes = new byte[n * 4];
                Buffer.BlockCopy(f32, 0, bytes, 0, n * 4);
                bw.Write(bytes);
                break;
            }
            case "F16":
            {
                var bytes = new byte[n * 2];
                unsafe
                {
                    fixed (float* pf = f32)
                    fixed (byte* pd = bytes)
                    {
                        ushort* o = (ushort*)pd;
                        for (int i = 0; i < n; i++) o[i] = Dtypes.FloatToF16(pf[i]);
                    }
                }
                bw.Write(bytes);
                break;
            }
            case "BF16":
            {
                var bytes = new byte[n * 2];
                unsafe
                {
                    fixed (float* pf = f32)
                    fixed (byte* pd = bytes)
                    {
                        ushort* o = (ushort*)pd;
                        for (int i = 0; i < n; i++) o[i] = Dtypes.FloatToBf16(pf[i]);
                    }
                }
                bw.Write(bytes);
                break;
            }
            default: throw new InvalidOperationException(dt.dtype);
        }
    }

    // ------------------------------------------------------------------ config / tokenizer

    private static void WriteConfigFiles(string outDir, ModelConfig cfg, string? srcArch,
        Dictionary<string, object> kvs, (string dtype, string torchDtype, int elemSize) dt,
        List<HfEntry> entries)
    {
        bool hasLmHead = entries.Any(e => e.Name == "lm_head.weight");
        string arch = srcArch ?? "llama";
        var names = ArchNames(arch);
        var config = new Dictionary<string, object?>
        {
            ["architectures"] = new[] { names.Item1 },
            ["model_type"] = names.Item2,
            ["hidden_size"] = cfg.HiddenSize,
            ["intermediate_size"] = cfg.FfnDim,
            ["num_hidden_layers"] = cfg.NumLayers,
            ["num_attention_heads"] = cfg.NumHeads,
            ["num_key_value_heads"] = cfg.EffectiveKvHeads,
            ["vocab_size"] = cfg.VocabSize,
            ["rms_norm_eps"] = cfg.RmsEps,
            ["rope_theta"] = cfg.RopeTheta,
            ["max_position_embeddings"] = cfg.MaxCtx,
            ["tie_word_embeddings"] = !hasLmHead,
            ["torch_dtype"] = dt.torchDtype,
            ["attention_bias"] = false,
            ["use_cache"] = true
        };
        if (cfg.IsMoe)
        {
            config["num_local_experts"] = cfg.NumExperts;
            config["num_experts_per_tok"] = cfg.NumActiveExperts;
            if (cfg.ExpertFfnDim > 0 && cfg.ExpertFfnDim != cfg.FfnDim) config["moe_intermediate_size"] = cfg.ExpertFfnDim;
        }
        long? bos = KvsLong(kvs, "tokenizer.ggml.bos_token_id");
        long? eos = KvsLong(kvs, "tokenizer.ggml.eos_token_id");
        long? pad = KvsLong(kvs, "tokenizer.ggml.pad_token_id");
        if (bos != null) config["bos_token_id"] = bos;
        if (eos != null) config["eos_token_id"] = eos;
        if (pad != null) config["pad_token_id"] = pad;

        File.WriteAllText(Path.Combine(outDir, "config.json"),
            JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true }));

        if (bos != null || eos != null)
        {
            var gen = new Dictionary<string, object?>();
            if (bos != null) gen["bos_token_id"] = bos;
            if (eos != null) gen["eos_token_id"] = eos;
            gen["do_sample"] = true;
            gen["max_length"] = cfg.MaxCtx;
            File.WriteAllText(Path.Combine(outDir, "generation_config.json"),
                JsonSerializer.Serialize(gen, new JsonSerializerOptions { WriteIndented = true }));
        }
    }

    /// <summary>gguf architecture -> (architectures[0], model_type). Unknown arches fall back to
    /// a Pascal-cased ForCausalLM guess with the raw arch as model_type.</summary>
    internal static (string, string) ArchNames(string arch) => arch.ToLowerInvariant() switch
    {
        "llama" => ("LlamaForCausalLM", "llama"),
        "qwen2" => ("Qwen2ForCausalLM", "qwen2"),
        "qwen3" => ("Qwen3ForCausalLM", "qwen3"),
        "qwen2moe" => ("Qwen2MoeForCausalLM", "qwen2_moe"),
        "qwen3moe" => ("Qwen3MoeForCausalLM", "qwen3_moe"),
        "mistral" => ("MistralForCausalLM", "mistral"),
        "deepseek2" => ("DeepseekV2ForCausalLM", "deepseek_v2"),
        "glm4" => ("Glm4ForCausalLM", "glm4"),
        "glm4moe" => ("Glm4MoeForCausalLM", "glm4_moe"),
        "gemma3" => ("Gemma3ForCausalLM", "gemma3_text"),
        "gemma3_text" => ("Gemma3ForCausalLM", "gemma3_text"),
        "phi3" => ("Phi3ForCausalLM", "phi3"),
        "olmo2" => ("Olmo2ForCausalLM", "olmo2"),
        _ => (char.ToUpperInvariant(arch[0]) + arch[1..] + "ForCausalLM", arch)
    };

    /// <summary>Builds a HF tokenizer.json string (BPE "gpt2" models) from GGUF metadata KVs.
    /// Returns null when the KVs carry no tokenizer, or when the vocabulary is sentencepiece
    /// (the binary SPM model cannot be reconstructed from GGUF metadata alone).</summary>
    public static string? BuildTokenizerJson(Dictionary<string, object> kvs)
    {
        if (!kvs.TryGetValue("tokenizer.ggml.tokens", out var tokObj) || tokObj is not string[] tokens)
            return null;

        string model = kvs.TryGetValue("tokenizer.ggml.model", out var m) && m is string ms ? ms : "gpt2";
        if (model != "gpt2" && model != "bpe") return null;

        long[]? tokenTypes = kvs.TryGetValue("tokenizer.ggml.token_type", out var tt) && tt is System.Collections.IEnumerable ie
            ? ie.Cast<object>().Select(Convert.ToInt64).ToArray() : null;
        long? unkId = KvsLong(kvs, "tokenizer.ggml.unknown_token_id");
        string? unkTok = unkId != null && unkId >= 0 && unkId < tokens.Length ? tokens[unkId.Value] : null;

        var merges = kvs.TryGetValue("tokenizer.ggml.merges", out var mg) && mg is string[] mrg ? mrg : Array.Empty<string>();

        // special tokens: ggml token_type 1=UNKNOWN 2=CONTROL 3=USER (5=BYTE is not special)
        var added = new List<object>();
        for (int i = 0; i < tokens.Length; i++)
        {
            int t = tokenTypes != null && i < tokenTypes.Length ? (int)tokenTypes[i] : 0;
            if (t is 1 or 2 or 3)
            {
                added.Add(new Dictionary<string, object>
                {
                    ["id"] = i, ["content"] = tokens[i], ["single_word"] = false,
                    ["lstrip"] = false, ["rstrip"] = false, ["normalized"] = false, ["special"] = true
                });
            }
        }

        var vocab = new Dictionary<string, int>();
        for (int i = 0; i < tokens.Length; i++) vocab[tokens[i]] = i;

        var tokenizerJson = new Dictionary<string, object?>
        {
            ["version"] = "1.0",
            ["truncation"] = null,
            ["padding"] = null,
            ["added_tokens"] = added,
            ["normalizer"] = null,
            ["pre_tokenizer"] = new Dictionary<string, object>
            {
                ["type"] = "ByteLevel", ["add_prefix_space"] = true, ["trim_offsets"] = true, ["use_regex"] = true
            },
            ["post_processor"] = new Dictionary<string, object>
            {
                ["type"] = "ByteLevel", ["add_prefix_space"] = true, ["trim_offsets"] = false, ["use_regex"] = true
            },
            ["decoder"] = new Dictionary<string, object>
            {
                ["type"] = "ByteLevel", ["add_prefix_space"] = true, ["trim_offsets"] = true, ["use_regex"] = true
            },
            ["model"] = new Dictionary<string, object?>
            {
                ["type"] = "BPE", ["dropout"] = null, ["unk_token"] = unkTok,
                ["continuing_subword_prefix"] = null, ["end_of_word_suffix"] = null,
                ["fuse_unk"] = false, ["byte_fallback"] = false,
                ["vocab"] = vocab, ["merges"] = merges
            }
        };
        return JsonSerializer.Serialize(tokenizerJson);
    }

    /// <summary>Writes tokenizer.json (BPE "gpt2" models) or tokenizer_config.json (SPM "llama"
    /// models — the sentencepiece binary cannot be reconstructed from GGUF metadata). Returns
    /// true when a tokenizer was written.</summary>
    internal static bool WriteTokenizer(string outDir, Dictionary<string, object> kvs,
        ModelConfig cfg, string? srcArch)
    {
        if (!kvs.TryGetValue("tokenizer.ggml.tokens", out var tokObj) || tokObj is not string[] tokens)
            return false;

        string model = kvs.TryGetValue("tokenizer.ggml.model", out var m) && m is string ms ? ms : "gpt2";
        long[]? tokenTypes = kvs.TryGetValue("tokenizer.ggml.token_type", out var tt) && tt is System.Collections.IEnumerable ie
            ? ie.Cast<object>().Select(Convert.ToInt64).ToArray() : null;
        string? chatTemplate = kvs.TryGetValue("tokenizer.chat_template", out var ct) && ct is string cts ? cts : null;
        long? bosId = KvsLong(kvs, "tokenizer.ggml.bos_token_id");
        long? eosId = KvsLong(kvs, "tokenizer.ggml.eos_token_id");
        long? padId = KvsLong(kvs, "tokenizer.ggml.pad_token_id");

        string? TokOf(long? id) => id != null && id >= 0 && id < tokens.Length ? tokens[id.Value] : null;
        var bosTok = TokOf(bosId); var eosTok = TokOf(eosId); var unkTok = TokOf(KvsLong(kvs, "tokenizer.ggml.unknown_token_id")); var padTok = TokOf(padId);

        var tc = new Dictionary<string, object?>
        {
            ["model_max_length"] = cfg.MaxCtx > 0 ? cfg.MaxCtx : 4096,
            ["clean_up_tokenization_spaces"] = false,
            ["extra_special_tokens"] = new Dictionary<string, object>()
        };
        if (bosTok != null) tc["bos_token"] = bosTok;
        if (eosTok != null) tc["eos_token"] = eosTok;
        if (unkTok != null) tc["unk_token"] = unkTok;
        if (padTok != null) tc["pad_token"] = padTok;
        if (chatTemplate != null) tc["chat_template"] = chatTemplate;
        if (srcArch != null) tc["tokenizer_class"] = srcArch.ToLowerInvariant() switch
        {
            "qwen2" or "qwen2moe" => "Qwen2Tokenizer",
            "qwen3" or "qwen3moe" => "Qwen2Tokenizer",
            "llama" or "mistral" => "LlamaTokenizer",
            _ => "PreTrainedTokenizerFast"
        };

        if (model == "gpt2" || model == "bpe")
        {
            string? tokenizerJson = BuildTokenizerJson(kvs);
            if (tokenizerJson == null) return false;
            File.WriteAllText(Path.Combine(outDir, "tokenizer.json"), tokenizerJson);
            tc["tokenizer_class"] = tc.TryGetValue("tokenizer_class", out var cls) ? cls : "PreTrainedTokenizerFast";
            File.WriteAllText(Path.Combine(outDir, "tokenizer_config.json"),
                JsonSerializer.Serialize(tc, new JsonSerializerOptions { WriteIndented = true }));
            var specials = new List<string>();
            for (int i = 0; i < tokens.Length; i++)
            {
                int t = tokenTypes != null && i < tokenTypes.Length ? (int)tokenTypes[i] : 0;
                if (t is 1 or 2 or 3) specials.Add(tokens[i]);
            }
            var stm = new Dictionary<string, object?>();
            if (bosTok != null) stm["bos_token"] = bosTok;
            if (eosTok != null) stm["eos_token"] = eosTok;
            if (unkTok != null) stm["unk_token"] = unkTok;
            if (padTok != null) stm["pad_token"] = padTok;
            if (specials.Except(new[] { bosTok, eosTok, unkTok, padTok }).Any())
                stm["additional_special_tokens"] = specials.Where(s => s != bosTok && s != eosTok && s != unkTok && s != padTok).ToArray();
            File.WriteAllText(Path.Combine(outDir, "special_tokens_map.json"),
                JsonSerializer.Serialize(stm, new JsonSerializerOptions { WriteIndented = true }));
            return true;
        }

        // sentencepiece (llama SPM): config files only, plus an honest note
        File.WriteAllText(Path.Combine(outDir, "tokenizer_config.json"),
            JsonSerializer.Serialize(tc, new JsonSerializerOptions { WriteIndented = true }));
        var stm2 = new Dictionary<string, object?>();
        if (bosTok != null) stm2["bos_token"] = bosTok;
        if (eosTok != null) stm2["eos_token"] = eosTok;
        if (unkTok != null) stm2["unk_token"] = unkTok;
        if (padTok != null) stm2["pad_token"] = padTok;
        File.WriteAllText(Path.Combine(outDir, "special_tokens_map.json"),
            JsonSerializer.Serialize(stm2, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine("  note       : sentencepiece tokenizer detected — wrote tokenizer_config.json only; " +
                          "grab tokenizer.model / tokenizer.json from the original HF repo for exact parity");
        return true;
    }

    private static long? KvsLong(Dictionary<string, object> kvs, string key) =>
        kvs.TryGetValue(key, out var v) && v is IConvertible c ? Convert.ToInt64(c) : null;

    private static Dictionary<string, object> KvsToDict(List<(string Key, GgufFile.GgufValueType Type, object Value)> list)
    {
        var d = new Dictionary<string, object>();
        foreach (var (key, _, value) in list) d[key] = value;
        return d;
    }

    private static (string dtype, string torchDtype, int elemSize) ParseDtype(string s) => s.ToLowerInvariant() switch
    {
        "f16" or "fp16" or "float16" or "half" => ("F16", "float16", 2),
        "bf16" or "bfloat16" => ("BF16", "bfloat16", 2),
        "f32" or "fp32" or "float32" or "float" => ("F32", "float32", 4),
        _ => throw new ArgumentException($"unknown --dtype '{s}' (f16|f32|bf16)")
    };

    private static void ReportSkipped(List<(string, string)> skipped)
    {
        foreach (var (name, why) in skipped.Take(8)) Console.WriteLine($"  skipped      : {name} — {why}");
        if (skipped.Count > 8) Console.WriteLine($"                 ... and {skipped.Count - 8} more");
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
