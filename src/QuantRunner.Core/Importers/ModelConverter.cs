using System.Text.Json;
using System.Text.RegularExpressions;
using QuantRunner.Core.Memory;
using QuantRunner.Core.Models;
using QuantRunner.Core.Quantization;

namespace QuantRunner.Core.Importers;

public sealed class ConvertOptions
{
    public QuantFormat Attention = QuantFormat.Q4_K;
    public QuantFormat Ffn = QuantFormat.Q4_K;
    public QuantFormat Emb = QuantFormat.F16;
    public QuantFormat Head = QuantFormat.F16;     // lm_head
    public QuantFormat Norm = QuantFormat.F32;
    public long MinQuantElements = 4096;           // tensors smaller than this stay fp
    public int MaxCtx = -1;                        // override context
    public long HeaderReserveBytes = 0;            // >0: reserve header space for later --append
    public bool KeepSourceFormats;                 // gguf import: keep every tensor in its own
                                                   // ggml format (layout-identical types pass
                                                   // through byte-exact; legacy types requant to
                                                   // the same precision) — used by `run model.gguf`

    // architecture overrides for shard quantize (applied after config.json / shape inference)
    public int? OverrideLayers { get; set; }
    public int? OverrideHeads { get; set; }
    public int? OverrideKvHeads { get; set; }
    public int? OverrideHeadDim { get; set; }
    public int? OverrideDim { get; set; }
    public int? OverrideFfn { get; set; }
    public int? OverrideVocab { get; set; }
    public int? OverrideExperts { get; set; }
    public int? OverrideActive { get; set; }

    public bool HasOverrides => OverrideLayers != null || OverrideHeads != null || OverrideKvHeads != null ||
                                OverrideHeadDim != null || OverrideDim != null || OverrideFfn != null ||
                                OverrideVocab != null || OverrideExperts != null || OverrideActive != null;

    public void ApplyOverrides(ModelConfig cfg)
    {
        if (OverrideLayers is int l) cfg.NumLayers = l;
        if (OverrideHeads is int h) cfg.NumHeads = h;
        if (OverrideKvHeads is int kv) cfg.NumKvHeads = kv;
        if (OverrideHeadDim is int hd) cfg.HeadDim = hd;
        if (OverrideDim is int d) cfg.HiddenSize = d;
        if (OverrideFfn is int f) cfg.FfnDim = f;
        if (OverrideVocab is int v) cfg.VocabSize = v;
        if (OverrideExperts is int e)
        {
            cfg.NumExperts = e;
            if (e > 0 && cfg.NumActiveExperts <= 0) cfg.NumActiveExperts = 2;
            if (e > 0 && cfg.Arch != "moe") cfg.Arch = "moe";
        }
        if (OverrideActive is int a) cfg.NumActiveExperts = a;
        if (MaxCtx > 0) cfg.MaxCtx = MaxCtx;
    }

    /// <summary>Routing groups shared by Pick (conversion) and the requantizer. Sensitive =
    /// projections kept in fp because quantization hurts them disproportionately
    /// (KDA gate/beta, mHC fn matvec) — the requantizer never touches those groups.</summary>
    public enum QuantGroup { Attention, Ffn, Emb, Head, Norm, Sensitive }

    public QuantGroup Group(string canonical)
    {
        if (canonical == "tok_emb") return QuantGroup.Emb;
        if (canonical == "lm_head") return QuantGroup.Head;
        if (canonical.EndsWith("norm")) return QuantGroup.Norm;
        if (canonical.EndsWith("router") || canonical.EndsWith("router_bias")) return QuantGroup.Norm;
        if (canonical.Contains(".hc_")) return canonical.EndsWith("_fn") ? QuantGroup.Sensitive : QuantGroup.Norm;   // mHC: fn matvec fp, base/scale f32
        if (canonical.Contains(".kda."))
        {
            if (canonical.EndsWith(".q") || canonical.EndsWith(".k") || canonical.EndsWith(".v") || canonical.EndsWith(".o"))
                return QuantGroup.Attention;
            if (canonical.EndsWith(".fa") || canonical.EndsWith(".fb") || canonical.EndsWith(".ga") ||
                canonical.EndsWith(".gb") || canonical.EndsWith(".b")) return QuantGroup.Sensitive;   // gate/beta projections: sensitive, keep fp
            return QuantGroup.Norm;                                                                  // alog/dt/onorm/*conv
        }
        if (canonical.Contains(".mla."))
        {
            if (canonical.EndsWith(".qanorm") || canonical.EndsWith(".kvanorm")) return QuantGroup.Norm;
            return QuantGroup.Attention;                                                             // qa/qb/kva/kvb/o
        }
        if (canonical.Contains(".sw_")) return QuantGroup.Ffn;                                       // shared expert
        if (canonical.Contains(".w_gate") || canonical.Contains(".w_up") || canonical.Contains(".w_down")) return QuantGroup.Ffn;
        return QuantGroup.Attention; // wq/wk/wv/wo
    }

    public QuantFormat Pick(string canonical) => Group(canonical) switch
    {
        QuantGroup.Attention => Attention,
        QuantGroup.Ffn => Ffn,
        QuantGroup.Emb => Emb,
        QuantGroup.Head => Head,
        QuantGroup.Norm => Norm,
        QuantGroup.Sensitive => QuantFormat.F16,
        _ => Norm
    };

    public static QuantFormat ParseQuant(string s) => s.ToLowerInvariant() switch
    {
        "f32" or "fp32" => QuantFormat.F32,
        "f16" or "fp16" or "half" => QuantFormat.F16,
        "bf16" => QuantFormat.BF16,
        "q4_0" => QuantFormat.Q4_0,
        "q4k" or "q4_k" or "q4_k_m" => QuantFormat.Q4_K,
        "q2k" or "q2_k" => QuantFormat.Q2_K,
        "q3k" or "q3_k" => QuantFormat.Q3_K,
        "q5k" or "q5_k" => QuantFormat.Q5_K,
        "q6k" or "q6_k" => QuantFormat.Q6_K,
        "q8" or "q8_0" or "int8" => QuantFormat.Q8_0,
        "i8" or "i8blk" or "i8_blk" => QuantFormat.I8_BLK,
        "tri" or "ternary" or "1.58" or "bitnet" => QuantFormat.TRI,
        "b1" or "binary" or "1bit" => QuantFormat.B1,
        "b1x" or "binaryx" => QuantFormat.B1X,
        "t1e" => QuantFormat.T1E,
        "t1e_lo" or "t1elo" => QuantFormat.T1E_LO,
        _ => throw new ArgumentException($"unknown quant '{s}'")
    };
}

public sealed record ConvertReport(int TensorCount, long TotalBytes, long OriginalBytes, double Seconds, ModelConfig Config);

/// <summary>Converts GGUF files or HuggingFace safetensors directories into streaming .qw models.</summary>
public static class ModelConverter
{
    // ------------------------------------------------------------------ GGUF

    public static ConvertReport ConvertGguf(string ggufPath, string outPath, ConvertOptions opt)
    {
        using var gguf = GgufFile.Open(ggufPath);
        var cfg = ConfigFromGguf(gguf, opt, ggufPath);
        var (plan, source) = PlanFromGguf(gguf, cfg, opt);
        return WritePlan(plan, gguf, source, null, outPath, cfg, GgufKvsConfigJson(gguf));
    }

    /// <summary>Serializes the source GGUF metadata (raw KV section, verbatim) for embedding in
    /// the .qw header, so qw -> gguf / qw -> safetensors exports can restore the tokenizer and
    /// architecture metadata losslessly instead of guessing them from shapes.</summary>
    internal static string? GgufKvsConfigJson(GgufFile gguf)
    {
        byte[] kv = gguf.ReadKvSection();
        if (kv.Length == 0) return null;
        return JsonSerializer.Serialize(new Dictionary<string, object>
        {
            ["v"] = gguf.Version,
            ["n_kv"] = gguf.KvCount,
            ["b64"] = Convert.ToBase64String(kv)
        });
    }

    internal static ModelConfig ConfigFromGguf(GgufFile gguf, ConvertOptions opt, string ggufPath)
    {
        string arch = gguf.Kv("general.architecture") as string ?? "llama";
        if (arch.Contains("deepseek") && gguf.Kv($"{arch}.attention.key_length") is object)
            throw new NotSupportedException(
                "DeepSeek-V2/V3 style MLA attention is not supported in this version. " +
                "Mixtral-style MoE (Llama attention + grouped experts) is fully supported.");

        var cfg = new ModelConfig
        {
            HiddenSize = (int)gguf.KvLong($"{arch}.embedding_length", gguf.KvLong("llama.embedding_length")),
            NumLayers = (int)gguf.KvLong($"{arch}.block_count", gguf.KvLong("llama.block_count")),
            NumHeads = (int)gguf.KvLong($"{arch}.attention.head_count", gguf.KvLong("llama.attention.head_count")),
            NumKvHeads = (int)gguf.KvLong($"{arch}.attention.head_count_kv", gguf.KvLong("llama.attention.head_count_kv", -1)),
            FfnDim = gguf.KvLong($"{arch}.feed_forward_length", gguf.KvLong("llama.feed_forward_length")),
            VocabSize = (int)gguf.KvLong($"{arch}.vocab_size", gguf.KvLong("tokenizer.ggml.tokens_count")),
            RopeTheta = gguf.KvFloat($"{arch}.rope.freq_base", gguf.KvFloat("llama.rope.freq_base", 10000f)),
            RmsEps = gguf.KvFloat($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f),
            MaxCtx = opt.MaxCtx > 0 ? opt.MaxCtx : (int)gguf.KvLong($"{arch}.context_length", 4096),
            SourceDescription = Path.GetFileName(ggufPath)
        };
        long expertCount = gguf.KvLong($"{arch}.expert_count", gguf.KvLong("llama.expert_count", 0));
        if (expertCount > 0)
        {
            cfg.NumExperts = (int)expertCount;
            cfg.NumActiveExperts = (int)gguf.KvLong($"{arch}.expert_used_count", gguf.KvLong("llama.expert_used_count", 2));
            cfg.Arch = "moe";
        }
        if (cfg.NumKvHeads <= 0) cfg.NumKvHeads = cfg.NumHeads;
        return cfg;
    }

    /// <summary>ggml type -> our closest format at the SAME precision, for as-is imports
    /// (`run model.gguf` must not silently change the model's quantization). Legacy ggml
    /// layouts (18/22-byte Q4_0/Q5_0, Q4_1/Q5_1 with min) have no byte-identical counterpart
    /// and requant to the matching depth: Q4_0/Q4_1 -> Q4_0, Q5_0/Q5_1 -> Q5_K.</summary>
    internal static QuantFormat GgmlAsIs(int ggmlType) => ggmlType switch
    {
        0 => QuantFormat.F32, 1 => QuantFormat.F16, 30 => QuantFormat.BF16,
        2 => QuantFormat.Q4_0, 3 => QuantFormat.Q4_0,
        6 => QuantFormat.Q5_K, 7 => QuantFormat.Q5_K,
        8 => QuantFormat.Q8_0,
        10 => QuantFormat.Q2_K, 11 => QuantFormat.Q3_K, 12 => QuantFormat.Q4_K,
        13 => QuantFormat.Q5_K, 14 => QuantFormat.Q6_K,
        _ => throw new NotSupportedException(
            $"ggml type {ggmlType} ({GgufFile.GgmlTypeName(ggmlType)}) cannot be run as-is — " +
            "convert it to .qw or gguf F16/Q8_0 first (unsupported types like IQ*/TQ*)")
    };

    private static (List<TensorPlan> plan, Dictionary<string, GgufFile.GgufTensorInfo> source) PlanFromGguf(
        GgufFile gguf, ModelConfig cfg, ConvertOptions opt)
    {
        var plan = new List<TensorPlan>();
        var source = new Dictionary<string, GgufFile.GgufTensorInfo>();

        foreach (var t in gguf.Tensors)
        {
            var mapped = NameMapper.Map(NameMapper.Source.Gguf, t.Name, t.Dims, cfg.NumLayers, out string note);
            if (note != "") Console.WriteLine($"  note: {note}");
            if (mapped == null) continue;

            if (mapped.StartsWith("EXPSPLIT:"))
            {
                var parts = mapped.Split(':');
                int layer = int.Parse(parts[1]);
                string which = parts[2];                        // gate/up/down
                if (t.Dims.Length != 3)
                    throw new InvalidDataException($"expert tensor {t.Name} expected 3-D, got {t.Dims.Length}-D");
                int nExpert = (int)t.Dims[2];
                if (cfg.NumExperts == 0)
                {
                    cfg.NumExperts = nExpert;
                    cfg.NumActiveExperts = Math.Max(2, cfg.NumActiveExperts);
                    cfg.Arch = "moe";
                }
                if (nExpert != cfg.NumExperts) throw new InvalidDataException($"expert count mismatch in {t.Name}");
                // GGUF ne: gate/up = {dim, ffn, n_exp}, down = {ffn, dim, n_exp}. Normalize to [rows, cols].
                long rows = t.Dims[1], cols = t.Dims[0];
                for (int e = 0; e < nExpert; e++)
                {
                    string cname = $"l{layer}.e{e}.w_{which}";
                    plan.Add(new TensorPlan(cname, new[] { rows, cols },
                        opt.KeepSourceFormats ? GgmlAsIs(t.GgmlType) : opt.Pick(cname)));
                    source[cname] = t;      // expert index resolved during streaming
                }
                continue;
            }

            long n = t.NumElements;
            var f = opt.KeepSourceFormats ? GgmlAsIs(t.GgmlType) : opt.Pick(mapped);
            if (!opt.KeepSourceFormats && n < opt.MinQuantElements && f.IsQuantized()) f = GGMLToMine(t.GgmlType, QuantFormat.F32);
            // GGUF ne[0] is the fastest dim (cols); normalize every 2-D weight to [rows, cols].
            var dims = t.Dims.Length == 2 ? new[] { t.Dims[1], t.Dims[0] } : t.Dims.ToArray();
            f = SafeQuantForRows(f, dims[^1]);   // ggml-aligned passthrough passes unchanged
            plan.Add(new TensorPlan(mapped, dims, f));
            source[mapped] = t;
        }
        var e0 = plan.FirstOrDefault(p => p.Name.EndsWith(".e0.w_gate") || p.Name.EndsWith(".e0.w_up"));
        if (e0 != null) cfg.ExpertFfnDim = e0.Dims[0];
        return (plan, source);
    }

    private static QuantFormat GGMLToMine(int ggmlType, QuantFormat fallback) => ggmlType switch
    {
        0 => QuantFormat.F32, 1 => QuantFormat.F16, 2 => QuantFormat.Q4_0, 8 => QuantFormat.Q8_0,
        10 => QuantFormat.Q2_K, 11 => QuantFormat.Q3_K, 12 => QuantFormat.Q4_K, 13 => QuantFormat.Q5_K,
        14 => QuantFormat.Q6_K, 30 => QuantFormat.BF16,
        _ => fallback
    };

    /// <summary>
    /// Dot kernels walk quantization blocks along each row: a row must be a whole number of
    /// blocks, or every subsequent row misaligns (engine outputs NaN/garbage). q4_0/q8_0
    /// (32-wide) and TRI (64-wide) divide all realistic dims; K-quants (256-wide) do not —
    /// e.g. an ffn of 128. Returns a row-safe substitute (q8_0, else f32) when the requested
    /// format cannot align with the row length.
    /// </summary>
    public static QuantFormat SafeQuantForRows(QuantFormat f, long rowElems, List<string>? log = null, string? tensorName = null)
    {
        if (!f.IsQuantized()) return f;
        int block = Dtypes.BlockSpec(f).blockSize;
        if (rowElems % block == 0) return f;
        var sub = rowElems % 32 == 0 ? QuantFormat.Q8_0 : QuantFormat.F32;
        log?.Add($"{tensorName} ({f} -> {sub})");
        return sub;
    }

    private static void PrintRowFallbacks(List<string> fallbacks)
    {
        if (fallbacks.Count == 0) return;
        var sample = string.Join(", ", fallbacks.Take(2)) + (fallbacks.Count > 2 ? $", +{fallbacks.Count - 2} more" : "");
        Console.WriteLine(
            $"  note     : {fallbacks.Count} tensor(s) stored with a row-safe format — row length not a multiple of the quant block: {sample}");
    }

    // ------------------------------------------------------------------ HuggingFace safetensors

    /// <summary>Convert a HuggingFace directory (config.json + *.safetensors) into a .qw model.</summary>
    public static ConvertReport ConvertHfDir(string dirPath, string outPath, ConvertOptions opt)
    {
        var files = Directory.GetFiles(dirPath, "*.safetensors");
        if (files.Length == 0) throw new FileNotFoundException($"no .safetensors files in {dirPath}");
        string configJson = Path.Combine(dirPath, "config.json");
        if (!File.Exists(configJson)) throw new FileNotFoundException($"config.json not found in {dirPath}");
        var cfg = ConfigFromHfJson(File.ReadAllText(configJson), opt, Path.GetFileName(dirPath.TrimEnd('/')));
        return ConvertSafeTensors(files, configJson, outPath, opt);
    }

    /// <summary>
    /// Quantize one .safetensors file (or several shard files) into a streaming .qw model.
    /// Config resolution order: explicit configPath -> config.json beside the first input file
    /// -> inferred from tensor names/shapes (works for a single self-contained file).
    ///
    /// appendToExisting: add the tensors of these files into an existing .qw (reserved header
    /// required) instead of creating a new one — sharded checkpoints are assembled shard by
    /// shard, existing tensor data is never copied. Tensors already present are skipped
    /// (dims must match); the arch config comes from the existing file and overrides are ignored.
    /// </summary>
    public static ConvertReport ConvertSafeTensors(IReadOnlyList<string> filePaths, string? configPath,
        string outPath, ConvertOptions opt, bool appendToExisting = false)
    {
        if (filePaths.Count == 0) throw new ArgumentException("no input .safetensors files");

        // config.json: explicit -> beside the first input file
        string? resolved = configPath;
        if (resolved == null)
        {
            string beside = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(filePaths[0]))!, "config.json");
            if (File.Exists(beside)) resolved = beside;
        }

        if (appendToExisting && File.Exists(outPath))
            return AppendToQw(filePaths, resolved, outPath, opt);

        // warn when the caller hands us one shard of a sharded checkpoint but not the whole set
        foreach (var path in filePaths)
        {
            var m = Regex.Match(Path.GetFileName(path), @"-(\d+)-of-(\d+)\.safetensors$", RegexOptions.IgnoreCase);
            if (m.Success && int.Parse(m.Groups[1].Value) < int.Parse(m.Groups[2].Value) &&
                filePaths.Count < int.Parse(m.Groups[2].Value))
            {
                Console.WriteLine($"  note: '{Path.GetFileName(path)}' is shard {m.Groups[1].Value} of {m.Groups[2].Value} " +
                                  "— this .qw will contain only the tensors of the files given.");
                Console.WriteLine("           add the remaining shards later with: quantize --append <shard> -o <same output>");
                break;
            }
        }

        var readers = new List<SafeTensorsFile>();
        // canonical name -> owning file + source tensor (+ fp8 scale companion)
        var owner = new Dictionary<string, (SafeTensorsFile File, SafeTensorsFile.TensorInfo Info, SafeTensorsFile.TensorInfo? Scale)>();
        var unmapped = new List<string>();
        var notes = new Dictionary<string, int>();
        try
        {
            foreach (var path in filePaths) readers.Add(SafeTensorsFile.Open(path));
            CollectSources(readers, owner, unmapped, notes);

            ModelConfig cfg;
            if (resolved != null)
            {
                Console.WriteLine($"  config   : {resolved}");
                cfg = ConfigFromHfJson(File.ReadAllText(resolved), opt, Path.GetFileName(resolved));
            }
            else
            {
                Console.WriteLine("  config   : not found — inferring architecture from tensor names/shapes");
                cfg = InferConfigFromTensors(ShapesOf(owner), Path.GetFileName(filePaths[0]));
            }
            EnrichConfigFromShapes(cfg, ShapesOf(owner));
            opt.ApplyOverrides(cfg);

            var entries = ExpandSplits(owner, cfg);
            ValidateConfig(cfg, Path.GetFileName(filePaths[0]));

            var plan = new List<TensorPlan>();
            int outOfRange = 0;
            var rowFallbacks = new List<string>();
            var src = new Dictionary<string, SplitEntry>();
            foreach (var e in entries)
            {
                // an explicit config may describe fewer layers than a shard carries — drop the extras
                if (Regex.IsMatch(e.Name, @"^l(\d+)\.") &&
                    int.Parse(e.Name[1..].Split('.')[0]) >= cfg.NumLayers)
                { outOfRange++; continue; }
                long n = 1; foreach (var d in e.Dims) n *= d;
                var f = opt.Pick(e.Name);
                if (n < opt.MinQuantElements && f.IsQuantized()) f = QuantFormat.F32;
                f = SafeQuantForRows(f, e.Dims.Length >= 2 ? e.Dims[^1] : n, rowFallbacks, e.Name);
                plan.Add(new TensorPlan(e.Name, e.Dims, f));
                src[e.Name] = e;
            }
            if (outOfRange > 0)
                Console.WriteLine($"  note     : {outOfRange} tensor(s) dropped — layer index >= num layers ({cfg.NumLayers})");
            PrintRowFallbacks(rowFallbacks);

            ReportDiagnostics(unmapped, notes);
            long originalBytes = src.Values.Select(e => e.Info).Distinct().Sum(i => i.DataEnd - i.DataStart);
            return WriteSafeTensors(plan, src, originalBytes, outPath, cfg, opt.HeaderReserveBytes);
        }
        finally { foreach (var r in readers) r.Dispose(); }
    }

    /// <summary>Add the tensors of the given .safetensors files into an existing .qw model.</summary>
    private static ConvertReport AppendToQw(IReadOnlyList<string> filePaths, string? configPath,
        string outPath, ConvertOptions opt)
    {
        using var app = new ModelStoreAppender(outPath);
        var cfg = ModelConfig.FromJson(app.ConfigJson);

        // heal a degenerate stored config (e.g. created before 'text_config' nesting was
        // understood, or from --layers overrides alone) before using it
        if (configPath != null)
        {
            var hf = ParseHfConfig(File.ReadAllText(configPath), opt, Path.GetFileName(configPath), out var unsupported);
            int healed = HealConfig(cfg, hf);
            if (healed > 0)
                Console.WriteLine($"  healed   : {healed} config field(s) from {Path.GetFileName(configPath)} " +
                                  $"-> arch={cfg.Arch}, layers={cfg.NumLayers}, dim={cfg.HiddenSize}, " +
                                  $"heads={cfg.NumHeads}, kv={cfg.EffectiveKvHeads}, vocab={cfg.VocabSize}");
            if (unsupported.Count > 0)
                Console.WriteLine(
                    "\n  *** WARNING: this checkpoint uses " + string.Join(" + ", unsupported) + ".\n" +
                    "      QuantRunner cannot run it — the assembled .qw will NOT be runnable\n" +
                    "      even when every shard is added.\n");
        }
        if (opt.HasOverrides)
        {
            opt.ApplyOverrides(cfg);
            Console.WriteLine("  note     : architecture overrides applied to the stored config");
        }
        EnrichConfigFromShapes(cfg, null);   // nothing mapped yet — validation of stored fields only
        ValidateConfig(cfg, $"appending to '{outPath}'");
        Console.WriteLine($"  appending: {outPath} ({app.ExistingCount} tensors present, arch {cfg.Arch} x {cfg.NumLayers} layers)");

        var readers = new List<SafeTensorsFile>();
        var owner = new Dictionary<string, (SafeTensorsFile File, SafeTensorsFile.TensorInfo Info, SafeTensorsFile.TensorInfo? Scale)>();
        var unmapped = new List<string>();
        var notes = new Dictionary<string, int>();
        try
        {
            foreach (var path in filePaths) readers.Add(SafeTensorsFile.Open(path));
            CollectSources(readers, owner, unmapped, notes);

            EnrichConfigFromShapes(cfg, ShapesOf(owner));
            var entries = ExpandSplits(owner, cfg);

            var plan = new List<TensorPlan>();
            int skippedPresent = 0, skippedOutOfRange = 0;
            var rowFallbacks = new List<string>();
            var src = new Dictionary<string, SplitEntry>();
            foreach (var e in entries)
            {
                if (app.Find(e.Name) is TensorPlan existing)
                {
                    if (!existing.Dims.SequenceEqual(e.Dims))
                        throw new InvalidDataException(
                            $"tensor '{e.Name}' already present with dims [{string.Join(',', existing.Dims)}] " +
                            $"but shard has [{string.Join(',', e.Dims)}] — wrong shard for this model?");
                    skippedPresent++;
                    continue;
                }
                if (Regex.IsMatch(e.Name, @"^l(\d+)\.") &&
                    int.Parse(e.Name[1..].Split('.')[0]) >= cfg.NumLayers)
                {
                    skippedOutOfRange++;
                    continue;
                }
                long n = 1; foreach (var d in e.Dims) n *= d;
                var f = opt.Pick(e.Name);
                if (n < opt.MinQuantElements && f.IsQuantized()) f = QuantFormat.F32;
                f = SafeQuantForRows(f, e.Dims.Length >= 2 ? e.Dims[^1] : n, rowFallbacks, e.Name);
                plan.Add(new TensorPlan(e.Name, e.Dims, f));
                src[e.Name] = e;
            }
            PrintRowFallbacks(rowFallbacks);

            foreach (var p in plan) app.Plan(p.Name, p.Dims, p.Format);
            app.ReplaceConfig(cfg);   // persist healed/enriched config, not just apply it in memory
            long originalBytes = src.Values.Select(e => e.Info).Distinct().Sum(i => i.DataEnd - i.DataStart);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            app.Commit((planned, stream) => StreamSourceChunks(planned, src[planned.Name], stream));
            sw.Stop();
            ReportDiagnostics(unmapped, notes);
            Console.WriteLine($"  added    : {plan.Count} tensors, skipped {skippedPresent} already present" +
                              (skippedOutOfRange > 0 ? $", {skippedOutOfRange} out of config layer range" : ""));
            if (plan.Count == 0 && skippedPresent == 0 && skippedOutOfRange == 0)
                Console.WriteLine("  *** NO tensors from this shard could be mapped — see the unrecognized names above.");
            long addedBytes = plan.Sum(p => p.SizeBytes);
            return new ConvertReport(plan.Count, addedBytes, originalBytes, sw.Elapsed.TotalSeconds, cfg);
        }
        finally { foreach (var r in readers) r.Dispose(); }
    }

    /// <summary>Parse a HuggingFace config.json into a ModelConfig. Throws only for features the
    /// engine genuinely cannot run (e.g. cross-layer shared DSA indexers).</summary>
    public static ModelConfig ConfigFromHfJson(string json, ConvertOptions opt, string sourceName)
    {
        var cfg = ParseHfConfig(json, opt, sourceName, out var unsupported);
        if (unsupported.Count > 0)
            throw new NotSupportedException(
                $"{sourceName}: this checkpoint uses {string.Join(" + ", unsupported)} — QuantRunner cannot " +
                "convert it into a runnable .qw.");
        return cfg;
    }

    /// <summary>
    /// Parse a HuggingFace config.json without throwing on unsupported architectures
    /// (they are reported through <paramref name="unsupported"/> instead).
    /// Numeric keys are read from 'text_config' first — multimodal containers (GLM-5.x,
    /// GLM-4V, Qwen-omni, ...) nest the language-model config there — then from the root.
    /// </summary>
    public static ModelConfig ParseHfConfig(string json, ConvertOptions opt, string sourceName, out List<string> unsupported)
    {
        unsupported = new List<string>();
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        JsonElement tc = root.TryGetProperty("text_config", out var textCfg) && textCfg.ValueKind == JsonValueKind.Object
            ? textCfg : root;

        string arch = "llama";
        if (TryFirstArchitecture(root, out var a0) || TryFirstArchitecture(tc, out a0))
            if (a0.Contains("Mixtral") || a0.Contains("MoE") || a0.Contains("Moe")) arch = "moe";

        int GetInt(string name, int def) =>
            tc.TryGetProperty(name, out var v) ? v.GetInt32() :
            root.TryGetProperty(name, out var v2) ? v2.GetInt32() : def;
        long GetLong(string name, long def) =>
            tc.TryGetProperty(name, out var v) ? v.GetInt64() :
            root.TryGetProperty(name, out var v2) ? v2.GetInt64() : def;
        float GetFloat(string name, float def) =>
            tc.TryGetProperty(name, out var v) ? v.GetSingle() :
            root.TryGetProperty(name, out var v2) ? v2.GetSingle() : def;
        bool GetBool(string name, bool def) =>
            tc.TryGetProperty(name, out var b) && b.ValueKind is (JsonValueKind.True or JsonValueKind.False) ? b.GetBoolean() :
            root.TryGetProperty(name, out var b2) && b2.ValueKind is (JsonValueKind.True or JsonValueKind.False) ? b2.GetBoolean() : def;

        var cfg = new ModelConfig
        {
            Arch = arch,
            HiddenSize = GetInt("hidden_size", 0),
            NumLayers = GetInt("num_hidden_layers", 0),
            NumHeads = GetInt("num_attention_heads", 0),
            NumKvHeads = GetInt("num_key_value_heads", -1),
            HeadDim = GetInt("head_dim", -1),
            FfnDim = GetLong("intermediate_size", 0),
            VocabSize = GetInt("vocab_size", 0),
            RopeTheta = GetFloat("rope_theta", 10000f),
            RmsEps = GetFloat("rms_norm_eps", 1e-5f),
            MaxCtx = opt.MaxCtx > 0 ? opt.MaxCtx : GetInt("max_position_embeddings", 4096),
            NumExperts = GetInt("num_local_experts", GetInt("num_experts", GetInt("n_routed_experts", 0))),
            NumActiveExperts = GetInt("num_experts_per_tok", 2),
            TieEmbeddings = GetBool("tie_word_embeddings", false),
            SourceDescription = sourceName
        };
        if (cfg.NumExperts > 0) cfg.Arch = "moe";
        if (cfg.Arch == "moe" && tc.TryGetProperty("moe_intermediate_size", out var moeFfn))
            cfg.ExpertFfnDim = moeFfn.GetInt64();
        if (cfg.NumKvHeads <= 0) cfg.NumKvHeads = cfg.NumHeads;
        if (cfg.Arch == "moe" && cfg.ExpertFfnDim <= 0) cfg.ExpertFfnDim = cfg.FfnDim;

        // ---- per-layer attention kinds: layer_types array (GLM-5.x / DeepSeek-V3.2 hybrids) ----
        var kindByLayer = new Dictionary<int, string>();
        if (tc.TryGetProperty("layer_types", out var lts) && lts.ValueKind == JsonValueKind.Array)
        {
            int idx = 0;
            foreach (var e in lts.EnumerateArray())
            {
                string s = e.GetString() ?? "";
                string kind = s.Contains("linear_attention", StringComparison.Ordinal) ? "kda"
                            : s.Contains("sparse_attention", StringComparison.Ordinal) ||
                              s.Contains("mla", StringComparison.Ordinal) ? "mla"
                            : "gqa";
                kindByLayer[idx++] = kind;
            }
        }
        // DeepSeek-style MLA without a layer_types array: every layer is MLA
        if (GetInt("q_lora_rank", 0) > 0 && kindByLayer.Count == 0 && cfg.NumLayers > 0)
            for (int i = 0; i < cfg.NumLayers; i++) kindByLayer[i] = "mla";
        if (kindByLayer.Count > 0 && cfg.NumLayers > 0)
        {
            cfg.AttnKinds = new string[cfg.NumLayers];
            for (int i = 0; i < cfg.NumLayers; i++)
                cfg.AttnKinds[i] = kindByLayer.TryGetValue(i, out var k) ? k : "gqa";
        }

        // ---- KDA linear-attention parameters (GLM-5.x linear_attn_config) ----
        if (cfg.HasKda && tc.TryGetProperty("linear_attn_config", out var lac) && lac.ValueKind == JsonValueKind.Object)
        {
            int SubInt(string name, int def) => lac.TryGetProperty(name, out var v) ? v.GetInt32() : def;
            cfg.LinearHeads = SubInt("num_heads", 0);
            cfg.LinearHeadDim = SubInt("head_dim", 0);
            cfg.LinearConvKernel = SubInt("short_conv_kernel_size", 4);
            if (lac.TryGetProperty("gate_lower_bound", out var glb) && glb.ValueKind == JsonValueKind.Number)
                cfg.LinearLowerBound = glb.GetSingle();
        }

        // ---- hyper-connections (GLM-5.x mHC) ----
        if (GetBool("mhc", false) || GetInt("hc_mult", 0) > 1)
        {
            cfg.HcMult = GetInt("hc_mult", 0);
            cfg.HcEps = GetFloat("hc_eps", 1e-6f);
            cfg.HcSinkhornIters = GetInt("hc_sinkhorn_iters", 1);
        }

        // ---- MLA dimensions ----
        if (cfg.HasMla)
        {
            cfg.QLoraRank = GetInt("q_lora_rank", 0);
            cfg.KvLoraRank = GetInt("kv_lora_rank", 0);
            cfg.QkNopeHeadDim = GetInt("qk_nope_head_dim", 0);
            cfg.QkRopeHeadDim = GetInt("qk_rope_head_dim", 0);
            cfg.VHeadDim = GetInt("v_head_dim", 0);
        }

        // ---- GLM-style MoE (sigmoid / noaux_tc router, routed scaling, shared expert, limited SwiGLU) ----
        string scoring = tc.TryGetProperty("scoring_func", out var sf) && sf.ValueKind == JsonValueKind.String
            ? sf.GetString() ?? "" : "";
        string topkMethod = tc.TryGetProperty("topk_method", out var tm) && tm.ValueKind == JsonValueKind.String
            ? tm.GetString() ?? "" : "";
        if (scoring == "sigmoid" || topkMethod == "noaux_tc")
        {
            cfg.MoeSigmoid = true;
            cfg.RoutedScale = GetFloat("routed_scaling_factor", 1f);
            cfg.NumSharedExperts = GetInt("n_shared_experts", 0);
        }
        cfg.SwigluLimit = GetFloat("swiglu_limit", 0f);

        // ---- remaining unsupported features ----
        if (tc.TryGetProperty("indexer_types", out var its) && its.ValueKind == JsonValueKind.Array)
            foreach (var e in its.EnumerateArray())
                if ((e.GetString() ?? "") == "shared")
                { unsupported.Add("cross-layer shared DSA indexer (indexer_types contains 'shared')"); break; }
        return cfg;
    }

    private static bool TryFirstArchitecture(JsonElement el, out string first)
    {
        first = "";
        if (el.ValueKind == JsonValueKind.Object && el.TryGetProperty("architectures", out var arr) &&
            arr.ValueKind == JsonValueKind.Array && arr.GetArrayLength() > 0)
        { first = arr[0].GetString() ?? ""; return first.Length > 0; }
        return false;
    }

    /// <summary>Fill missing (zero) config fields from the shapes of the mapped tensors.
    /// Never overrides values that are already set. Pass null shapes to skip.</summary>
    public static void EnrichConfigFromShapes(ModelConfig cfg, Dictionary<string, long[]>? shapes)
    {
        if (shapes == null) return;
        bool tok = shapes.TryGetValue("tok_emb", out var te) && te!.Length == 2;
        bool lmh = shapes.TryGetValue("lm_head", out var lh) && lh!.Length == 2;
        if (cfg.HiddenSize <= 0)
        {
            if (tok) cfg.HiddenSize = (int)te![1];
            else if (lmh) cfg.HiddenSize = (int)lh![1];
            else if (shapes.TryGetValue("final_norm", out var fn) && fn.Length == 1) cfg.HiddenSize = (int)fn[0];
        }
        if (cfg.VocabSize <= 0)
        {
            if (tok) cfg.VocabSize = (int)te![0];
            else if (lmh) cfg.VocabSize = (int)lh![0];
        }
        if (cfg.NumHeads <= 0 && cfg.HiddenSize > 0)
        {
            foreach (var kv in shapes)
                if (Regex.IsMatch(kv.Key, @"^l\d+\.wq$"))
                {
                    int hd = PickHeadDim(kv.Value[0], cfg.HiddenSize);
                    if (kv.Value[0] % hd == 0 && cfg.HiddenSize % hd == 0)
                    {
                        cfg.NumHeads = (int)(kv.Value[0] / hd);
                        if (cfg.HeadDim <= 0) cfg.HeadDim = hd;
                    }
                    break;
                }
        }
        if (cfg.NumKvHeads <= 0 && cfg.HeadDim > 0)
        {
            foreach (var kv in shapes)
                if (Regex.IsMatch(kv.Key, @"^l\d+\.wk$")) { cfg.NumKvHeads = (int)(kv.Value[0] / cfg.HeadDim); break; }
        }
        if (cfg.FfnDim <= 0)
            foreach (var kv in shapes)
                if (Regex.IsMatch(kv.Key, @"^l\d+\.w_gate$")) { cfg.FfnDim = kv.Value[0]; break; }
        if (cfg.NumExperts <= 0)
        {
            foreach (var kv in shapes)
                if (Regex.IsMatch(kv.Key, @"^l\d+\.router$"))
                {
                    cfg.NumExperts = (int)kv.Value[0];
                    if (cfg.NumActiveExperts <= 0) cfg.NumActiveExperts = 2;
                    cfg.Arch = "moe";
                    break;
                }
        }
        if (cfg.ExpertFfnDim <= 0)
            foreach (var kv in shapes)
                if (Regex.IsMatch(kv.Key, @"^l\d+\.e0\.w_gate$")) { cfg.ExpertFfnDim = kv.Value[0]; break; }
    }

    /// <summary>Refuse to build a model whose architecture config is incomplete — a .qw with
    /// dim=0/vocab=0 can never run, and silent garbage is worse than a clear error.</summary>
    internal static void ValidateConfig(ModelConfig cfg, string context)
    {
        var missing = new List<string>();
        if (cfg.NumLayers <= 0) missing.Add("layers (--layers / num_hidden_layers)");
        if (cfg.HiddenSize <= 0) missing.Add("hidden size (--dim / hidden_size)");
        if (cfg.NumHeads <= 0) missing.Add("attention heads (--heads / num_attention_heads)");
        if (cfg.VocabSize <= 0) missing.Add("vocab size (--vocab / vocab_size)");
        if (cfg.HasKda && (cfg.LinearHeads <= 0 || cfg.LinearHeadDim <= 0))
            missing.Add("KDA heads/head-dim (linear_attn_config: num_heads, head_dim)");
        if (cfg.HasMla && (cfg.QLoraRank <= 0 || cfg.KvLoraRank <= 0 || cfg.QkNopeHeadDim <= 0 || cfg.VHeadDim <= 0))
            missing.Add("MLA ranks (q_lora_rank, kv_lora_rank, qk_nope_head_dim, v_head_dim)");
        if (missing.Count == 0) return;
        throw new InvalidDataException(
            $"{context}: incomplete architecture config — missing {string.Join(", ", missing)}. " +
            "Pass a config.json (nested 'text_config' is supported) or the CLI overrides shown above.");
    }

    /// <summary>Fill zero/negative fields of a stored .qw config from a freshly parsed HF config.</summary>
    private static int HealConfig(ModelConfig cfg, ModelConfig hf)
    {
        int n = 0;
        if (cfg.NumLayers <= 0 && hf.NumLayers > 0) { cfg.NumLayers = hf.NumLayers; n++; }
        if (cfg.HiddenSize <= 0 && hf.HiddenSize > 0) { cfg.HiddenSize = hf.HiddenSize; n++; }
        if (cfg.NumHeads <= 0 && hf.NumHeads > 0) { cfg.NumHeads = hf.NumHeads; n++; }
        if (cfg.NumKvHeads <= 0 && hf.NumKvHeads > 0) { cfg.NumKvHeads = hf.NumKvHeads; n++; }
        if (cfg.HeadDim <= 0 && hf.HeadDim > 0) { cfg.HeadDim = hf.HeadDim; n++; }
        if (cfg.FfnDim <= 0 && hf.FfnDim > 0) { cfg.FfnDim = hf.FfnDim; n++; }
        if (cfg.VocabSize <= 0 && hf.VocabSize > 0) { cfg.VocabSize = hf.VocabSize; n++; }
        if (cfg.NumExperts <= 0 && hf.NumExperts > 0) { cfg.NumExperts = hf.NumExperts; n++; }
        if (cfg.NumActiveExperts <= 0 && hf.NumActiveExperts > 0) { cfg.NumActiveExperts = hf.NumActiveExperts; n++; }
        if (cfg.ExpertFfnDim <= 0 && hf.ExpertFfnDim > 0) { cfg.ExpertFfnDim = hf.ExpertFfnDim; n++; }
        if (cfg.Arch == "llama" && hf.Arch == "moe") { cfg.Arch = "moe"; n++; }

        // GLM-5.x fields
        if (cfg.AttnKinds.Length == 0 && hf.AttnKinds.Length > 0) { cfg.AttnKinds = hf.AttnKinds; n++; }
        if (cfg.HcMult <= 0 && hf.HcMult > 0)
        { cfg.HcMult = hf.HcMult; cfg.HcEps = hf.HcEps; cfg.HcSinkhornIters = hf.HcSinkhornIters; n++; }
        if (cfg.QLoraRank <= 0 && hf.QLoraRank > 0) { cfg.QLoraRank = hf.QLoraRank; n++; }
        if (cfg.KvLoraRank <= 0 && hf.KvLoraRank > 0) { cfg.KvLoraRank = hf.KvLoraRank; n++; }
        if (cfg.QkNopeHeadDim <= 0 && hf.QkNopeHeadDim > 0) { cfg.QkNopeHeadDim = hf.QkNopeHeadDim; n++; }
        if (cfg.QkRopeHeadDim <= 0 && hf.QkRopeHeadDim > 0) { cfg.QkRopeHeadDim = hf.QkRopeHeadDim; n++; }
        if (cfg.VHeadDim <= 0 && hf.VHeadDim > 0) { cfg.VHeadDim = hf.VHeadDim; n++; }
        if (cfg.LinearHeads <= 0 && hf.LinearHeads > 0) { cfg.LinearHeads = hf.LinearHeads; n++; }
        if (cfg.LinearHeadDim <= 0 && hf.LinearHeadDim > 0) { cfg.LinearHeadDim = hf.LinearHeadDim; n++; }
        if (cfg.LinearLowerBound == 0 && hf.LinearLowerBound != 0) { cfg.LinearLowerBound = hf.LinearLowerBound; n++; }
        if (!cfg.MoeSigmoid && hf.MoeSigmoid)
        {
            cfg.MoeSigmoid = true; cfg.RoutedScale = hf.RoutedScale;
            cfg.NumSharedExperts = hf.NumSharedExperts; n++;
        }
        if (cfg.SwigluLimit <= 0 && hf.SwigluLimit > 0) { cfg.SwigluLimit = hf.SwigluLimit; n++; }
        return n;
    }

    /// <summary>
    /// Derive a ModelConfig from canonical tensor names + shapes when no config.json is available.
    /// Works for standard Llama-family layouts: hidden size from tok_emb, FFN from w_gate,
    /// head layout from wq/wk rows (head_dim 64 preferred for small dims, 128 otherwise),
    /// MoE detected from l{i}.e{j}.* expert tensors.
    /// </summary>
    public static ModelConfig InferConfigFromTensors(Dictionary<string, long[]> shapes, string sourceName)
    {
        int maxLayer = -1, maxExpert = -1;
        bool moe = false;
        foreach (var n in shapes.Keys)
        {
            var m = Regex.Match(n, @"^l(\d+)\.");
            if (m.Success) maxLayer = Math.Max(maxLayer, int.Parse(m.Groups[1].Value));
            var me = Regex.Match(n, @"^l\d+\.e(\d+)\.w_");
            if (me.Success) { moe = true; maxExpert = Math.Max(maxExpert, int.Parse(me.Groups[1].Value)); }
        }
        if (maxLayer < 0)
            throw new InvalidDataException(
                "cannot infer architecture: no 'model.layers.*' tensors found. This file may be a partial shard " +
                "or a non-Llama-family model — pass --config config.json explicitly.");

        int dim;
        if (shapes.TryGetValue("tok_emb", out var emb) && emb.Length == 2) dim = (int)emb[1];
        else if (shapes.TryGetValue("final_norm", out var fn) && fn.Length == 1) dim = (int)fn[0];
        else if (shapes.TryGetValue("lm_head", out var lh) && lh.Length == 2) dim = (int)lh[1];
        else
        {
            var wq = shapes.FirstOrDefault(kv => Regex.IsMatch(kv.Key, @"^l\d+\.wq$"));
            if (wq.Value == null)
                throw new InvalidDataException("cannot infer hidden size (no tok_emb / final_norm / lm_head / wq tensor)");
            dim = (int)wq.Value[1];
        }
        int vocab = shapes.TryGetValue("tok_emb", out var e2) && e2.Length == 2 ? (int)e2[0]
                  : shapes.TryGetValue("lm_head", out var l2) && l2.Length == 2 ? (int)l2[0] : 0;

        long ffn = 0;
        foreach (var pat in new[] { @"^l\d+\.w_gate$", @"^l\d+\.w_up$" })
            foreach (var kv in shapes)
                if (Regex.IsMatch(kv.Key, pat)) { ffn = kv.Value[0]; goto done; }
        if (ffn == 0)
            foreach (var kv in shapes)
                if (Regex.IsMatch(kv.Key, @"^l\d+\.w_down$")) { ffn = kv.Value[1]; break; }
    done:;

        long qRows = 0, kRows = 0;
        foreach (var kv in shapes)
        {
            if (qRows == 0 && Regex.IsMatch(kv.Key, @"^l\d+\.wq$")) qRows = kv.Value[0];
            if (kRows == 0 && Regex.IsMatch(kv.Key, @"^l\d+\.wk$")) kRows = kv.Value[0];
        }
        int headDim = qRows > 0 ? PickHeadDim(qRows, dim) : (dim % 64 == 0 ? 64 : dim);
        int heads = headDim > 0 && qRows > 0 ? (int)(qRows / headDim)
                  : dim % headDim == 0 ? dim / headDim : 1;
        int kvHeads = kRows > 0 ? (int)(kRows / headDim) : heads;
        if (kvHeads <= 0) kvHeads = heads;

        var cfg = new ModelConfig
        {
            Arch = moe ? "moe" : "llama",
            HiddenSize = dim,
            NumLayers = maxLayer + 1,
            NumHeads = heads,
            NumKvHeads = kvHeads,
            HeadDim = headDim,
            FfnDim = ffn,
            VocabSize = vocab,
            NumExperts = moe ? maxExpert + 1 : 0,
            NumActiveExperts = moe ? 2 : 0,
            MaxCtx = 4096,
            SourceDescription = $"{sourceName} (inferred)"
        };
        if (moe)
        {
            var eg = shapes.FirstOrDefault(kv => Regex.IsMatch(kv.Key, @"^l\d+\.e0\.w_gate$"));
            if (eg.Value != null) cfg.ExpertFfnDim = eg.Value[0];
        }
        return cfg;
    }

    private static int PickHeadDim(long qRows, int dim)
    {
        // small models usually use head_dim 64; 128 dominates modern >=2B checkpoints
        var candidates = dim <= 1024 ? new[] { 64, 128, 32, 80, 96 } : new[] { 128, 80, 96, 64, 32 };
        foreach (int h in candidates)
            if (qRows % h == 0 && qRows / h >= 2 && dim % h == 0) return h;
        return (int)qRows;   // single-head fallback
    }

    // ------------------------------------------------------------------ writers

    private static ConvertReport WritePlan(List<TensorPlan> plan, GgufFile gguf,
        Dictionary<string, GgufFile.GgufTensorInfo> source, object unused, string outPath, ModelConfig cfg,
        string? extraConfigJson = null)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        long originalBytes = gguf.Stream.Length;
        JsonElement configElement;
        if (extraConfigJson != null)
        {
            // embedded under a dedicated key; ModelConfig.FromJson ignores unknown properties
            var node = System.Text.Json.Nodes.JsonNode.Parse(cfg.ToJson())!;
            node["gguf_kvs"] = System.Text.Json.Nodes.JsonNode.Parse(extraConfigJson)!.DeepClone();
            configElement = JsonDocument.Parse(node.ToJsonString()).RootElement;
        }
        else
        {
            configElement = JsonDocument.Parse(cfg.ToJson()).RootElement;
        }
        using var builder = new ModelStoreBuilder(outPath, configElement);
        foreach (var p in plan) builder.Plan(p.Name, p.Dims, p.Format);
        builder.Commit((planned, stream) => StreamGgufTensor(planned, gguf, source, stream));
        sw.Stop();
        return new ConvertReport(plan.Count, plan.Sum(p => p.SizeBytes), originalBytes, sw.Elapsed.TotalSeconds, cfg);
    }

    private static unsafe void StreamGgufTensor(TensorPlan p, GgufFile gguf,
        Dictionary<string, GgufFile.GgufTensorInfo> source, Stream outStream)
    {
        var t = source[p.Name];
        long dataPos = gguf.DataStart + t.Offset;
        long n = p.NumElements;
        var (tbs, _) = Dtypes.BlockSpec(p.Format);

        bool passthrough = p.Format switch
        {
            // byte-exact only when the ggml block layout is IDENTICAL to ours. Q4_0 is NOT:
            // ggml Q4_0 is 18 B with a signed delta, ours is 20 B with a positive delta —
            // it always goes through dequant -> requant (see GgmlDequant).
            QuantFormat.Q8_0 => t.GgmlType == 8,
            QuantFormat.Q2_K => t.GgmlType == 10,
            QuantFormat.Q3_K => t.GgmlType == 11,
            QuantFormat.Q4_K => t.GgmlType == 12,
            QuantFormat.Q5_K => t.GgmlType == 13,
            QuantFormat.Q6_K => t.GgmlType == 14,
            QuantFormat.F32 => t.GgmlType == 0,
            QuantFormat.F16 => t.GgmlType == 1,
            QuantFormat.BF16 => t.GgmlType == 30,
            _ => false
        };

        bool isExpert = p.Name.Contains(".e") && p.Name.Contains(".w_");
        int expertIdx = 0;
        if (isExpert)
        {
            int ePos = p.Name.IndexOf(".e", StringComparison.Ordinal);
            int dot = p.Name.IndexOf('.', ePos + 1);
            expertIdx = int.Parse(p.Name.Substring(ePos + 2, dot - ePos - 2));
        }

        const long CHUNK_ELEMS = 1 << 20;
        var srcBuf = new byte[CHUNK_ELEMS * 2];
        var f32 = new float[CHUNK_ELEMS];
        var dstBuf = new byte[Dtypes.TensorBytes(CHUNK_ELEMS, p.Format)];

        var (sbs, sbb) = GgufFile.GgmlBlockSpec(t.GgmlType);
        long expertSpanBlocks = isExpert ? (t.Dims[0] / sbs) * t.Dims[1] : 0;

        long remaining = n, elemBase = 0;
        while (remaining > 0)
        {
            long elems = Math.Min(CHUNK_ELEMS - CHUNK_ELEMS % Dtypes.BlockSpec(p.Format).blockSize, remaining);
            long blocks = elems / sbs;
            long srcBlockStart = isExpert
                ? (long)expertIdx * expertSpanBlocks + elemBase / sbs
                : elemBase / sbs;
            long srcBytes = blocks * sbb;
            gguf.Stream.Position = gguf.DataStart + t.Offset + srcBlockStart * sbb;

            int total = (int)srcBytes, got = 0;
            while (got < total)
            {
                int r = gguf.Stream.Read(srcBuf, got, total - got);
                if (r <= 0) throw new EndOfStreamException($"GGUF tensor {t.Name}");
                got += r;
            }

            if (passthrough)
            {
                outStream.Write(srcBuf, 0, total);
            }
            else
            {
                // dequantize with the exact ggml semantics (GgmlDequant handles the legacy
                // 18/20/22/24-byte formats and delegates layout-identical types)
                GgmlDequant.Dequant(t.GgmlType, srcBuf.AsSpan(0, total), f32.AsSpan(0, (int)elems));
                long outBytes = Dtypes.TensorBytes(elems, p.Format);
                BlockQuantizer.Quantize(p.Format, f32.AsSpan(0, (int)elems), dstBuf.AsSpan(0, (int)outBytes));
                outStream.Write(dstBuf, 0, (int)outBytes);
            }
            remaining -= elems;
            elemBase += elems;
        }
    }

    /// <summary>A source tensor to stream from — either whole (Segments == null) or as a set
    /// of element ranges produced by splitting a fused projection (contiguous or interleaved).
    /// ScaleInv is the fp8 block-scale companion tensor for F8_E4M3 sources.</summary>
    private sealed record SplitEntry(string Name, long[] Dims, SafeTensorsFile File,
        SafeTensorsFile.TensorInfo Info, (long Start, long Elems)[]? Segments, SafeTensorsFile.TensorInfo? ScaleInv = null);

    private static Dictionary<string, long[]> ShapesOf(
        Dictionary<string, (SafeTensorsFile File, SafeTensorsFile.TensorInfo Info, SafeTensorsFile.TensorInfo? Scale)> owner) =>
        owner.ToDictionary(kv => kv.Key, kv => kv.Value.Info.Shape);

    /// <summary>Maps every tensor of every file, collecting canonical owners, fp8 scale companions,
    /// skip notes and unrecognized names. Nothing is silently dropped: unrecognized names are
    /// always reported.</summary>
    private static void CollectSources(IEnumerable<SafeTensorsFile> readers,
        Dictionary<string, (SafeTensorsFile File, SafeTensorsFile.TensorInfo Info, SafeTensorsFile.TensorInfo? Scale)> owner,
        List<string> unmapped, Dictionary<string, int> notes)
    {
        var all = readers.SelectMany(r => r.Tensors.Keys).ToList();
        var kinds = NameMapper.DetectLayerKinds(all);

        // fp8 block scales: "X.weight_scale_inv" belongs to canonical(X.weight)
        var scaleByRawBase = new Dictionary<string, (SafeTensorsFile File, SafeTensorsFile.TensorInfo Info)>();
        foreach (var r in readers)
            foreach (var (name, info) in r.Tensors)
                if (name.EndsWith(".weight_scale_inv", StringComparison.Ordinal))
                    scaleByRawBase[name[..^".weight_scale_inv".Length]] = (r, info);
        var scaleByCanonical = new Dictionary<string, SafeTensorsFile.TensorInfo>();
        foreach (var (rawBase, (file, info)) in scaleByRawBase)
        {
            // scale names are "X.weight_scale_inv" where the sibling weight is "X.weight"
            // (X = e.g. "...mlp.down_proj" / "...self_attn.kv_a_proj_with_mqa")
            string weightName = rawBase.EndsWith(".weight", StringComparison.Ordinal) ? rawBase : rawBase + ".weight";
            var mapped = NameMapper.Map(NameMapper.Source.HuggingFace, weightName, info.Shape, int.MaxValue, kinds, out _);
            if (mapped != null && !scaleByCanonical.ContainsKey(mapped)) scaleByCanonical[mapped] = info;
        }

        foreach (var r in readers)
            foreach (var (name, info) in r.Tensors)
            {
                if (name.EndsWith(".weight_scale_inv", StringComparison.Ordinal)) continue;   // consumed above
                // layer count is unknown until config is resolved: map everything, filter after
                var mapped = NameMapper.Map(NameMapper.Source.HuggingFace, name, info.Shape, int.MaxValue, kinds, out string note);
                if (note != "") notes[note] = (notes.TryGetValue(note, out int c) ? c : 0) + 1;
                if (mapped == null)
                {
                    if (note == "") unmapped.Add(name);   // intentional skips carry a note instead
                    continue;
                }
                if (!owner.ContainsKey(mapped))
                    owner[mapped] = (r, info, scaleByCanonical.GetValueOrDefault(mapped));
            }
    }

    private static void ReportDiagnostics(List<string> unmapped, Dictionary<string, int> notes)
    {
        foreach (var kv in notes.Take(5))
            Console.WriteLine($"  note     : {kv.Value}x {kv.Key}");
        if (notes.Count > 5) Console.WriteLine($"             (+ {notes.Count - 5} more skip reasons)");
        if (unmapped.Count > 0)
        {
            Console.WriteLine($"  WARNING  : {unmapped.Count} tensor name(s) were not recognized and are NOT in the .qw:");
            foreach (var n in unmapped.Take(6)) Console.WriteLine($"             - {n}");
            if (unmapped.Count > 6) Console.WriteLine($"             ... and {unmapped.Count - 6} more");
            Console.WriteLine("             supported: Llama-family layouts (GQA attention, Mixtral-style MoE).");
        }
    }

    /// <summary>
    /// Expands fused-projection pseudo names into concrete canonical tensors:
    ///   GU:{i} / EGU:{i}:{j}  -> w_gate | w_up            (contiguous row halves)
    ///   QKV:{i}               -> wq | wk | wv             (contiguous q | k | v rows)
    ///   QKVGLM:{i}            -> wq | wk | wv             (per-head interleaved [q|k|v])
    /// Everything else passes through unchanged. May fill FfnDim / ExpertFfnDim / NumExperts
    /// when they are still unset.
    /// </summary>
    private static List<SplitEntry> ExpandSplits(
        Dictionary<string, (SafeTensorsFile File, SafeTensorsFile.TensorInfo Info, SafeTensorsFile.TensorInfo? Scale)> owner, ModelConfig cfg)
    {
        var result = new List<SplitEntry>(owner.Count + 8);
        int maxExpert = -1;
        foreach (var (canonical, (file, info, scale)) in owner)
        {
            if (canonical.StartsWith("EGU:", StringComparison.Ordinal) || canonical.StartsWith("GU:", StringComparison.Ordinal))
            {
                if (info.Shape.Length != 2 || info.Shape[0] % 2 != 0)
                    throw new InvalidDataException(
                        $"fused gate/up tensor '{canonical}' must be 2-D with even rows, got [{string.Join(',', info.Shape)}]");
                long rows = info.Shape[0], cols = info.Shape[1];
                long ffn = rows / 2;
                string baseName;
                if (canonical.StartsWith("EGU:", StringComparison.Ordinal))
                {
                    var parts = canonical.Split(':');
                    baseName = $"l{parts[1]}.e{parts[2]}.";
                    if (cfg.ExpertFfnDim <= 0) cfg.ExpertFfnDim = ffn;
                    if (int.TryParse(parts[2], out int j) && j > maxExpert) maxExpert = j;
                }
                else
                {
                    baseName = $"l{canonical[3..]}.";
                    if (cfg.FfnDim <= 0) cfg.FfnDim = ffn;
                }
                result.Add(new SplitEntry(baseName + "w_gate", new[] { ffn, cols }, file, info, new[] { (0L, ffn * cols) }, scale));
                result.Add(new SplitEntry(baseName + "w_up", new[] { ffn, cols }, file, info, new[] { (ffn * cols, ffn * cols) }, scale));
            }
            else if (!canonical.StartsWith("QKV:", StringComparison.Ordinal) &&
                     !canonical.StartsWith("QKVGLM:", StringComparison.Ordinal))
            {
                // nn.Conv1d depthwise weights arrive as [channels, 1, kernel] — store as [channels, kernel]
                long[] dims = info.Shape.Length == 3 && info.Shape[1] == 1
                    ? new[] { info.Shape[0], info.Shape[2] }
                    : info.Shape.ToArray();
                result.Add(new SplitEntry(canonical, dims, file, info, null, scale));
            }
        }
        if (maxExpert >= 0 && cfg.NumExperts <= 0)
        {
            cfg.NumExperts = maxExpert + 1;
            if (cfg.NumActiveExperts <= 0) cfg.NumActiveExperts = 2;
            cfg.Arch = "moe";
        }

        // fused attention projections need the head layout to split
        foreach (var (canonical, (file, info, scale)) in owner)
        {
            bool contiguous = canonical.StartsWith("QKV:", StringComparison.Ordinal);
            bool interleaved = canonical.StartsWith("QKVGLM:", StringComparison.Ordinal);
            if (!contiguous && !interleaved) continue;
            if (info.Shape.Length != 2)
                throw new InvalidDataException($"fused attention tensor '{canonical}' must be 2-D, got [{string.Join(',', info.Shape)}]");
            if (cfg.NumHeads <= 0)
                throw new InvalidDataException(
                    $"fused attention tensor '{canonical}' cannot be split without the head layout — " +
                    "pass --heads/--kv-heads/--head-dim or a config.json with num_attention_heads.");
            long rows = info.Shape[0], cols = info.Shape[1];
            int hd = cfg.EffectiveHeadDim;
            int heads = cfg.NumHeads, kv = cfg.EffectiveKvHeads;
            string layer = contiguous ? canonical[4..] : canonical[7..];
            string baseName = $"l{layer}.";
            if (contiguous)
            {
                long qRows = (long)heads * hd, kvRows = (long)kv * hd;
                if (rows != qRows + 2 * kvRows)
                    throw new InvalidDataException(
                        $"fused QKV tensor '{canonical}' has {rows} rows but heads*head_dim + 2*kv*head_dim = {qRows + 2 * kvRows} " +
                        $"(heads={heads}, kv-heads={kv}, head_dim={hd}) — check --heads/--kv-heads/--head-dim.");
                result.Add(new SplitEntry(baseName + "wq", new[] { qRows, cols }, file, info, new[] { (0L, qRows * cols) }, scale));
                result.Add(new SplitEntry(baseName + "wk", new[] { kvRows, cols }, file, info, new[] { (qRows * cols, kvRows * cols) }, scale));
                result.Add(new SplitEntry(baseName + "wv", new[] { kvRows, cols }, file, info, new[] { ((qRows + kvRows) * cols, kvRows * cols) }, scale));
            }
            else
            {
                // GLM/BLOOM layout: rows grouped per head as [q(hd) | k(hd) | v(hd)]
                if (kv != heads)
                    throw new InvalidDataException($"interleaved QKV tensor '{canonical}' requires kv-heads == heads (MHA), got {kv} vs {heads}");
                if (rows != (long)heads * 3 * hd)
                    throw new InvalidDataException(
                        $"interleaved QKV tensor '{canonical}' has {rows} rows but 3*heads*head_dim = {(long)heads * 3 * hd} " +
                        $"(heads={heads}, head_dim={hd}) — check --heads/--head-dim.");
                var qSegs = new List<(long, long)>(heads);
                var kSegs = new List<(long, long)>(heads);
                var vSegs = new List<(long, long)>(heads);
                for (int h = 0; h < heads; h++)
                {
                    long rowBase = (long)h * 3 * hd;
                    qSegs.Add((rowBase * cols, hd * cols));
                    kSegs.Add(((rowBase + hd) * cols, hd * cols));
                    vSegs.Add(((rowBase + 2 * hd) * cols, hd * cols));
                }
                long per = (long)heads * hd;
                result.Add(new SplitEntry(baseName + "wq", new[] { per, cols }, file, info, qSegs.ToArray(), scale));
                result.Add(new SplitEntry(baseName + "wk", new[] { per, cols }, file, info, kSegs.ToArray(), scale));
                result.Add(new SplitEntry(baseName + "wv", new[] { per, cols }, file, info, vSegs.ToArray(), scale));
            }
        }
        return result;
    }

    private static ConvertReport WriteSafeTensors(List<TensorPlan> plan, Dictionary<string, SplitEntry> src,
        long originalBytes, string outPath, ModelConfig cfg, long headerReserveBytes = 0)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var builder = new ModelStoreBuilder(outPath, JsonDocument.Parse(cfg.ToJson()).RootElement, headerReserveBytes);
        foreach (var p in plan) builder.Plan(p.Name, p.Dims, p.Format);
        builder.Commit((planned, stream) => StreamSourceChunks(planned, src[planned.Name], stream));
        sw.Stop();
        return new ConvertReport(plan.Count, plan.Sum(p => p.SizeBytes), originalBytes, sw.Elapsed.TotalSeconds, cfg);
    }

    private static void StreamSourceChunks(TensorPlan planned, SplitEntry src, Stream stream)
    {
        long n = planned.NumElements;
        const long CHUNK = 1 << 20;
        var f32 = new float[CHUNK];
        long streamed = 0;
        if (src.Segments == null)
        {
            long done = 0;
            while (done < n)
            {
                long elems = Math.Min(CHUNK, n - done);
                src.File.ReadChunkF32(src.Info, done, f32.AsSpan(0, (int)elems), src.ScaleInv);
                WriteQuantChunk(planned.Format, f32, elems, stream);
                done += elems; streamed += elems;
            }
        }
        else
        {
            foreach (var (start, elems) in src.Segments)
            {
                long done = 0;
                while (done < elems)
                {
                    long take = Math.Min(CHUNK, elems - done);
                    src.File.ReadChunkF32(src.Info, start + done, f32.AsSpan(0, (int)take), src.ScaleInv);
                    WriteQuantChunk(planned.Format, f32, take, stream);
                    done += take; streamed += take;
                }
            }
        }
        if (streamed != n)
            throw new IOException($"tensor '{planned.Name}': streamed {streamed} elements, planned {n}");
    }

    private static void WriteQuantChunk(QuantFormat fmt, float[] f32, long elems, Stream stream)
    {
        long bytes = Dtypes.TensorBytes(elems, fmt);
        var dst = new byte[bytes];
        BlockQuantizer.Quantize(fmt, f32.AsSpan(0, (int)elems), dst.AsSpan(0, (int)bytes));
        stream.Write(dst, 0, (int)bytes);
    }
}
