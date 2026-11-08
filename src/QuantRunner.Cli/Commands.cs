using System.Diagnostics;
using QuantRunner.Core.Memory;
using QuantRunner.Core.Models;
using QuantRunner.Core.Quantization;
using QuantRunner.Core.Runtime;
using QuantRunner.Core.Tokenizers;

using QuantRunner.Core;
using QuantRunner.Core.Importers;
using QuantRunner.Core.Kernels;

namespace QuantRunner.Cli;

internal static class Commands
{
    // ------------------------------------------------------------------ info (.qw / .gguf / .safetensors / HF dir)

    public static int CmdInfo(string[] a)
    {
        if (a.Length == 0)
        {
            Console.Error.WriteLine("usage: info <model.qw | model.gguf | model.safetensors | hf-dir/>");
            return 2;
        }
        string path = a[0];
        if (Directory.Exists(path)) return InfoHfDir(path);
        if (!File.Exists(path)) { Console.Error.WriteLine($"error: file not found: {path}"); return 2; }
        string ext = Path.GetExtension(path).ToLowerInvariant();
        if (ext == ".qw") return InfoQw(path);
        if (ext == ".gguf") return InfoGguf(path);
        if (ext == ".safetensors") return InfoSafeTensors(path);
        Console.Error.WriteLine($"error: unknown model kind '{Path.GetExtension(path)}' — want .qw / .gguf / .safetensors, or an HF directory");
        return 2;
    }

    private static int InfoQw(string path)
    {
        using var store = ModelStore.Open(path);
        var cfg = ModelConfig.FromJson(store.ConfigJson);
        long bytes = store.Tensors.Values.Sum(t => t.SizeBytes);

        Console.WriteLine($"file        : {store.Path}");
        string kinds = "";
        if (cfg.AttnKinds.Length > 0)
        {
            int kda = cfg.AttnKinds.Count(k => k == "kda"), mla = cfg.AttnKinds.Count(k => k == "mla"),
                gqa = cfg.AttnKinds.Length - kda - mla;
            kinds = $" | attn: " + string.Join(" + ", new[] {
                kda > 0 ? $"{kda} KDA" : null, mla > 0 ? $"{mla} MLA/DSA" : null,
                gqa > 0 ? $"{gqa} GQA" : null }.Where(s => s != null));
        }
        string hc = cfg.HasMhc ? $" | mHC x{cfg.HcMult}" : "";
        string moeStyle = cfg.IsMoe && cfg.MoeSigmoid
            ? $" (MoE sigmoid/noaux_tc: {cfg.NumExperts} experts, {cfg.NumActiveExperts} active, {cfg.NumSharedExperts} shared)" : "";
        Console.WriteLine($"arch        : {cfg.Arch}{moeStyle}{kinds}{hc}");
        Console.WriteLine($"layers      : {cfg.NumLayers}   dim: {cfg.HiddenSize}   heads: {cfg.NumHeads}   kv-heads: {cfg.EffectiveKvHeads}");
        if (cfg.HasKda)
            Console.WriteLine($"kda         : {cfg.LinearHeads} heads x {cfg.LinearHeadDim} dim, conv {cfg.LinearConvKernel}, gate lb {cfg.LinearLowerBound}");
        if (cfg.HasMla)
            Console.WriteLine($"mla         : q_lora {cfg.QLoraRank}, kv_lora {cfg.KvLoraRank}, qk {cfg.QkNopeHeadDim}(+rope {cfg.QkRopeHeadDim}), v {cfg.VHeadDim}");
        Console.WriteLine($"ffn         : {cfg.FfnDim}{(cfg.IsMoe ? $" (expert ffn {cfg.EffectiveExpertFfn})" : "")}");
        Console.WriteLine($"vocab       : {cfg.VocabSize}   rope theta: {cfg.RopeTheta}   ctx: {cfg.MaxCtx}");

        PrintParams(store.Tensors.Values.Select(t => (t.Name, t.Dims)),
            cfg.IsMoe ? cfg.NumExperts : 0, cfg.NumActiveExperts, new FileInfo(path).Length);
        PrintFormatMix(store.Tensors.Values.Select(t => (t.Format.ToString(), t.NumElements)));

        Console.WriteLine($"file size   : {Program.Fmt(bytes)}  -> all-streaming RAM need ~ layer bytes + activations");
        var (present, missing) = Completeness(store, cfg);
        if (missing.Count > 0)
        {
            Console.WriteLine($"completeness: {present}/{cfg.NumLayers} layers, {missing.Count} tensors missing" +
                              $"  — quantize --append <shard> -o {store.Path}");
            Console.WriteLine($"              (parameter counts above cover the tensors present so far)");
        }
        Console.WriteLine();
        Console.WriteLine($"{"tensor",-24} {"dims",-16} {"format",-8} {"size",12}");
        foreach (var t in store.Tensors.Values.Take(14))
            Console.WriteLine($"{t.Name,-24} {string.Join('x', t.Dims),-16} {t.Format,-8} {Program.Fmt(t.SizeBytes),12}");
        if (store.Tensors.Count > 14) Console.WriteLine($"... and {store.Tensors.Count - 14} more");
        return 0;
    }

    private static int InfoGguf(string path)
    {
        using var g = GgufFile.Open(path);
        string arch = g.Kv("general.architecture") as string ?? "gguf";
        long KvL(string key, long def = 0) => g.KvLong($"{arch}.{key}", g.KvLong(key, def));
        long layers = KvL("block_count");
        long dim = KvL("embedding_length");
        long heads = KvL("attention.head_count");
        long kvHeads = KvL("attention.head_count_kv", heads);
        long vocab = KvL("vocab_size");
        long experts = KvL("expert_count");
        long nActive = KvL("expert_used_count");
        float rope = g.KvFloat($"{arch}.rope.freq_base", g.KvFloat("rope.freq_base", 0));

        Console.WriteLine($"file        : {path}  (GGUF v{g.Version})");
        Console.WriteLine($"arch        : {arch}" + (experts > 0 ? $" (MoE: {experts} experts, {nActive} active)" : ""));
        Console.WriteLine($"layers      : {layers}   dim: {dim}   heads: {heads}   kv-heads: {kvHeads}");
        Console.WriteLine($"vocab       : {vocab}" + (rope > 0 ? $"   rope theta: {rope:N0}" : ""));

        var tensors = g.Tensors.Select(t => (t.Name, t.Dims)).ToList();
        PrintParams(tensors, (int)experts, (int)nActive, new FileInfo(path).Length);
        PrintFormatMix(g.Tensors.Select(t => (GgufFile.GgmlTypeName(t.GgmlType), t.NumElements)));

        Console.WriteLine($"file size   : {Program.Fmt(new FileInfo(path).Length)}");
        Console.WriteLine();
        Console.WriteLine($"{"tensor",-24} {"dims",-16} {"format",-8} {"size",12}");
        foreach (var t in g.Tensors.Take(14))
            Console.WriteLine($"{t.Name,-24} {string.Join('x', t.Dims.Reverse()),-16} {GgufFile.GgmlTypeName(t.GgmlType),-8} {Program.Fmt(t.SizeBytes),12}");
        if (g.Tensors.Count > 14) Console.WriteLine($"... and {g.Tensors.Count - 14} more");
        return 0;
    }

    private static int InfoSafeTensors(string path)
    {
        using var st = SafeTensorsFile.Open(path);
        Console.WriteLine($"file        : {path}  (HuggingFace safetensors)");
        var cfg = ConfigBeside(path);
        PrintCfg(cfg);
        var tensors = st.Tensors.Values.Select(t => (t.Name, t.Shape)).ToList();
        PrintParams(tensors, cfg is { IsMoe: true } ? cfg.NumExperts : 0, cfg?.NumActiveExperts ?? 0, new FileInfo(path).Length,
            "single shard — point info at the containing directory for whole-model totals");
        Console.WriteLine();
        Console.WriteLine($"{"tensor",-24} {"dims",-16} {"dtype",-8} {"size",12}");
        foreach (var t in st.Tensors.Values.Take(14))
            Console.WriteLine($"{t.Name,-24} {string.Join('x', t.Shape),-16} {t.Dtype,-8} {Program.Fmt(t.DataEnd - t.DataStart),12}");
        if (st.Tensors.Count > 14) Console.WriteLine($"... and {st.Tensors.Count - 14} more");
        return 0;
    }

    private static int InfoHfDir(string dir)
    {
        var shards = Directory.GetFiles(dir, "*.safetensors").OrderBy(f => f, StringComparer.OrdinalIgnoreCase).ToArray();
        if (shards.Length == 0)
        {
            Console.Error.WriteLine($"error: no .safetensors files in {dir}");
            return 2;
        }
        Console.WriteLine($"file        : {dir}  (HuggingFace directory, {shards.Length} shard file(s))");
        ModelConfig? cfg = null;
        string cfgPath = Path.Combine(dir, "config.json");
        if (File.Exists(cfgPath))
        {
            try
            {
                cfg = ModelConverter.ParseHfConfig(File.ReadAllText(cfgPath), new ConvertOptions(), "config.json", out var unsup);
                if (unsup.Count > 0)
                    Console.WriteLine($"            (config uses features the engine can't run: {string.Join(", ", unsup)})");
            }
            catch (Exception ex) { Console.WriteLine($"            (config.json unreadable: {ex.Message})"); }
        }
        PrintCfg(cfg);

        var all = new List<(string Name, long[] Dims)>();
        foreach (var f in shards)
            using (var st = SafeTensorsFile.Open(f))
                foreach (var t in st.Tensors.Values) all.Add((t.Name, t.Shape));
        long fileBytes = shards.Sum(f => new FileInfo(f).Length);
        PrintParams(all, cfg is { IsMoe: true } ? cfg.NumExperts : 0, cfg?.NumActiveExperts ?? 0, fileBytes);

        string idxPath = Path.Combine(dir, "model.safetensors.index.json");
        if (File.Exists(idxPath))
        {
            try
            {
                using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(idxPath));
                long total = doc.RootElement.TryGetProperty("metadata", out var md) &&
                             md.TryGetProperty("total_size", out var ts) ? ts.GetInt64() : 0;
                int wm = doc.RootElement.TryGetProperty("weight_map", out var map)
                    ? map.EnumerateObject().Count() : 0;
                Console.WriteLine($"    index       : {wm} weights in weight_map, total_size {total:N0}");
            }
            catch { /* index is informational only */ }
        }
        return 0;
    }

    private static ModelConfig? ConfigBeside(string file)
    {
        string? dir = Path.GetDirectoryName(Path.GetFullPath(file));
        string cfgPath = Path.Combine(dir!, "config.json");
        if (!File.Exists(cfgPath)) return null;
        try { return ModelConverter.ParseHfConfig(File.ReadAllText(cfgPath), new ConvertOptions(), "config.json", out _); }
        catch { return null; }
    }

    private static void PrintCfg(ModelConfig? cfg)
    {
        if (cfg == null) { Console.WriteLine("arch        : (no config.json found — parameter counts from tensor shapes)"); return; }
        string moe = cfg.IsMoe ? $", experts={cfg.NumExperts}x{cfg.NumActiveExperts} active" : "";
        Console.WriteLine($"arch        : {cfg.Arch}{moe}");
        if (cfg.NumLayers > 0)
            Console.WriteLine($"layers      : {cfg.NumLayers}   dim: {cfg.HiddenSize}   heads: {cfg.NumHeads}   kv-heads: {cfg.EffectiveKvHeads}");
        if (cfg.VocabSize > 0)
            Console.WriteLine($"vocab       : {cfg.VocabSize}   rope theta: {cfg.RopeTheta}   ctx: {cfg.MaxCtx}");
    }

    /// <summary>The star of `info`: total parameters + per-component breakdown, honest about
    /// what is actually stored (partial models count present tensors only).</summary>
    private static void PrintParams(IEnumerable<(string Name, long[] Dims)> tensors, int nExperts, int nActive,
        long fileBytes, string? note = null)
    {
        var list = tensors.ToList();
        var s = ParamReport.Count(list);
        long active = ParamReport.ActivePerToken(s, nExperts, nActive);
        int layers = ParamReport.LayersSeen(list.Select(t => t.Name));
        string bw = fileBytes > 0 && s.Total > 0 ? $"; {fileBytes * 8.0 / s.Total:F2} bits/weight effective" : "";
        Console.WriteLine($"parameters  : {s.Total:N0}  ({ParamReport.Human(s.Total)}, {s.TensorCount} tensors{bw})");
        foreach (var (label, n) in s.NonZeroBuckets())
        {
            bool perLayerOk = layers > 0 && (label.StartsWith("attention") || label.StartsWith("linear") ||
                                             label.StartsWith("ffn") || label.StartsWith("experts"));
            string per = perLayerOk ? $"   ~{ParamReport.Human(n / layers)}/layer x {layers}" : "";
            Console.WriteLine($"    {label,-18} {n,18:N0}  ({ParamReport.Human(n)}){per}");
        }
        if (s.Embedding + s.Head > 0)
            Console.WriteLine($"    {"non-embedding",-18} {s.NonEmbedding,18:N0}  ({ParamReport.Human(s.NonEmbedding)})");
        if (s.RoutedExperts > 0 && nExperts > 0 && nActive > 0)
            Console.WriteLine($"    {"active/token",-18} {("~" + ParamReport.Human(active)),18}  ({nActive} of {nExperts} routed experts + shared + dense)");
        if (note != null) Console.WriteLine($"    note: {note}");
    }

    private static void PrintFormatMix(IEnumerable<(string Fmt, long Elems)> parts)
    {
        var groups = parts.GroupBy(p => p.Fmt)
                          .Select(k => (Key: k.Key, N: k.Sum(x => x.Elems)))
                          .OrderByDescending(x => x.N).ToList();
        long total = groups.Sum(x => x.N);
        if (total <= 0 || groups.Count <= 1) return;
        string s = string.Join(", ", groups.Select(x => $"{x.Key} {ParamReport.Human(x.N)} ({100.0 * x.N / total:F1}%)"));
        Console.WriteLine($"    {"by format",-18} {s}");
    }


    // ------------------------------------------------------------------ quantize (.safetensors shard(s) -> .qw)

    public static int CmdQuantize(string[] a)
    {
        var files = new List<string>();
        string? output = null, configPath = null;
        bool append = false;
        long reserveMb = 8;                            // default: allow later --append passes
        var opt = new ConvertOptions();
        for (int i = 0; i < a.Length; i++)
        {
            switch (a[i])
            {
                case "-o" or "--out": output = a[++i]; break;
                case "--config": configPath = a[++i]; break;
                case "--append": append = true; break;
                case "--reserve-mb": reserveMb = long.Parse(a[++i]); break;
                case "--layers": opt.OverrideLayers = int.Parse(a[++i]); break;
                case "--heads": opt.OverrideHeads = int.Parse(a[++i]); break;
                case "--kv-heads": opt.OverrideKvHeads = int.Parse(a[++i]); break;
                case "--head-dim": opt.OverrideHeadDim = int.Parse(a[++i]); break;
                case "--dim": opt.OverrideDim = int.Parse(a[++i]); break;
                case "--ffn": opt.OverrideFfn = int.Parse(a[++i]); break;
                case "--vocab": opt.OverrideVocab = int.Parse(a[++i]); break;
                case "--experts": opt.OverrideExperts = int.Parse(a[++i]); break;
                case "--active": opt.OverrideActive = int.Parse(a[++i]); break;
                case "--qattn": opt.Attention = ConvertOptions.ParseQuant(a[++i]); break;
                case "--qffn": opt.Ffn = ConvertOptions.ParseQuant(a[++i]); break;
                case "--fmt": opt.Attention = opt.Ffn = ConvertOptions.ParseQuant(a[++i]); break;
                case "--qemb": opt.Emb = ConvertOptions.ParseQuant(a[++i]); break;
                case "--qhead": opt.Head = ConvertOptions.ParseQuant(a[++i]); break;
                case "--qnorm": opt.Norm = ConvertOptions.ParseQuant(a[++i]); break;
                case "--min-elems": opt.MinQuantElements = long.Parse(a[++i]); break;
                case "--max-ctx": opt.MaxCtx = int.Parse(a[++i]); break;
                default:
                    if (!a[i].StartsWith('-')) files.Add(a[i]);
                    else { Console.Error.WriteLine($"error: unknown option '{a[i]}'"); return 2; }
                    break;
            }
        }
        if (files.Count == 0)
        {
            Console.Error.WriteLine("usage: quantize [--append] model-00001-of-00062.safetensors [more.shards] -o model.qw [--config config.json] [quant options]");
            return 2;
        }
        if (output == null) { Console.Error.WriteLine("error: -o out.qw required"); return 2; }
        foreach (var f in files)
            if (!File.Exists(f)) { Console.Error.WriteLine($"error: file not found: {f}"); return 2; }
        if (configPath != null && !File.Exists(configPath)) { Console.Error.WriteLine($"error: config not found: {configPath}"); return 2; }

        bool outExists = File.Exists(output);
        if (outExists && !append)
        {
            Console.Error.WriteLine($"error: {output} already exists — add these tensors to it with --append, or delete it / pick another -o.");
            return 2;
        }
        if (!outExists && append)
            Console.WriteLine("  note     : output does not exist yet — creating it now (first shard; header space reserved for the rest)");
        opt.HeaderReserveBytes = reserveMb << 20;

        Console.WriteLine($"QuantRunner quantize{(append ? " --append" : "")}: {string.Join(" + ", files.Select(Path.GetFileName))} -> {output}");
        var rep = ModelConverter.ConvertSafeTensors(files, configPath, output, opt, append);

        double ratio = rep.OriginalBytes > 0 ? rep.OriginalBytes / (double)Math.Max(1, rep.TotalBytes) : 0;
        Console.WriteLine($"  tensors      : {rep.TensorCount}");
        Console.WriteLine($"  output size  : {Program.Fmt(rep.TotalBytes)}");
        Console.WriteLine($"  source size  : {Program.Fmt(rep.OriginalBytes)}  ({ratio:F2}x smaller)");
        Console.WriteLine($"  time         : {rep.Seconds:F1}s  ({Program.Fmt((long)(rep.TotalBytes / Math.Max(0.1, rep.Seconds)))}/s)");
        var cfg = rep.Config;
        Console.WriteLine($"  arch         : {cfg.Arch}, layers={cfg.NumLayers}, dim={cfg.HiddenSize}, heads={cfg.NumHeads}" +
                          $", kv-heads={cfg.EffectiveKvHeads}, vocab={cfg.VocabSize}" +
                          (cfg.IsMoe ? $", experts={cfg.NumExperts}x{cfg.NumActiveExperts} active" : ""));

        using (var store = ModelStore.Open(output))
        {
            var (present, missing) = Completeness(store, cfg);
            if (missing.Count > 0)
            {
                Console.WriteLine($"  completeness : {present}/{cfg.NumLayers} layers present, {missing.Count} tensors missing");
                Console.WriteLine($"                 -> quantize --append <next-shard> -o {output}");
            }
            else
            {
                Console.WriteLine($"  completeness : {present}/{cfg.NumLayers} layers — model is complete and runnable");
            }
        }
        return 0;
    }

    // ------------------------------------------------------------------ requant (.qw -> smaller .qw)

    /// <summary>
    /// requant model.qw -o model-small.qw [--fmt tri] [--qattn b1x] [--qffn t1e] [--qemb q4_k]
    /// [--qhead q4_k] [--min-elems N] [--exclude a,b] [--only a,b] [--dry-run]
    /// [--from-report report.json --max-rell2 0.35]
    ///
    /// Dequantizes each tensor of an already-quantized model and re-encodes it with a
    /// smaller format in one streaming pass. Tensors whose format is unchanged are copied
    /// byte-exact; norms/router/KDA-gate/mHC groups are never touched; row-aligned targets
    /// only (a misaligned target is skipped with a note, not force-fallback).
    /// --from-report picks per-tensor formats from an `analyze` report: smallest bpw whose
    /// relL2 fits the gate, never larger than the source.
    /// </summary>
    public static int CmdRequant(string[] a)
    {
        string? src = null, output = null, reportPath = null;
        double maxRelL2 = 0.35;
        bool dryRun = false;
        var opt = new RequantOptions();
        for (int i = 0; i < a.Length; i++)
        {
            switch (a[i])
            {
                case "-o" or "--out": output = a[++i]; break;
                case "--fmt": opt.Attention = opt.Ffn = ConvertOptions.ParseQuant(a[++i]); break;
                case "--qattn": opt.Attention = ConvertOptions.ParseQuant(a[++i]); break;
                case "--qffn": opt.Ffn = ConvertOptions.ParseQuant(a[++i]); break;
                case "--qemb": opt.Emb = ConvertOptions.ParseQuant(a[++i]); break;
                case "--qhead": opt.Head = ConvertOptions.ParseQuant(a[++i]); break;
                case "--min-elems": opt.MinQuantElements = long.Parse(a[++i]); break;
                case "--exclude":
                    opt.Exclude = a[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
                    break;
                case "--only":
                    opt.Only = a[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
                    break;
                case "--dry-run": dryRun = true; break;
                case "--from-report": reportPath = a[++i]; break;
                case "--max-rell2": maxRelL2 = double.Parse(a[++i], System.Globalization.CultureInfo.InvariantCulture); break;
                default:
                    if (!a[i].StartsWith('-') && src == null) src = a[i];
                    else { Console.Error.WriteLine($"error: unknown option '{a[i]}'"); return 2; }
                    break;
            }
        }
        if (src == null)
        {
            Console.Error.WriteLine("usage: requant model.qw -o model-small.qw [--fmt tri] [--qattn b1x] [--qffn t1e] [--qemb q4_k] [--dry-run] [--from-report report.json --max-rell2 0.35]");
            return 2;
        }
        if (!File.Exists(src)) { Console.Error.WriteLine($"error: file not found: {src}"); return 2; }
        if (reportPath != null && !File.Exists(reportPath)) { Console.Error.WriteLine($"error: report not found: {reportPath}"); return 2; }
        if (!dryRun && output == null) { Console.Error.WriteLine("error: -o out.qw required (or pass --dry-run to just preview the plan)"); return 2; }
        if (opt.Attention == null && opt.Ffn == null && opt.Emb == null && opt.Head == null && reportPath == null)
        {
            Console.Error.WriteLine("error: nothing to do — pass a target format (--fmt tri, --qattn b1x, --qemb q4_k, ...) or --from-report report.json");
            return 2;
        }
        opt.FromReport = reportPath;
        opt.DryRun = dryRun;

        Console.WriteLine($"QuantRunner requant: {Path.GetFileName(src)} -> {(dryRun ? "(dry run)" : Path.GetFileName(output!))}");
        Console.WriteLine($"  policy     : attention={opt.Attention?.ToString() ?? "keep"}, ffn={opt.Ffn?.ToString() ?? "keep"}, " +
                          $"emb={opt.Emb?.ToString() ?? "keep"}, head={opt.Head?.ToString() ?? "keep"}" +
                          (reportPath != null ? $" | report={Path.GetFileName(reportPath)}, relL2 <= {maxRelL2:0.###}" : ""));
        var res = Requantizer.Run(src, output ?? src, opt);

        if (dryRun)
        {
            Console.WriteLine();
            Console.WriteLine($"  {"tensor",-28} {"from",-7} {"to",-7} {"action",-12} {"bytes",-12}  reason");
            foreach (var r in res.Rows)
                Console.WriteLine($"  {r.Name,-28} {r.From,-7} {r.To,-7} {r.Action,-12} {r.FromBytes + " -> " + r.ToBytes,-12}  {r.Reason ?? ""}");
        }

        double srcBpw = res.SourceBytes * 8.0 / (double)res.Rows.Sum(r => r.NumElements);
        double dstBpw = res.OutputBytes * 8.0 / (double)res.Rows.Sum(r => r.NumElements);
        Console.WriteLine();
        Console.WriteLine($"  tensors    : {res.Rows.Count} total — {res.Requantized} requantized, {res.Passthrough} passthrough, {res.Kept} kept");
        Console.WriteLine($"  size       : {Program.Fmt(res.SourceBytes)} -> {Program.Fmt(res.OutputBytes)}" +
                          $"  ({res.SourceBytes / (double)Math.Max(1, res.OutputBytes):F2}x smaller, {srcBpw:F2} -> {dstBpw:F2} bpw effective)");
        if (!dryRun)
            Console.WriteLine($"  time       : {res.Seconds:F1}s  ({Program.Fmt((long)(res.SourceBytes / Math.Max(0.1, res.Seconds)))}/s)");
        if (res.OutputBytes >= res.SourceBytes)
            Console.WriteLine("  note       : output is not smaller — targets must have fewer bits/weight than the source formats");
        Console.WriteLine();
        Console.WriteLine("  quality    : double quantization adds error on top of the first pass — check `analyze` per-tensor");
        Console.WriteLine("               numbers (or --from-report) and verify with `logits`/generation before shipping.");
        return 0;
    }

    // ------------------------------------------------------------------ analyze (extreme quantization quality study)

    /// <summary>
    /// analyze &lt;model.safetensors&gt; [-o report.json] [--formats tri,b1,b1x,t1e,t1e_lo,q3_k,q4_k,q8]
    /// [--max-elems N] [--min-elems N] [--top K]
    ///
    /// Measures how much weight quality survives every candidate format — including the
    /// sub-TRI extreme formats (B1 / B1X / T1E / T1E_LO) — per tensor, by quantizing each
    /// tensor in memory and scoring ||W − Q||_F / ||W||_F + SNR. This is the tool for the
    /// "how good can a model still be below 1.58 bit" question: it produces the exact
    /// numbers to pick per-tensor formats from, on real checkpoint weights.
    /// </summary>
    public static int CmdAnalyze(string[] a)
    {
        string? path = null, reportPath = null;
        var formats = new List<QuantFormat>();
        long maxElems = QuantAnalyzer.DefaultMaxElems, minElems = 4096;
        int top = 40;
        for (int i = 0; i < a.Length; i++)
        {
            switch (a[i])
            {
                case "-o" or "--report": reportPath = a[++i]; break;
                case "--formats":
                    foreach (var s in a[++i].Split(',', StringSplitOptions.RemoveEmptyEntries))
                        formats.Add(ConvertOptions.ParseQuant(s.Trim()));
                    break;
                case "--max-elems": maxElems = long.Parse(a[++i]); break;
                case "--min-elems": minElems = long.Parse(a[++i]); break;
                case "--top": top = int.Parse(a[++i]); break;
                default:
                    if (!a[i].StartsWith('-') && path == null) path = a[i];
                    else { Console.Error.WriteLine($"error: unknown option '{a[i]}'"); return 2; }
                    break;
            }
        }
        if (path == null)
        {
            Console.Error.WriteLine("usage: analyze <model.safetensors> [-o report.json] [--formats tri,b1,b1x,t1e,t1e_lo,q3_k,q4_k,q8_0] [--max-elems 1048576] [--min-elems 4096] [--top 40]");
            return 2;
        }
        if (!File.Exists(path)) { Console.Error.WriteLine($"error: file not found: {path}"); return 2; }
        if (formats.Count == 0) formats.AddRange(QuantAnalyzer.DefaultFormats);
        formats = formats.Distinct().OrderByDescending(QuantAnalyzer.BitsPerWeight).ToList();

        Console.WriteLine($"QuantRunner analyze: {Path.GetFileName(path)}");
        Console.WriteLine($"  formats    : {string.Join(", ", formats)}");
        Console.WriteLine($"  budget     : {Program.Fmt(maxElems)} weights per tensor (evenly row-spread subsample)");
        Console.WriteLine($"  metric     : relL2 = ||W - Q||_F / ||W||_F  (weight-only proxy; lower is better)");
        var results = QuantAnalyzer.AnalyzeFile(path, formats, maxElems, minElems, msg => Console.WriteLine(msg));
        if (results.Count == 0)
        {
            Console.WriteLine("  no block-alignable tensors >= --min-elems found");
            return 1;
        }
        QuantAnalyzer.PrintReport(results, top);
        if (reportPath != null)
        {
            File.WriteAllText(reportPath, QuantAnalyzer.ToJson(results));
            Console.WriteLine($"\n  report     : {reportPath}");
        }
        return 0;
    }

    /// <summary>Which required tensors are absent from the store (per the config).</summary>
    internal static (int LayersPresent, List<string> Missing) Completeness(ModelStore store, ModelConfig cfg)
    {
        var missing = new List<string>();
        void Need(List<string> bucket, params string[] names)
        {
            foreach (var n in names) if (!store.Has(n)) bucket.Add(n);
        }
        if (!store.Has("tok_emb")) missing.Add("tok_emb");
        int present = 0;
        for (int l = 0; l < cfg.NumLayers; l++)
        {
            var lm = new List<string>();
            Need(lm, $"l{l}.attn_norm", $"l{l}.ffn_norm");

            // attention tensors by layer kind (GLM-5.x hybrids vs legacy GQA)
            switch (cfg.AttnKind(l))
            {
                case "kda":
                    Need(lm, $"l{l}.kda.q", $"l{l}.kda.k", $"l{l}.kda.v", $"l{l}.kda.o",
                        $"l{l}.kda.qconv", $"l{l}.kda.kconv", $"l{l}.kda.vconv",
                        $"l{l}.kda.fa", $"l{l}.kda.fb", $"l{l}.kda.ga", $"l{l}.kda.gb", $"l{l}.kda.b",
                        $"l{l}.kda.alog", $"l{l}.kda.dt", $"l{l}.kda.onorm");
                    break;
                case "mla":
                    Need(lm, $"l{l}.mla.qa", $"l{l}.mla.qanorm", $"l{l}.mla.qb",
                        $"l{l}.mla.kva", $"l{l}.mla.kvanorm", $"l{l}.mla.kvb", $"l{l}.mla.o");
                    break;
                default:
                    Need(lm, $"l{l}.wq", $"l{l}.wk", $"l{l}.wv", $"l{l}.wo");
                    break;
            }

            // mHC hyper-connection params when the model uses streams
            if (cfg.HasMhc)
                Need(lm, $"l{l}.hc_attn_fn", $"l{l}.hc_attn_base", $"l{l}.hc_attn_scale",
                    $"l{l}.hc_ffn_fn", $"l{l}.hc_ffn_base", $"l{l}.hc_ffn_scale");

            bool moeLayer = store.Has($"l{l}.router") || store.Has($"l{l}.e0.w_gate");
            if (cfg.IsMoe && moeLayer)
            {
                Need(lm, $"l{l}.router");
                if (cfg.MoeSigmoid) Need(lm, $"l{l}.router_bias");
                for (int e = 0; e < cfg.NumExperts; e++)
                    Need(lm, $"l{l}.e{e}.w_gate", $"l{l}.e{e}.w_up", $"l{l}.e{e}.w_down");
                if (cfg.MoeSigmoid && cfg.NumSharedExperts > 0)
                    Need(lm, $"l{l}.sw_gate", $"l{l}.sw_up", $"l{l}.sw_down");
            }
            else
                Need(lm, $"l{l}.w_gate", $"l{l}.w_up", $"l{l}.w_down");   // dense FFN (also first-k-dense MoE layers)
            if (lm.Count == 0) present++; else missing.AddRange(lm);
        }
        return (present, missing);
    }

    /// <summary>Debug: run the engine over explicit token ids and print last-position logits.
    /// Used by the numeric cross-validation against the python reference forward.</summary>
    public static int CmdLogits(string[] a)
    {
        string? path = null;
        string? tokensArg = null;
        int top = 12;
        bool all = false;
        for (int i = 0; i < a.Length; i++)
        {
            switch (a[i])
            {
                case "--tokens": tokensArg = a[++i]; break;
                case "--top": top = int.Parse(a[++i]); break;
                case "--all": all = true; break;
                default: if (!a[i].StartsWith('-')) path = a[i]; break;
            }
        }
        if (path == null) { Console.Error.WriteLine("usage: logits model.qw --tokens 1,2,3 [--top 12] [--all]"); return 2; }
        var ids = new List<int>();
        if (tokensArg != null)
            foreach (var s in tokensArg.Split(',', StringSplitOptions.RemoveEmptyEntries))
                ids.Add(int.Parse(s.Trim()));
        if (ids.Count == 0) { Console.Error.WriteLine("error: --tokens required"); return 2; }

        using var store = ModelStore.Open(path);
        using var engine = new StreamingEngine(store, new EngineOptions { PrefetchBudgetBytes = 0 });
        float[] logits = Array.Empty<float>();
        for (int pos = 0; pos < ids.Count; pos++) logits = engine.Forward(ids[pos], pos);

        Console.WriteLine($"tokens   : {string.Join(',', ids)}");
        Console.WriteLine($"last logits (top {top} of {logits.Length}):");
        var order = Enumerable.Range(0, logits.Length).OrderByDescending(j => logits[j]).Take(top);
        foreach (var j in order)
            Console.WriteLine($"  [{j,6}] {logits[j],10:F5}");
        if (all)
        {
            Console.WriteLine("all logits:");
            var sb = new System.Text.StringBuilder();
            for (int j = 0; j < logits.Length; j++)
            {
                sb.Append(logits[j].ToString("F6", System.Globalization.CultureInfo.InvariantCulture));
                sb.Append(j + 1 < logits.Length ? ',' : '\n');
            }
            Console.WriteLine(sb.ToString());
        }
        else
        {
            // machine-readable line for diffing against the python reference
            var sb = new System.Text.StringBuilder();
            foreach (var j in Enumerable.Range(0, logits.Length).OrderByDescending(j => logits[j]).Take(top))
            {
                sb.Append(j).Append(':').Append(logits[j].ToString("F6", System.Globalization.CultureInfo.InvariantCulture));
                sb.Append(' ');
            }
            Console.WriteLine($"top12 {sb.ToString().TrimEnd()}");
        }
        return 0;
    }

    // ------------------------------------------------------------------ run

    public static int CmdRun(string[] a)
    {
        if (a.Length == 0) { Console.Error.WriteLine("usage: run <model.qw | model.gguf> [-p \"prompt\"] [--tokenizer tokenizer.json] [--draft draft.qw] [options]"); return 2; }
        string modelPath = a[0];
        string? tokPath = null, prompt = null, draftPath = null;
        int maxNew = 256, prefetchMb = 1024, maxCtx = -1, seed = -1, topk = 40, draftTokens = 4;
        float temp = 0.8f, topp = 0.95f, repPenalty = 1.1f, minp = 0f;
        bool noPrefetch = false;

        for (int i = 1; i < a.Length; i++)
        {
            switch (a[i])
            {
                case "--tokenizer" or "-t": tokPath = a[++i]; break;
                case "-p" or "--prompt": prompt = a[++i]; break;
                case "-n" or "--max-new": maxNew = int.Parse(a[++i]); break;
                case "--temp": temp = float.Parse(a[++i]); break;
                case "--topk": topk = int.Parse(a[++i]); break;
                case "--topp": topp = float.Parse(a[++i]); break;
                case "--minp": minp = float.Parse(a[++i]); break;
                case "--seed": seed = int.Parse(a[++i]); break;
                case "--repeat-penalty": repPenalty = float.Parse(a[++i]); break;
                case "--draft": draftPath = a[++i]; break;      // greedy speculative decoding
                case "--draft-tokens": draftTokens = int.Parse(a[++i]); break;
                case "--no-prefetch": noPrefetch = true; break;
                case "--prefetch-mb": prefetchMb = int.Parse(a[++i]); break;
                case "--max-ctx": maxCtx = int.Parse(a[++i]); break;
            }
        }

        if (tokPath != null && !File.Exists(tokPath))
        { Console.Error.WriteLine($"error: tokenizer file not found: {tokPath}"); return 2; }
        if (draftPath != null && !File.Exists(draftPath))
        { Console.Error.WriteLine($"error: draft model not found: {draftPath}"); return 2; }

        // GGUF source: the engine runs from .qw, so convert once into a cached .qw beside the
        // gguf (formats kept as-is) and reuse that cache on every later run
        string qwPath = modelPath;
        if (modelPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            if (!File.Exists(modelPath)) { Console.Error.WriteLine($"error: file not found: {modelPath}"); return 2; }
            qwPath = GgufRun.EnsureQwFromGguf(modelPath).QwPath;
        }
        else if (!File.Exists(qwPath))
        { Console.Error.WriteLine($"error: file not found: {qwPath}"); return 2; }

        if (prompt == null)
        {
            Console.Error.WriteLine("reading prompt from stdin... (end with Ctrl+D)");
            prompt = Console.In.ReadToEnd().TrimEnd('\n');
        }

        Console.Error.WriteLine($"loading {qwPath} ...");
        using var store = ModelStore.Open(qwPath);
        var cfg = ModelConfig.FromJson(store.ConfigJson);
        using var engine = new StreamingEngine(store, new EngineOptions
        {
            PrefetchBudgetBytes = noPrefetch ? 0 : (long)prefetchMb << 20,
            MaxCtx = maxCtx
        });

        // tokenizer resolution: explicit --tokenizer -> tokenizer.json beside the model ->
        // embedded GGUF metadata (gguf carries it, and a .qw imported from gguf keeps it)
        BpeTokenizer tok;
        if (tokPath != null)
        {
            tok = BpeTokenizer.Load(tokPath);
            Console.Error.WriteLine($"tokenizer  : {tokPath} ({tok.VocabSize} tokens)");
        }
        else
        {
            string beside = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(qwPath))!, "tokenizer.json");
            if (File.Exists(beside))
            {
                tok = BpeTokenizer.Load(beside);
                Console.Error.WriteLine($"tokenizer  : {beside} ({tok.VocabSize} tokens)");
            }
            else
            {
                var embedded = GgufRun.LoadTokenizerFromStore(store, out string reason);
                if (embedded == null)
                {
                    Console.Error.WriteLine("error: no tokenizer available:");
                    Console.Error.WriteLine($"  {reason}");
                    Console.Error.WriteLine("  pass --tokenizer tokenizer.json (from the original HF repo), or run a model");
                    Console.Error.WriteLine("  imported from GGUF — those carry their tokenizer inside the file");
                    return 2;
                }
                tok = embedded;
                Console.Error.WriteLine($"tokenizer  : built from the model's embedded GGUF metadata ({tok.VocabSize} tokens)");
            }
        }

        var sampler = new Sampler { Temperature = temp, TopK = topk, TopP = topp, MinP = minp, Seed = seed, RepeatPenalty = repPenalty };

        var ids = tok.Encode(prompt);
        if (cfg.NumKvHeads > 0) { /* GQA info only */ }
        Console.Error.WriteLine($"arch={cfg.Arch} layers={cfg.NumLayers} dim={cfg.HiddenSize} " +
                                 (cfg.IsMoe ? $"moe={cfg.NumExperts}x{cfg.NumActiveExperts} " : "") +
                                 $"prompt={ids.Length} tok, prefetch={(noPrefetch ? "off" : $"{prefetchMb} MiB/layer")}");

        var sw = Stopwatch.StartNew();
        int[] gen;
        (int Drafted, int Accepted, int Rounds) spec = (0, 0, 0);
        if (draftPath != null)
        {
            // greedy speculative decoding: output == the target's plain greedy trajectory
            var dstore = ModelStore.Open(draftPath);
            using var deng = new StreamingEngine(dstore, new EngineOptions
            {
                PrefetchBudgetBytes = noPrefetch ? 0 : (long)prefetchMb << 20,
                MaxCtx = maxCtx
            });
            var runner = new SpeculativeRunner(engine, deng) { DraftTokens = draftTokens };
            Console.Error.WriteLine($"speculative: draft={Path.GetFileName(draftPath)} " +
                                     $"({deng.Config.NumLayers}L/{deng.Config.HiddenSize}D), {draftTokens} tok/round, greedy (repeat penalty honored; --temp/topk/topp ignored)");
            var specSampler = new Sampler { Temperature = 0f, RepeatPenalty = repPenalty, RepeatWindow = 64 };
            gen = runner.Generate(ids, maxNew, specSampler, (id, _) =>
            {
                Console.Write(tok.DecodeToken(id));
                if (Console.Out is StreamWriter swc && !swc.AutoFlush) swc.AutoFlush = true;
            });
            spec = runner.LastStats;
        }
        else
        {
            gen = engine.Generate(ids, maxNew, sampler, (id, idx, tps) =>
            {
                Console.Write(tok.DecodeToken(id));
                if (Console.Out is StreamWriter swc && !swc.AutoFlush) swc.AutoFlush = true;
            });
        }
        sw.Stop();

        var st = engine.LastStats;
        Console.WriteLine();
        Console.Error.WriteLine($"""
            ── stats ─────────────────────────────────────────────
             prompt    : {st.PromptTokens} tok in {st.PromptSeconds:F2}s ({(st.PromptSeconds > 0 ? st.PromptTokens / st.PromptSeconds : 0):F1} tok/s prefill)
             generated : {st.GeneratedTokens} tok in {sw.Elapsed.TotalSeconds - st.PromptSeconds:F2}s
             decode    : {st.TokensPerSecond:F2} tok/s
             streamed  : {Program.Fmt(st.PrefetchBytes)} through prefetch, stall {st.PrefetchWaitMs:F0} ms
             kv cache  : {Program.Fmt(st.KvBytes)} (paged, {PagedKvCache.PageTokens} tok/page)
            ──────────────────────────────────────────────────────
            """);
        if (spec.Rounds > 0)
            Console.Error.WriteLine($" speculative: {spec.Rounds} rounds, {spec.Accepted}/{spec.Drafted} drafts accepted " +
                                     $"({(spec.Drafted > 0 ? 100.0 * spec.Accepted / spec.Drafted : 0):F0}%) — output = target greedy, exactly");
        return 0;
    }

    // ------------------------------------------------------------------ bench

    public static int CmdBench(string[] a)
    {
        if (a.Length == 0) { Console.Error.WriteLine("usage: bench model.qw [--steps 6] [--no-prefetch]"); return 2; }
        string path = a[0];
        int steps = 6, prefetchMb = 1024;
        bool noPrefetch = false;
        for (int i = 1; i < a.Length; i++)
        {
            switch (a[i])
            {
                case "--steps": steps = int.Parse(a[++i]); break;
                case "--prefetch-mb": prefetchMb = int.Parse(a[++i]); break;
                case "--no-prefetch": noPrefetch = true; break;
            }
        }

        var proc = Process.GetCurrentProcess();
        long ws0 = proc.WorkingSet64;

        Console.WriteLine("╔══════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║            QuantRunner streaming benchmark                       ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════════╝");
        var store = ModelStore.Open(path);
        var cfg = ModelConfig.FromJson(store.ConfigJson);
        long totalBytes = store.Tensors.Values.Sum(t => t.SizeBytes);
        long totalParams = store.Tensors.Values.Sum(t => t.NumElements);
        long layerBytes = SumLayer(store, 0);
        Console.WriteLine($" model        : {cfg.Arch}, {cfg.NumLayers} layers, {totalParams:N0} params, {Program.Fmt(totalBytes)}");
        Console.WriteLine($" layer bytes  : {Program.Fmt(layerBytes)} (dense{(cfg.IsMoe ? "; MoE layers larger" : "")})");
        Console.WriteLine($" working set  : {Program.Fmt(ws0)} before engine");
        Console.WriteLine();

        // ---- 1. raw disk streaming (what every layer must pay per token) ----
        Console.WriteLine("[1] sequential layer stream (per-layer cold-ish read, 1 MiB chunks)");
        double totalSecs = 0;
        for (int l = 0; l < Math.Min(cfg.NumLayers, 64); l++)
        {
            double secs = StreamBytes(path, SumLayer(store, l));
            totalSecs += secs;
            double mbps = SumLayer(store, l) / 1048576.0 / Math.Max(secs, 1e-9);
            if (l < 6 || l == cfg.NumLayers - 1)
                Console.WriteLine($"     layer {l,3}: {SumLayer(store, l) / 1048576.0,8:F1} MiB in {secs * 1000,7:F2} ms  = {mbps,8:F0} MiB/s");
        }
        int measured = Math.Min(cfg.NumLayers, 64);
        Console.WriteLine($"     avg stream time per layer : {totalSecs / measured * 1000:F2} ms");
        Console.WriteLine($"     avg throughput            : {SumLayer(store, 0) * measured / 1048576.0 / totalSecs:F0} MiB/s");
        Console.WriteLine($"     => synchronous streaming ceiling: {1000.0 / (totalSecs / measured):F1} tok/s (AirLLM-style, no overlap)");
        Console.WriteLine();

        // ---- 2. forward pass: prefetch ON vs OFF ----
        double measuredOverlap = 0;
        foreach (bool withPrefetch in new[] { !noPrefetch, false })
        {
            if (!withPrefetch && noPrefetch) continue;
            var engStore = ModelStore.Open(path);
            using var engine = new StreamingEngine(engStore, new EngineOptions
            {
                PrefetchBudgetBytes = withPrefetch ? (long)prefetchMb << 20 : 0
            });
            var sw = Stopwatch.StartNew();
            for (int s = 0; s < steps; s++)
                engine.Forward(1 + s % 1024, s);
            sw.Stop();

            double msPerTok = sw.Elapsed.TotalMilliseconds / steps;
            var pf = engine.Prefetch;
            double waitMs = pf != null ? pf.WaitTicks * 1000.0 / Stopwatch.Frequency : 0;
            double overlap = pf != null ? Math.Max(0, 1 - waitMs / Math.Max(0.001, sw.Elapsed.TotalMilliseconds)) : 1.0;
            if (withPrefetch) measuredOverlap = overlap;
            Console.WriteLine($"[2] forward pass  prefetch={(withPrefetch ? $"ON  ({prefetchMb} MiB/layer)" : "OFF (mmap only)     ")}");
            Console.WriteLine($"     {msPerTok,8:F2} ms/token  =  {1000.0 / msPerTok,7:F2} tok/s  over {steps} steps");
            if (pf != null)
                Console.WriteLine($"     I/O stall {waitMs,7:F1} ms  ({100 * waitMs / Math.Max(0.001, sw.Elapsed.TotalMilliseconds):F1}% of wall)   overlap efficiency {overlap * 100:F1}%   streamed {Program.Fmt(pf.BytesStreamed)}");
            long wsNow = Process.GetCurrentProcess().WorkingSet64;
            Console.WriteLine($"     working set {Program.Fmt(wsNow)}  (independent of model size: {Program.Fmt(totalBytes)} model)");
            Console.WriteLine();
            engine.Dispose();
        }

        // ---- 3. comparison table ----
        Console.WriteLine("[3] vs AirLLM (qualitative + measured-on-this-machine)");
        Console.WriteLine("     ┌────────────────────────┬──────────────────────────┬──────────────────────────┐");
        Console.WriteLine("     │ aspect                 │ AirLLM                   │ QuantRunner (measured)   │");
        Console.WriteLine("     ├────────────────────────┼──────────────────────────┼──────────────────────────┤");
        Console.WriteLine("     │ layer streaming        │ synchronous              │ async prefetch pipeline  │");
        Console.WriteLine($"     │ I/O on critical path   │ 100% of layer time       │ measured {100 * (1 - measuredOverlap),4:F1}% stall    │");
        Console.WriteLine("     │ dequantize             │ fp16 round-trip in torch │ fused dequant+dot kernel │");
        Console.WriteLine("     │ allocations per layer  │ torch tensor per weight  │ zero (pointer arithmetic)│");
        Console.WriteLine("     │ runtime                │ python + pytorch         │ .NET 8, no dependencies  │");
        Console.WriteLine("     │ quant formats          │ gguf subset              │ Q2K..Q6K + Q4_0/Q8_0 + TRI│");
        Console.WriteLine("     └────────────────────────┴──────────────────────────┴──────────────────────────┘");
        Console.WriteLine("     measure both engines on the same machine/model for honest tok/s numbers.");
        return 0;
    }

    private static long SumLayer(ModelStore store, int layer)
    {
        string prefix = $"l{layer}.";
        long sum = 0;
        foreach (var t in store.Tensors.Values)
            if (t.Name.StartsWith(prefix, StringComparison.Ordinal) &&
                !t.Name.EndsWith("norm", StringComparison.Ordinal) && !t.Name.EndsWith("router", StringComparison.Ordinal))
                sum += t.SizeBytes;
        return sum;
    }

    private static double StreamBytes(string path, long bytes)
    {
        var buf = new byte[1 << 20];
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
        var sw = Stopwatch.StartNew();
        long off = 0;
        // stream a layer-sized span from the middle of the file (data region)
        fs.Seek(Math.Max(0, fs.Length / 4), SeekOrigin.Begin);
        long remaining = bytes;
        while (remaining > 0)
        {
            int chunk = (int)Math.Min(buf.Length, remaining);
            int r = fs.Read(buf, 0, chunk);
            if (r <= 0) break;
            remaining -= r;
            off += r;
        }
        sw.Stop();
        return sw.Elapsed.TotalSeconds;
    }

    // ------------------------------------------------------------------ selftest

    public static int CmdSelftest()
    {
        Console.WriteLine("QuantRunner selftest");
        int pass = 0, fail = 0;
        void Check(string name, Action fn)
        {
            try { fn(); pass++; Console.WriteLine($"  [ok]   {name}"); }
            catch (Exception ex) { fail++; Console.WriteLine($"  [FAIL] {name}: {ex.Message}"); }
        }

        Check("f16 round trip", TestsSelf.F16RoundTrip);
        Check("quant round trip RMS (all formats)", TestsSelf.QuantRoundTrip);
        Check("rANS entropy coder round trip (fuzz)", TestsSelf.EntropyCoderRoundTrip);
        Check("extreme low-bit round trip + error ordering (B1/B1X/T1E)", TestsSelf.ExtremeRoundTripAndOrdering);
        Check("TRI optimal-threshold upgrade beats legacy heuristic", TestsSelf.TriUpgradeBeatsLegacy);
        Check("extreme formats end-to-end through the engine", TestsSelf.ExtremeFormatsEngine);
        Check("quant analyzer scoring + ranking", TestsSelf.QuantAnalyzerSmoke);
        Check("dot kernels match dequant reference", TestsSelf.DotMatches);
        Check(".qw write/read round trip", TestsSelf.QwRoundTrip);
        Check("safetensors single-file quantize", TestsSelf.SafeTensorsFileConvert);
        Check("safetensors quantize with sub-TRI formats (tri/b1/b1x/t1e/t1e_lo)", TestsSelf.QuantizeSubTriFormats);
        Check("requant downgrade + byte-exact passthrough + dry run", TestsSelf.RequantDowngradeAndPassthrough);
        Check("requant smart formats from analyze report", TestsSelf.RequantSmartFromReport);
        Check("ggml legacy block layouts match the llama.cpp reference (hand-built blocks)", TestsSelf.GgmlLegacyCompat);
        Check("gguf Q4_0 import + qw->gguf export round trip + tokenizer metadata restore", TestsSelf.GgufQ4ImportAndWriterRoundTrip);
        Check("qw -> HF safetensors export (shards + config.json + index.json)", TestsSelf.HfExportFromQw);
        Check("gguf -> HF export with BPE tokenizer restore", TestsSelf.HfExportFromGguf);
        Check("safetensors shard assembly (--append)", TestsSelf.SafeTensorsShardAppend);
        Check("fused projection splits (qkv / gate_up / GLM-interleaved)", TestsSelf.FusedProjectionSplits);
        Check("fp8 e4m3 decode/encode + block-scale reader", TestsSelf.Fp8E4M3Roundtrip);
        Check("mHC sinkhorn + collapse/combine", TestsSelf.MhcSinkhornAndCombine);
        Check("MLA absorbed attention == expanded reference", TestsSelf.MlaAbsorbedVsExpanded);
        Check("KDA conv step == direct convolution", TestsSelf.KdaConvMatchesDirect);
        Check("GLM-5.x config parse + kind-aware name mapping", TestsSelf.MlaRefusalAndNestedConfig);
        Check("append heals degenerate stored config", TestsSelf.AppendHealsDegenerateConfig);
        Check("K-quant row-alignment fallback (q4_k on 128-wide rows)", TestsSelf.KQuantRowFallback);
        Check("appender honors 64-byte aligned offsets (GLM-5.3 odd-size regression)", TestsSelf.AppenderOffsetAlignment);
        Check("engine end-to-end (synthetic, dense)", TestsSelf.EngineDense);
        Check("engine end-to-end (synthetic, MoE)", TestsSelf.EngineMoe);
        Check("engine end-to-end (synthetic, GLM-5 hybrid: mHC + KDA + MLA + GLM-MoE)", TestsSelf.EngineGlm5Hybrid);
        Check("speculative decoding == plain greedy (unrelated draft, corrections + rollback)", TestsSelf.SpeculativeGreedyEquivalence);
        Check("bpe encode/decode round trip", TestsSelf.BpeRoundTrip);
        Check("paged kv cache", TestsSelf.KvCache);
        Check("param report: bucket classification across qw/gguf/HF names", TestsSelf.ParamBuckets);
        Check("param report: MoE active-per-token + human formatting", TestsSelf.ParamMoeAndHuman);
        Check("tokenizer embedded in .qw / GGUF metadata restores without --tokenizer", TestsSelf.GgufEmbeddedTokenizer);
        Check("run model.gguf: as-is cache conversion + embedded tokenizer + generation", TestsSelf.GgufRunFromGguf);

        Console.WriteLine($"\n{pass} passed, {fail} failed");
        return fail == 0 ? 0 : 1;
    }
}

internal static class TestsSelf
{
    public static void F16RoundTrip()
    {
        var vals = new float[] { 0f, 1f, -1.5f, 65504f, 1e-5f, 0.333f, 1234.5678f, -0.0001f };
        foreach (var v in vals)
        {
            float back = Dtypes.F16ToFloat(Dtypes.FloatToF16(v));
            bool sub = MathF.Abs(v) < 6.1e-5f;               // subnormal half range
            float tol = sub ? 6e-8f : MathF.Abs(v) * 0.0005f;
            if (Math.Abs(back - v) > tol) throw new Exception($"f16 rt {v} -> {back}");
        }
        if (Dtypes.F16ToFloat(Dtypes.FloatToF16(65504f)) != 65504f) throw new Exception("max half");
    }

    public static void QuantRoundTrip()
    {
        var rng = new Random(7);
        foreach (var f in new[] { QuantFormat.Q4_0, QuantFormat.Q8_0, QuantFormat.Q2_K, QuantFormat.Q3_K,
                                  QuantFormat.Q4_K, QuantFormat.Q5_K, QuantFormat.Q6_K, QuantFormat.TRI, QuantFormat.I8_BLK,
                                  QuantFormat.B1, QuantFormat.B1X, QuantFormat.T1E, QuantFormat.T1E_LO })
        {
            var (bs, bb) = Dtypes.BlockSpec(f);
            int n = bs * 16;
            var x = new float[n];
            for (int i = 0; i < n; i++) x[i] = Synth.NextGaussian(rng) * 0.5f;
            float amax = x.Max(Math.Abs);
            var bytes = new byte[Dtypes.TensorBytes(n, f)];
            BlockQuantizer.Quantize(f, x, bytes);
            var y = new float[n];
            BlockDequantizer.Dequantize(f, bytes, y);

            double se = 0, seRef = 0;
            for (int i = 0; i < n; i++) { double d = y[i] - x[i]; se += d * d; seRef += x[i] * x[i]; }
            double rms = Math.Sqrt(se / n) / Math.Sqrt(Math.Max(seRef / n, 1e-12));
            double tol = f switch
            {
                QuantFormat.Q8_0 or QuantFormat.I8_BLK => 0.02,
                QuantFormat.Q6_K => 0.03,
                QuantFormat.Q5_K => 0.05,
                QuantFormat.Q4_K or QuantFormat.Q4_0 => 0.11,
                QuantFormat.Q3_K => 0.20,
                QuantFormat.Q2_K => 0.36,
                QuantFormat.TRI => 0.55,
                QuantFormat.B1X => 0.62,
                QuantFormat.T1E => 0.72,
                QuantFormat.B1 => 0.66,
                QuantFormat.T1E_LO => 0.80,
                _ => 0.5
            };
            if (rms > tol) throw new Exception($"{f}: rel RMS {rms:F3} > {tol}");
        }
    }

    /// <summary>rANS round trip must be bit-exact over randomized frequency splits and symbol
    /// streams, including degenerate distributions (single-symbol blocks, 255/1 splits).</summary>
    public static void EntropyCoderRoundTrip()
    {
        var rng = new Random(41);
        for (int trial = 0; trial < 400; trial++)
        {
            uint f0 = (uint)rng.Next(257);
            uint f1 = (uint)rng.Next(257 - (int)f0);
            int n = trial < 8 ? 256 : rng.Next(1, 513);
            var sym = new byte[n];
            for (int i = 0; i < n; i++)
            {
                int pick = rng.Next(256);
                // draw from the actual distribution so zero-freq symbols never occur
                sym[i] = pick < f0 ? (byte)0 : pick < f0 + f1 ? (byte)1 : (byte)2;
            }
            // degenerate explicit cases
            if (trial == 0) { f0 = 256; f1 = 0; Array.Fill(sym, (byte)0); }
            if (trial == 1) { f0 = 0; f1 = 256; Array.Fill(sym, (byte)1); }
            if (trial == 2) { f0 = 0; f1 = 0; Array.Fill(sym, (byte)2); }
            if (trial == 3) { f0 = 255; f1 = 1; for (int i = 0; i < n; i++) sym[i] = i == n - 1 ? (byte)1 : (byte)0; }

            var buf = new byte[1024];
            int used = EntropyCoder.Encode3(sym, f0, f1, buf);
            if (used < 0) throw new Exception($"trial {trial}: encode overflowed 1024B (f0={f0} f1={f1} n={n})");
            var back = new byte[n + 8];
            EntropyCoder.Decode3(buf.AsSpan(0, used), n, f0, f1, back);
            for (int i = 0; i < n; i++)
                if (back[i] != sym[i]) throw new Exception($"trial {trial}: symbol {i}: {back[i]} != {sym[i]} (f0={f0}, f1={f1})");

            // decoder must also work when the payload is zero-padded to the block budget
            var padded = new byte[used + 8];   // zero-filled tail, size independent of stream length
            buf.AsSpan(0, used).CopyTo(padded);
            EntropyCoder.Decode3(padded, n, f0, f1, back);
            for (int i = 0; i < n; i++)
                if (back[i] != sym[i]) throw new Exception($"trial {trial} (padded): symbol {i} mismatch");
        }
    }

    /// <summary>
    /// The sub-TRI ladder must behave as designed:
    ///   - B1X error is monotonically <= B1 (exceptions only remove the worst residual);
    ///   - on outlier-heavy weights B1X clearly wins over B1 (exceptions catch the tails);
    ///   - T1E blocks always round-trip exactly through header + entropy stream;
    ///   - more stream budget (T1E vs T1E_LO) never costs more error;
    ///   - every extreme format stays far better than sign-flipping garbage.
    /// </summary>
    public static void ExtremeRoundTripAndOrdering()
    {
        var rng = new Random(1234);
        const int n = 256 * 24;   // 24 blocks

        // heavy-tailed weights (real LLM weight blocks are far from Gaussian):
        // product of two gaussians + rare huge outliers, mirroring BiLLM's motivating setup
        float[] HeavyTail()
        {
            var w = new float[n];
            for (int i = 0; i < n; i++)
            {
                float v = Synth.NextGaussian(rng) * Synth.NextGaussian(rng) * 0.5f;
                if (rng.NextDouble() < 0.004) v *= 12f;           // planted outliers (~1 in 250)
                w[i] = v;
            }
            return w;
        }

        double Err(QuantFormat f, float[] w)
        {
            var q = new byte[Dtypes.TensorBytes(w.Length, f)];
            BlockQuantizer.Quantize(f, w, q);
            var y = new float[w.Length];
            BlockDequantizer.Dequantize(f, q, y);
            double se = 0, sr = 0;
            for (int i = 0; i < w.Length; i++) { double d = y[i] - w[i]; se += d * d; sr += (double)w[i] * w[i]; }
            return Math.Sqrt(se / Math.Max(sr, 1e-30));
        }

        // 1) outlier-heavy block: exceptions must capture what pure binary cannot
        var wh = HeavyTail();
        double eB1 = Err(QuantFormat.B1, wh);
        double eB1X = Err(QuantFormat.B1X, wh);
        if (eB1X >= eB1) throw new Exception($"B1X ({eB1X:F4}) should beat B1 ({eB1:F4}) on outlier-heavy weights");
        // ~1 planted outlier per block + the natural product-gaussian tail vs 4 exceptions:
        // measured gain is ~22-25%; the bar asserts a clear win without pinning exact noise
        if (eB1X > 0.80 * eB1) throw new Exception($"B1X barely improved on outliers: {eB1X:F4} vs {eB1:F4}");

        // 2) T1E round trip exactness is covered by dequantize (symbols must reproduce);
        //    here: T1E(_LO) must also beat plain B1 on heavy tails (ternary keeps near-zeros at 0)
        double eT1E = Err(QuantFormat.T1E, wh);
        double eT1ELo = Err(QuantFormat.T1E_LO, wh);
        if (eT1E > 0.9) throw new Exception($"T1E error out of range on heavy tails: {eT1E:F4}");
        if (eT1ELo > 1.0) throw new Exception($"T1E_LO error out of range on heavy tails: {eT1ELo:F4}");
        if (eT1E > eT1ELo + 0.01) throw new Exception($"T1E ({eT1E:F4}) should not lose to T1E_LO ({eT1ELo:F4})");

        // 3) Gaussian block: B1X still <= B1 (monotone by construction)
        var wg = new float[n];
        for (int i = 0; i < n; i++) wg[i] = Synth.NextGaussian(rng) * 0.5f;
        double gB1 = Err(QuantFormat.B1, wg), gB1X = Err(QuantFormat.B1X, wg);
        if (gB1X > gB1 + 1e-6) throw new Exception($"B1X ({gB1X:F4}) regressed vs B1 ({gB1:F4}) on gaussian");

        // 4) stored sizes must match the declared block specs exactly
        foreach (var f in new[] { QuantFormat.B1, QuantFormat.B1X, QuantFormat.T1E, QuantFormat.T1E_LO })
        {
            var (bs, bb) = Dtypes.BlockSpec(f);
            var q = new byte[Dtypes.TensorBytes(wh.Length, f)];
            BlockQuantizer.Quantize(f, wh, q);
            if (q.Length != wh.Length / bs * bb) throw new Exception($"{f}: TensorBytes mismatch");
            double bpw = bb * 8.0 / bs;
            if (bpw >= 2.25) throw new Exception($"{f}: {bpw:F3} bpw is not below TRI's 2.25 stored bpw");
        }
    }

    /// <summary>The upgraded TRI (threshold grid + optimal scale) must never lose to the legacy
    /// mean|w| heuristic, and must win clearly on outlier-shaped blocks.</summary>
    public static void TriUpgradeBeatsLegacy()
    {
        var rng = new Random(77);
        const int n = 64 * 32;
        double totNew = 0, totOld = 0;
        for (int t = 0; t < 3; t++)
        {
            var w = new float[n];
            for (int i = 0; i < n; i++)
            {
                w[i] = t switch
                {
                    0 => Synth.NextGaussian(rng) * 0.5f,                                   // gaussian
                    1 => Synth.NextGaussian(rng) * Synth.NextGaussian(rng) * 0.5f,         // product (heavy tail)
                    2 => (rng.NextDouble() < 0.02 ? 8f : 0.2f) * Synth.NextGaussian(rng),  // sparse + outliers
                    _ => 0f
                };
            }
            double ErrOf(bool legacy)
            {
                var q = new byte[Dtypes.TensorBytes(n, QuantFormat.TRI)];
                if (legacy) BlockQuantizer.QuantizeTriLegacy(w, q);
                else BlockQuantizer.QuantizeTri(w, q);
                var y = new float[n];
                BlockDequantizer.Dequantize(QuantFormat.TRI, q, y);
                double se = 0, sr = 0;
                for (int i = 0; i < n; i++) { double d = y[i] - w[i]; se += d * d; sr += (double)w[i] * w[i]; }
                return Math.Sqrt(se / Math.Max(sr, 1e-30));
            }
            double eNew = ErrOf(false), eOld = ErrOf(true);
            if (eNew > eOld + 1e-9) throw new Exception($"case {t}: upgraded TRI {eNew:F4} > legacy {eOld:F4}");
            totNew += eNew; totOld += eOld;
        }
        if (totNew >= totOld) throw new Exception($"no aggregate improvement: {totNew:F4} vs {totOld:F4}");
    }

    /// <summary>Every extreme format must run through the real engine: synthesize dense models
    /// in B1 / B1X / T1E / T1E_LO, then decode-generate and require finite, position-dependent logits.</summary>
    public static unsafe void ExtremeFormatsEngine()
    {
        foreach (var f in new[] { QuantFormat.B1, QuantFormat.B1X, QuantFormat.T1E, QuantFormat.T1E_LO })
        {
            string tmp = Path.Combine(Path.GetTempPath(), $"qr_ext_eng_{Guid.NewGuid():N}.qw");
            try
            {
                Synth.Write(tmp, 256, 2, 4, 512, 256, 0, 0, seed: 9, fmt: f);
                using var store = ModelStore.Open(tmp);
                using var eng = new StreamingEngine(store, new EngineOptions { PrefetchBudgetBytes = 16 << 20 });
                var logits0 = eng.Forward(3, 0)[..32].ToArray();
                CheckFinite(logits0);
                var logits1 = eng.Forward(9, 1)[..32].ToArray();
                CheckFinite(logits1);
                float d = 0;
                for (int i = 0; i < logits0.Length; i++) d += Math.Abs(logits0[i] - logits1[i]);
                if (d == 0) throw new Exception($"{f}: logits identical across positions (suspicious)");
                var gen = eng.Generate(new[] { 3, 9 }, 4, new Sampler { Temperature = 0f });
                if (gen.Length == 0) throw new Exception($"{f}: no tokens generated");
            }
            finally { if (File.Exists(tmp)) File.Delete(tmp); }
        }
    }

    /// <summary>The analyze scorer must rank formats sensibly (more bits -> less error on
    /// gaussian data) and the subsampler must stay block-aligned.</summary>
    public static void QuantAnalyzerSmoke()
    {
        var rng = new Random(555);
        const int rows = 96, cols = 256;
        var w = new float[rows * cols];
        for (int i = 0; i < w.Length; i++) w[i] = Synth.NextGaussian(rng) * 0.5f;
        var formats = new[] { QuantFormat.Q8_0, QuantFormat.Q4_K, QuantFormat.TRI, QuantFormat.B1X, QuantFormat.B1 };
        var res = QuantAnalyzer.AnalyzeTensor("smoke", new long[] { rows, cols }, w, formats, maxElems: 1 << 20);
        if (res.Count != formats.Length) throw new Exception($"analyzed {res.Count} formats, expected {formats.Length}");
        double RelL2(QuantFormat f) => res.First(r => r.Format == f).RelL2;
        if (!(RelL2(QuantFormat.Q8_0) < RelL2(QuantFormat.Q4_K))) throw new Exception("q8_0 should beat q4_k");
        if (!(RelL2(QuantFormat.Q4_K) < RelL2(QuantFormat.TRI))) throw new Exception("q4_k should beat TRI");
        if (!(RelL2(QuantFormat.TRI) < RelL2(QuantFormat.B1))) throw new Exception("TRI should beat B1");
        if (!(RelL2(QuantFormat.B1X) < RelL2(QuantFormat.B1))) throw new Exception("B1X should beat B1");

        // subsample: rows spread + block alignment (dims must describe the actual buffer)
        var sample = QuantAnalyzer.Subsample(w, new long[] { rows, cols }, 1 << 13, out long measured);
        if (measured % 256 != 0 || measured == 0) throw new Exception($"subsample not block-aligned: {measured}");
        if (sample.Length != measured) throw new Exception("subsample length mismatch");
    }

    public static unsafe void DotMatches()
    {
        var rng = new Random(11);
        // fp formats included: F16 must go through the REAL half decode (regression guard —
        // the bits<<16 bf16 shortcut once corrupted every f16 dot)
        foreach (var f in new[] { QuantFormat.F16, QuantFormat.BF16, QuantFormat.F32,
                                  QuantFormat.Q4_0, QuantFormat.Q8_0, QuantFormat.Q4_K, QuantFormat.Q6_K, QuantFormat.TRI,
                                  QuantFormat.B1, QuantFormat.B1X, QuantFormat.T1E, QuantFormat.T1E_LO })
        {
            var (bs, bb) = Dtypes.BlockSpec(f);
            int cols = bs * 4, rows = 8;
            var w = new float[rows * cols];
            for (int i = 0; i < w.Length; i++) w[i] = Synth.NextGaussian(rng) * 0.4f;
            if (f == QuantFormat.F16)
                w[^1] = 6e-5f;   // subnormal half — exercises the subnormal decode lane
            var x = new float[cols];
            for (int i = 0; i < cols; i++) x[i] = Synth.NextGaussian(rng) * 0.5f;
            var wq = new byte[Dtypes.TensorBytes((long)rows * cols, f)];
            BlockQuantizer.Quantize(f, w, wq);

            // reference: dequantize then fp32 matvec
            var wdq = new float[rows * cols];
            BlockDequantizer.Dequantize(f, wq, wdq);

            fixed (byte* wp = wq)
            fixed (float* xp = x, rp = wdq)
            fixed (float* yp = new float[rows])
            {
                var yK = new float[rows];
                var yR = new float[rows];
                DotKernels.MatVec(f, wp, rows, cols, xp, yp);
                for (int r = 0; r < rows; r++)
                {
                    float s = 0;
                    for (int c = 0; c < cols; c++) s += rp[r * cols + c] * xp[c];
                    yR[r] = s;
                }
                for (int r = 0; r < rows; r++)
                {
                    float scale = Math.Max(Math.Abs(yR[r]), 1e-3f);
                    if (Math.Abs(yp[r] - yR[r]) / scale > 2e-4f)
                        throw new Exception($"{f} row {r}: kernel {yp[r]} vs ref {yR[r]}");
                }
                GC.KeepAlive(yK);
            }
        }
    }

    public static unsafe void QwRoundTrip()
    {
        string tmp = Path.Combine(Path.GetTempPath(), $"qr_test_{Guid.NewGuid():N}.qw");
        var cfgEl = System.Text.Json.JsonDocument.Parse("""{"arch":"llama","dim":256,"layers":1,"heads":4,"vocab":256,"ffn":256}""").RootElement;
        var data = new byte[Dtypes.TensorBytes(256 * 256, QuantFormat.Q4_K)];
        new Random(3).NextBytes(data);
        var f32 = new float[64];
        for (int i = 0; i < 64; i++) f32[i] = i;

        using (var b = new ModelStoreBuilder(tmp, cfgEl))
        {
            b.Plan("tok_emb", new long[] { 256, 256 }, QuantFormat.Q4_K);
            b.Plan("final_norm", new long[] { 64 }, QuantFormat.F32);
            b.Commit((p, s) =>
            {
                if (p.Name == "tok_emb") s.Write(data);
                else { var bb = new byte[64 * 4]; Buffer.BlockCopy(f32, 0, bb, 0, bb.Length); s.Write(bb); }
            });
        }
        using var store = ModelStore.Open(tmp);
        var t = store.Get("tok_emb");
        if (t.SizeBytes != data.Length) throw new Exception("size mismatch");
        var back = new byte[data.Length];
        new ReadOnlySpan<byte>(t.Ptr, data.Length).CopyTo(back);
        for (int i = 0; i < data.Length; i++) if (back[i] != data[i]) throw new Exception("byte mismatch");
        var nrm = store.Get("final_norm");
        var normBytes = new ReadOnlySpan<byte>(nrm.Ptr, (int)nrm.SizeBytes).ToArray();
        for (int i = 0; i < 64; i++)
            if (BitConverter.ToSingle(normBytes, i * 4) != f32[i]) throw new Exception("f32 mismatch");
        File.Delete(tmp);
    }

    /// <summary>
    /// Hand-writes a tiny single-file .safetensors checkpoint (1-layer Llama, GQA 4q/2kv),
    /// quantizes it via the single-file path (no config.json anywhere — architecture must be
    /// inferred from tensor names/shapes), then verifies the resulting .qw model.
    /// </summary>
    public static void SafeTensorsFileConvert()
    {
        string stPath = Path.Combine(Path.GetTempPath(), $"qr_st_{Guid.NewGuid():N}.safetensors");
        string qwPath = Path.Combine(Path.GetTempPath(), $"qr_st_{Guid.NewGuid():N}.qw");
        try
        {
            const int dim = 256, ffn = 512, vocab = 128, kvDim = 128;   // heads=4 headDim=64 kvHeads=2
            var rng = new Random(21);
            var tensors = new List<(string Name, string Dtype, long[] Shape, byte[] Data)>();

            tensors.Add(("model.embed_tokens.weight", "F16", new long[] { vocab, dim }, F16Tensor(rng, vocab, dim, 0.05f)));
            tensors.Add(("model.norm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)));
            tensors.Add(("model.layers.0.input_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)));
            tensors.Add(("model.layers.0.post_attention_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)));
            tensors.Add(("model.layers.0.self_attn.q_proj.weight", "F16", new long[] { dim, dim }, F16Tensor(rng, dim, dim, 0.05f)));
            tensors.Add(("model.layers.0.self_attn.k_proj.weight", "F16", new long[] { kvDim, dim }, F16Tensor(rng, kvDim, dim, 0.05f)));
            tensors.Add(("model.layers.0.self_attn.v_proj.weight", "F16", new long[] { kvDim, dim }, F16Tensor(rng, kvDim, dim, 0.05f)));
            tensors.Add(("model.layers.0.self_attn.o_proj.weight", "F16", new long[] { dim, dim }, F16Tensor(rng, dim, dim, 0.05f)));
            tensors.Add(("model.layers.0.mlp.gate_proj.weight", "F16", new long[] { ffn, dim }, F16Tensor(rng, ffn, dim, 0.05f)));
            tensors.Add(("model.layers.0.mlp.up_proj.weight", "F16", new long[] { ffn, dim }, F16Tensor(rng, ffn, dim, 0.05f)));
            tensors.Add(("model.layers.0.mlp.down_proj.weight", "F16", new long[] { dim, ffn }, F16Tensor(rng, dim, ffn, 0.05f)));
            File.WriteAllBytes(stPath, BuildSafeTensors(tensors));

            var opt = new ConvertOptions();   // defaults: attn/ffn q4_k, emb/head f16, norm f32
            var rep = ModelConverter.ConvertSafeTensors(new[] { stPath }, null, qwPath, opt);

            var cfg = rep.Config;
            if (cfg.NumLayers != 1) throw new Exception($"layers {cfg.NumLayers} != 1");
            if (cfg.HiddenSize != dim) throw new Exception($"dim {cfg.HiddenSize} != {dim}");
            if (cfg.NumHeads != 4) throw new Exception($"heads {cfg.NumHeads} != 4");
            if (cfg.HeadDim != 64) throw new Exception($"headDim {cfg.HeadDim} != 64");
            if (cfg.EffectiveKvHeads != 2) throw new Exception($"kvHeads {cfg.EffectiveKvHeads} != 2 (GQA inference)");
            if (cfg.FfnDim != ffn) throw new Exception($"ffn {cfg.FfnDim} != {ffn}");
            if (cfg.VocabSize != vocab) throw new Exception($"vocab {cfg.VocabSize} != {vocab}");
            if (rep.TensorCount != tensors.Count) throw new Exception($"tensor count {rep.TensorCount} != {tensors.Count}");

            using var store = ModelStore.Open(qwPath);
            var emb = store.Get("tok_emb");
            if (emb.Format != QuantFormat.F16 || emb.Dims[0] != vocab || emb.Dims[1] != dim)
                throw new Exception($"tok_emb {emb.Format} {string.Join('x', emb.Dims)}");
            if (store.Get("final_norm").Format != QuantFormat.F32) throw new Exception("final_norm not f32");
            if (store.Get("l0.wk").Dims[0] != kvDim) throw new Exception("wk rows wrong");
            if (store.Get("l0.w_gate").Format != QuantFormat.Q4_K) throw new Exception("w_gate not q4_k");
            if (store.Has("lm_head")) throw new Exception("unexpected lm_head (source has none — tied-emb fallback expected)");
        }
        finally
        {
            if (File.Exists(stPath)) File.Delete(stPath);
            if (File.Exists(qwPath)) File.Delete(qwPath);
        }
    }

    /// <summary>
    /// The extreme sub-TRI formats must work through the exact `quantize` path used for real
    /// checkpoints (safetensors -> ConvertSafeTensors -> SafeQuantForRows): accepted by name,
    /// stored exactly where the block divides the row length, and runnable end to end.
    /// </summary>
    public static unsafe void QuantizeSubTriFormats()
    {
        string stPath = Path.Combine(Path.GetTempPath(), $"qr_subtri_{Guid.NewGuid():N}.safetensors");
        string qwPath = Path.Combine(Path.GetTempPath(), $"qr_subtri_{Guid.NewGuid():N}.qw");
        try
        {
            const int dim = 512, ffn = 768, vocab = 128, kvDim = 128;
            // every col count (512 / 768) is a multiple of the 256-wide sub-TRI block -> no fallback
            var rng = new Random(77);
            var tensors = new List<(string Name, string Dtype, long[] Shape, byte[] Data)>
            {
                ("model.embed_tokens.weight", "F16", new long[] { vocab, dim }, F16Tensor(rng, vocab, dim, 0.05f)),
                ("model.norm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                ("model.layers.0.input_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                ("model.layers.0.post_attention_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                ("model.layers.0.self_attn.q_proj.weight", "F16", new long[] { dim, dim }, F16Tensor(rng, dim, dim, 0.05f)),
                ("model.layers.0.self_attn.k_proj.weight", "F16", new long[] { kvDim, dim }, F16Tensor(rng, kvDim, dim, 0.05f)),
                ("model.layers.0.self_attn.v_proj.weight", "F16", new long[] { kvDim, dim }, F16Tensor(rng, kvDim, dim, 0.05f)),
                ("model.layers.0.self_attn.o_proj.weight", "F16", new long[] { dim, dim }, F16Tensor(rng, dim, dim, 0.05f)),
                ("model.layers.0.mlp.gate_proj.weight", "F16", new long[] { ffn, dim }, F16Tensor(rng, ffn, dim, 0.05f)),
                ("model.layers.0.mlp.up_proj.weight", "F16", new long[] { ffn, dim }, F16Tensor(rng, ffn, dim, 0.05f)),
                ("model.layers.0.mlp.down_proj.weight", "F16", new long[] { dim, ffn }, F16Tensor(rng, dim, ffn, 0.05f)),
            };
            File.WriteAllBytes(stPath, BuildSafeTensors(tensors));

            foreach (var f in new[] { QuantFormat.TRI, QuantFormat.B1, QuantFormat.B1X, QuantFormat.T1E, QuantFormat.T1E_LO })
            {
                if (File.Exists(qwPath)) File.Delete(qwPath);
                var opt = new ConvertOptions { Attention = f, Ffn = f };
                var rep = ModelConverter.ConvertSafeTensors(new[] { stPath }, null, qwPath, opt);
                if (rep.TensorCount != tensors.Count) throw new Exception($"{f}: tensor count {rep.TensorCount}");
                using var store = ModelStore.Open(qwPath);
                foreach (var name in new[] { "l0.wq", "l0.wk", "l0.wo", "l0.w_gate", "l0.w_down" })
                    if (store.Get(name).Format != f)
                        throw new Exception($"{f}: {name} stored as {store.Get(name).Format}");
                if (store.Get("tok_emb").Format != QuantFormat.F16) throw new Exception($"{f}: emb must stay f16");

                using var eng = new StreamingEngine(store, new EngineOptions { PrefetchBudgetBytes = 16 << 20 });
                CheckFinite(eng.Forward(3, 0)[..32].ToArray());
            }
        }
        finally
        {
            if (File.Exists(stPath)) File.Delete(stPath);
            if (File.Exists(qwPath)) File.Delete(qwPath);
        }
    }

    /// <summary>
    /// requant: downgraded tensors re-encode in the target format, untouched tensors are
    /// byte-exact passthrough, norms/emb/head stay protected, --dry-run writes nothing,
    /// and the requantized model still runs on the engine.
    /// </summary>
    public static unsafe void RequantDowngradeAndPassthrough()
    {
        string src = Path.Combine(Path.GetTempPath(), $"qr_req_{Guid.NewGuid():N}.qw");
        string dst = Path.Combine(Path.GetTempPath(), $"qr_req_{Guid.NewGuid():N}.qw");
        string dry = Path.Combine(Path.GetTempPath(), $"qr_req_{Guid.NewGuid():N}.qw");
        try
        {
            Synth.Write(src, 256, 2, 4, 512, 256, 0, 0, seed: 9, fmt: QuantFormat.Q8_0);

            var resDry = Requantizer.Run(src, dry, new RequantOptions { Attention = QuantFormat.TRI, Ffn = QuantFormat.TRI, DryRun = true });
            if (File.Exists(dry)) throw new Exception("dry run wrote a file");
            if (resDry.Requantized == 0) throw new Exception("dry-run plan requantizes nothing");

            var res = Requantizer.Run(src, dst, new RequantOptions { Attention = QuantFormat.TRI, Ffn = QuantFormat.TRI });
            if (res.Requantized != resDry.Requantized) throw new Exception("dry-run plan diverged from real run");
            if (res.OutputBytes >= res.SourceBytes) throw new Exception($"no shrink: {res.SourceBytes} -> {res.OutputBytes}");

            using (var s = ModelStore.Open(src))
            using (var d = ModelStore.Open(dst))
            {
                foreach (var name in new[] { "l0.wq", "l1.wo", "l0.w_gate", "l1.w_down" })
                    if (d.Get(name).Format != QuantFormat.TRI) throw new Exception($"{name}: {d.Get(name).Format} != TRI");
                foreach (var name in new[] { "tok_emb", "lm_head" })
                    if (d.Get(name).Format != QuantFormat.F16) throw new Exception($"{name}: {d.Get(name).Format} != F16");
                if (d.Get("final_norm").Format != QuantFormat.F32) throw new Exception("final_norm changed");

                // passthrough byte-exactness for untouched tensors
                if (!s.Get("tok_emb").Bytes.SequenceEqual(d.Get("tok_emb").Bytes))
                    throw new Exception("tok_emb passthrough bytes differ");
                if (!s.Get("final_norm").Bytes.SequenceEqual(d.Get("final_norm").Bytes))
                    throw new Exception("final_norm passthrough bytes differ");

                using var eng = new StreamingEngine(d, new EngineOptions { PrefetchBudgetBytes = 16 << 20 });
                CheckFinite(eng.Forward(3, 0)[..32].ToArray());
                var gen = eng.Generate(new[] { 3, 9 }, 4, new Sampler { Temperature = 0f });
                if (gen.Length == 0) throw new Exception("no tokens generated from requantized model");
            }
        }
        finally
        {
            if (File.Exists(src)) File.Delete(src);
            if (File.Exists(dst)) File.Delete(dst);
        }
    }

    /// <summary>
    /// requant --from-report: an `analyze` report drives per-tensor format choice — the
    /// smallest bpw whose relL2 fits the quality gate, never growing a tensor, and tensors
    /// missing from the report stay at their source format (byte-exact).
    /// </summary>
    public static unsafe void RequantSmartFromReport()
    {
        string src = Path.Combine(Path.GetTempPath(), $"qr_reqs_{Guid.NewGuid():N}.qw");
        string dst = Path.Combine(Path.GetTempPath(), $"qr_reqs_{Guid.NewGuid():N}.qw");
        string report = Path.Combine(Path.GetTempPath(), $"qr_reqs_{Guid.NewGuid():N}.json");
        try
        {
            Synth.Write(src, 256, 2, 4, 512, 256, 0, 0, seed: 13, fmt: QuantFormat.Q4_K);
            // fabricated analyze report: wq affords TRI (0.30) but not B1 (0.66) / T1E_LO (0.60);
            // w_gate affords B1 (0.30); w_up affords nothing under the gate -> kept.
            File.WriteAllText(report, """
            {
              "tensors": [
                { "name": "l0.wq", "formats": [
                    { "format": "B1", "bpw": 1.0625, "rel_l2": 0.66 },
                    { "format": "T1E_LO", "bpw": 1.0, "rel_l2": 0.60 },
                    { "format": "TRI", "bpw": 2.25, "rel_l2": 0.30 },
                    { "format": "Q4_K", "bpw": 4.5, "rel_l2": 0.10 } ] },
                { "name": "l0.w_gate", "formats": [
                    { "format": "B1", "bpw": 1.0625, "rel_l2": 0.30 },
                    { "format": "TRI", "bpw": 2.25, "rel_l2": 0.20 } ] },
                { "name": "l0.w_up", "formats": [
                    { "format": "B1", "bpw": 1.0625, "rel_l2": 0.90 },
                    { "format": "TRI", "bpw": 2.25, "rel_l2": 0.55 } ] }
              ]
            }
            """);

            var res = Requantizer.Run(src, dst, new RequantOptions { FromReport = report, MaxRelL2 = 0.35 });
            if (res.Requantized != 2) throw new Exception($"smart plan requantized {res.Requantized} tensors, expected 2");
            using (var d = ModelStore.Open(dst))
            {
                if (d.Get("l0.wq").Format != QuantFormat.TRI)
                    throw new Exception($"l0.wq -> {d.Get("l0.wq").Format}, expected TRI (B1/T1E_LO above the gate)");
                if (d.Get("l0.w_gate").Format != QuantFormat.B1)
                    throw new Exception($"l0.w_gate -> {d.Get("l0.w_gate").Format}, expected B1");
                if (d.Get("l0.w_up").Format != QuantFormat.Q4_K)
                    throw new Exception($"l0.w_up -> {d.Get("l0.w_up").Format}, expected keep q4_k");
                if (d.Get("l1.wq").Format != QuantFormat.Q4_K)
                    throw new Exception($"l1.wq (not in report) -> {d.Get("l1.wq").Format}, expected keep q4_k");

                using var eng = new StreamingEngine(d, new EngineOptions { PrefetchBudgetBytes = 16 << 20 });
                CheckFinite(eng.Forward(3, 0)[..32].ToArray());
            }
        }
        finally
        {
            if (File.Exists(src)) File.Delete(src);
            if (File.Exists(dst)) File.Delete(dst);
            if (File.Exists(report)) File.Delete(report);
        }
    }

    /// <summary>
    /// Assembles a 2-layer model from two .safetensors shards using the --append path:
    /// quantize shard 1 (with --layers override, no config.json), append shard 2, re-append
    /// (idempotent no-op), then verify the result matches a single-shot conversion byte-for-byte.
    /// </summary>
    public static unsafe void SafeTensorsShardAppend()
    {
        string st1 = Path.Combine(Path.GetTempPath(), $"qr_sh1_{Guid.NewGuid():N}.safetensors");
        string st2 = Path.Combine(Path.GetTempPath(), $"qr_sh2_{Guid.NewGuid():N}.safetensors");
        string qwApp = Path.Combine(Path.GetTempPath(), $"qr_sh_{Guid.NewGuid():N}.qw");
        string qwRef = Path.Combine(Path.GetTempPath(), $"qr_sh_{Guid.NewGuid():N}.qw");
        try
        {
            const int dim = 256, ffn = 512, vocab = 128, kvDim = 128;
            var rng = new Random(33);
            var shared = new List<(string Name, string Dtype, long[] Shape, byte[] Data)>
            {
                ("model.embed_tokens.weight", "F16", new long[] { vocab, dim }, F16Tensor(rng, vocab, dim, 0.05f)),
                ("model.norm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
            };
            var layers = new List<(string Name, string Dtype, long[] Shape, byte[] Data)>[2];
            for (int l = 0; l < 2; l++)
            {
                string p = $"model.layers.{l}.";
                layers[l] = new List<(string, string, long[], byte[])>
                {
                    (p + "input_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                    (p + "post_attention_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                    (p + "self_attn.q_proj.weight", "F16", new long[] { dim, dim }, F16Tensor(rng, dim, dim, 0.05f)),
                    (p + "self_attn.k_proj.weight", "F16", new long[] { kvDim, dim }, F16Tensor(rng, kvDim, dim, 0.05f)),
                    (p + "self_attn.v_proj.weight", "F16", new long[] { kvDim, dim }, F16Tensor(rng, kvDim, dim, 0.05f)),
                    (p + "self_attn.o_proj.weight", "F16", new long[] { dim, dim }, F16Tensor(rng, dim, dim, 0.05f)),
                    (p + "mlp.gate_proj.weight", "F16", new long[] { ffn, dim }, F16Tensor(rng, ffn, dim, 0.05f)),
                    (p + "mlp.up_proj.weight", "F16", new long[] { ffn, dim }, F16Tensor(rng, ffn, dim, 0.05f)),
                    (p + "mlp.down_proj.weight", "F16", new long[] { dim, ffn }, F16Tensor(rng, dim, ffn, 0.05f)),
                };
            }
            File.WriteAllBytes(st1, BuildSafeTensors(shared.Concat(layers[0])));   // shard 1: embed + norm + layer 0
            File.WriteAllBytes(st2, BuildSafeTensors(layers[1]));                  // shard 2: layer 1 only (no embed!)

            // 1) fresh create from shard 1 — layer count would wrongly infer as 1, pin with override
            var opt1 = new ConvertOptions { HeaderReserveBytes = 1 << 20, OverrideLayers = 2 };
            var rep1 = ModelConverter.ConvertSafeTensors(new[] { st1 }, null, qwApp, opt1);
            if (rep1.TensorCount != 11) throw new Exception($"shard 1 added {rep1.TensorCount}, expected 11");
            if (rep1.Config.NumLayers != 2) throw new Exception($"layers {rep1.Config.NumLayers} != 2 after override");
            if (rep1.Config.VocabSize != vocab || rep1.Config.HiddenSize != dim)
                throw new Exception("shard-1 inference of vocab/dim wrong");

            // 2) append shard 2 — inference must not be needed (config comes from the .qw)
            var rep2 = ModelConverter.ConvertSafeTensors(new[] { st2 }, null, qwApp,
                new ConvertOptions(), appendToExisting: true);
            if (rep2.TensorCount != 9) throw new Exception($"shard 2 added {rep2.TensorCount}, expected 9");

            // 3) re-append shard 2: idempotent, nothing added
            var rep3 = ModelConverter.ConvertSafeTensors(new[] { st2 }, null, qwApp,
                new ConvertOptions(), appendToExisting: true);
            if (rep3.TensorCount != 0) throw new Exception($"re-append added {rep3.TensorCount}, expected 0");

            // 4) reference: single-shot conversion of both shards together (inference: layers=2)
            var repRef = ModelConverter.ConvertSafeTensors(new[] { st1, st2 }, null, qwRef, new ConvertOptions());
            if (repRef.TensorCount != 20) throw new Exception($"reference has {repRef.TensorCount}, expected 20");

            using var app = ModelStore.Open(qwApp);
            using var refr = ModelStore.Open(qwRef);
            if (app.Tensors.Count != refr.Tensors.Count)
                throw new Exception($"tensor count {app.Tensors.Count} != reference {refr.Tensors.Count}");
            foreach (var (name, tv) in app.Tensors)
            {
                var rv = refr.Get(name);
                if (!tv.Dims.SequenceEqual(rv.Dims) || tv.Format != rv.Format || tv.SizeBytes != rv.SizeBytes)
                    throw new Exception($"plan mismatch for {name}");
                var a = new ReadOnlySpan<byte>(tv.Ptr, (int)tv.SizeBytes).ToArray();
                var b = new ReadOnlySpan<byte>(rv.Ptr, (int)rv.SizeBytes).ToArray();
                for (int i = 0; i < a.Length; i++)
                    if (a[i] != b[i]) throw new Exception($"byte mismatch in {name} at {i}");
            }

            // 5) completeness: assembled model must be runnable
            var cfgApp = ModelConfig.FromJson(app.ConfigJson);
            var (present, missing) = Commands.Completeness(app, cfgApp);
            if (missing.Count != 0 || present != 2)
                throw new Exception($"completeness: {present}/2 layers, {missing.Count} missing");

            // 6) appending to a file without reserved header must fail clearly
            string noReserve = Path.Combine(Path.GetTempPath(), $"qr_nores_{Guid.NewGuid():N}.qw");
            try
            {
                var cfgEl = System.Text.Json.JsonDocument.Parse("""{"arch":"llama","dim":256,"layers":1,"heads":4,"vocab":128,"ffn":512}""").RootElement;
                using (var b = new ModelStoreBuilder(noReserve, cfgEl))
                {
                    b.Plan("tok_emb", new long[] { vocab, dim }, QuantFormat.F16);
                    b.Commit((p, s) => s.Write(F16Tensor(rng, vocab, dim, 0.05f)));
                }
                bool threw = false;
                try { using var _ = new ModelStoreAppender(noReserve); }
                catch (InvalidDataException) { threw = true; }
                if (!threw) throw new Exception("appender accepted a file without reserved header");
            }
            finally { if (File.Exists(noReserve)) File.Delete(noReserve); }
        }
        finally
        {
            foreach (var f in new[] { st1, st2, qwApp, qwRef })
                if (File.Exists(f)) File.Delete(f);
        }
    }

    /// <summary>
    /// Fused projections: a checkpoint with contiguous fused QKV (qkv_proj) + fused gate_up
    /// (gate_up_proj) must be split into wq/wk/wv/w_gate/w_up with byte-exact rows, and the
    /// resulting model must run. A second pass checks the GLM/BLOOM interleaved QKV layout.
    /// </summary>
    public static unsafe void FusedProjectionSplits()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"qr_fused_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        string stPath = Path.Combine(dir, "model.safetensors");
        string cfgPath = Path.Combine(dir, "config.json");
        string qwRun = Path.Combine(dir, "run.qw");
        string qwIlv = Path.Combine(dir, "ilv.qw");
        try
        {
            const int dim = 256, heads = 4, hd = 64, kv = 2, ffn = 512, vocab = 128;
            var rng = new Random(77);

            float[] Rnd(long rows, long cols, float s)
            {
                var f = new float[rows * cols];
                for (long i = 0; i < f.Length; i++) f[i] = Synth.NextGaussian(rng) * s;
                return f;
            }

            // ---- pass A: contiguous fused QKV + fused gate_up, engine smoke ----
            var wq = Rnd(heads * hd, dim, 0.05f);
            var wk = Rnd(kv * hd, dim, 0.05f);
            var wv = Rnd(kv * hd, dim, 0.05f);
            var wg = Rnd(ffn, dim, 0.05f);
            var wu = Rnd(ffn, dim, 0.05f);
            var fusedQkv = new float[(heads * hd + 2 * kv * hd) * dim];
            Array.Copy(wq, 0, fusedQkv, 0, wq.Length);
            Array.Copy(wk, 0, fusedQkv, wq.Length, wk.Length);
            Array.Copy(wv, 0, fusedQkv, wq.Length + wk.Length, wv.Length);
            var fusedGu = new float[2 * ffn * dim];
            Array.Copy(wg, 0, fusedGu, 0, wg.Length);
            Array.Copy(wu, 0, fusedGu, wg.Length, wu.Length);

            var tensorsA = new List<(string, string, long[], byte[])>
            {
                ("model.embed_tokens.weight", "F16", new long[] { vocab, dim }, F16Tensor(rng, vocab, dim, 0.05f)),
                ("model.norm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                ("model.layers.0.input_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                ("model.layers.0.post_attention_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                ("model.layers.0.self_attn.qkv_proj.weight", "F32", new long[] { fusedQkv.Length / dim, dim }, ToBytes(fusedQkv)),
                ("model.layers.0.self_attn.o_proj.weight", "F16", new long[] { dim, dim }, F16Tensor(rng, dim, dim, 0.05f)),
                ("model.layers.0.mlp.gate_up_proj.weight", "F32", new long[] { 2 * ffn, dim }, ToBytes(fusedGu)),
                ("model.layers.0.mlp.down_proj.weight", "F16", new long[] { dim, ffn }, F16Tensor(rng, dim, ffn, 0.05f)),
            };
            File.WriteAllBytes(stPath, BuildSafeTensors(tensorsA));
            File.WriteAllText(cfgPath, """
                {"architectures":["LlamaForCausalLM"],"hidden_size":256,"num_hidden_layers":1,
                 "num_attention_heads":4,"num_key_value_heads":2,"head_dim":64,
                 "intermediate_size":512,"vocab_size":128}
                """);

            var optA = new ConvertOptions { MinQuantElements = long.MaxValue };   // force F32 -> exact compare
            var repA = ModelConverter.ConvertSafeTensors(new[] { stPath }, cfgPath, qwRun, optA);
            if (repA.TensorCount != 11)
            {
                using var dbg = ModelStore.Open(qwRun);
                throw new Exception($"split pass: {repA.TensorCount} tensors ({string.Join(", ", dbg.Tensors.Keys)}), expected 11");
            }

            using (var store = ModelStore.Open(qwRun))
            {
                Exact(store.Get("l0.wq"), wq);
                Exact(store.Get("l0.wk"), wk);
                Exact(store.Get("l0.wv"), wv);
                Exact(store.Get("l0.w_gate"), wg);
                Exact(store.Get("l0.w_up"), wu);
                if (store.Get("l0.wk").Dims[0] != kv * hd) throw new Exception("wk rows wrong");

                using var eng = new StreamingEngine(store, new EngineOptions { PrefetchBudgetBytes = 16 << 20 });
                CheckFinite(eng.Forward(3, 0)[..32].ToArray());
                CheckFinite(eng.Forward(9, 1)[..32].ToArray());
            }

            // ---- pass B: GLM/BLOOM interleaved query_key_value ----
            var wq2 = Rnd(heads * hd, dim, 0.05f);
            var wk2 = Rnd(heads * hd, dim, 0.05f);
            var wv2 = Rnd(heads * hd, dim, 0.05f);
            var ilv = new float[heads * 3 * hd * dim];
            for (int h = 0; h < heads; h++)
                for (int t = 0; t < hd; t++)
                {
                    long src = (h * hd + t) * dim, dst = ((long)h * 3 * hd + t) * dim;
                    Array.Copy(wq2, src, ilv, dst, dim);                          // q block (hd rows)
                    Array.Copy(wk2, src, ilv, dst + (long)hd * dim, dim);         // k block
                    Array.Copy(wv2, src, ilv, dst + 2L * hd * dim, dim);          // v block
                }
            var tensorsB = new List<(string, string, long[], byte[])>
            {
                ("model.embed_tokens.weight", "F16", new long[] { vocab, dim }, F16Tensor(rng, vocab, dim, 0.05f)),
                ("model.norm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                ("model.layers.0.input_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                ("model.layers.0.post_attention_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                ("model.layers.0.self_attention.query_key_value.weight", "F32", new long[] { heads * 3 * hd, dim }, ToBytes(ilv)),
                ("model.layers.0.self_attn.o_proj.weight", "F16", new long[] { dim, dim }, F16Tensor(rng, dim, dim, 0.05f)),
                ("model.layers.0.mlp.gate_proj.weight", "F16", new long[] { ffn, dim }, F16Tensor(rng, ffn, dim, 0.05f)),
                ("model.layers.0.mlp.up_proj.weight", "F16", new long[] { ffn, dim }, F16Tensor(rng, ffn, dim, 0.05f)),
                ("model.layers.0.mlp.down_proj.weight", "F16", new long[] { dim, ffn }, F16Tensor(rng, dim, ffn, 0.05f)),
            };
            File.WriteAllBytes(stPath, BuildSafeTensors(tensorsB));
            string cfgB = Path.Combine(dir, "config_b.json");
            File.WriteAllText(cfgB, """
                {"architectures":["ChatGLMModel"],"hidden_size":256,"num_hidden_layers":1,
                 "num_attention_heads":4,"num_key_value_heads":4,"head_dim":64,
                 "intermediate_size":512,"vocab_size":128}
                """);   // interleaved layout is MHA: kv-heads == heads
            var repB = ModelConverter.ConvertSafeTensors(new[] { stPath }, cfgB, qwIlv,
                new ConvertOptions { MinQuantElements = long.MaxValue });
            using (var store = ModelStore.Open(qwIlv))
            {
                Exact(store.Get("l0.wq"), wq2);
                Exact(store.Get("l0.wk"), wk2);
                Exact(store.Get("l0.wv"), wv2);
            }
            if (repB.TensorCount != 11) throw new Exception($"interleaved pass: {repB.TensorCount} tensors, expected 11");
        }
        finally
        {
            if (File.Exists(stPath)) File.Delete(stPath);
            if (File.Exists(cfgPath)) File.Delete(cfgPath);
            if (File.Exists(qwRun)) File.Delete(qwRun);
            if (File.Exists(qwIlv)) File.Delete(qwIlv);
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }

    private static byte[] ToBytes(float[] f)
    {
        var b = new byte[f.Length * 4];
        Buffer.BlockCopy(f, 0, b, 0, b.Length);
        return b;
    }

    private static unsafe void Exact(TensorView t, float[] expected)
    {
        if (t.Format != QuantFormat.F32) throw new Exception($"{t.Name}: expected F32 for exact compare, got {t.Format}");
        var raw = new byte[t.SizeBytes];
        new ReadOnlySpan<byte>(t.Ptr, (int)t.SizeBytes).CopyTo(raw);
        for (int i = 0; i < expected.Length; i++)
            if (BitConverter.ToSingle(raw, i * 4) != expected[i])
            {
                string dims = string.Join('x', t.Dims);
                throw new Exception(
                    $"{t.Name} [{dims}]: row data mismatch at element {i}: " +
                    $"got {BitConverter.ToSingle(raw, i * 4):g6}, expected {expected[i]:g6} " +
                    $"(first got [{BitConverter.ToSingle(raw, 0):g6}, {BitConverter.ToSingle(raw, 4):g6}], " +
                    $"first expected [{expected[0]:g6}, {expected[1]:g6}]) — split wrote wrong rows");
            }
        long n = 1; foreach (var d in t.Dims) n *= d;
        if (n != expected.Length) throw new Exception($"{t.Name}: element count {n} != {expected.Length}");
    }

    /// <summary>
    /// GLM-5.x-style checkpoints: nested text_config parses into the full GLM-5 ModelConfig
    /// (attention kinds, KDA/MLA dims, mHC, GLM MoE), shared indexers are refused, and the
    /// name mapper routes KDA/MLA/mHC tensors to the new canonical names using kind context.
    /// </summary>
    public static void MlaRefusalAndNestedConfig()
    {
        // GLM-5.3-Flash-like: everything under text_config, MLA + KDA layer types
        const string glm5 = """
            {"architectures":["Glm5NextForConditionalGeneration"],
             "text_config":{"hidden_size":4096,"num_hidden_layers":4,"num_attention_heads":64,
               "num_key_value_heads":64,"head_dim":0,"vocab_size":154880,
               "intermediate_size":12288,"moe_intermediate_size":2048,
               "n_routed_experts":288,"num_experts_per_tok":8,"n_shared_experts":1,
               "routed_scaling_factor":2.5,"scoring_func":"sigmoid","topk_method":"noaux_tc",
               "swiglu_limit":10.0,"first_k_dense_replace":3,
               "q_lora_rank":1536,"kv_lora_rank":512,"qk_nope_head_dim":256,
               "qk_rope_head_dim":0,"v_head_dim":256,
               "mhc":true,"hc_mult":4,"hc_eps":1e-6,"hc_sinkhorn_iters":20,
               "linear_attn_config":{"num_heads":64,"head_dim":128,
                 "short_conv_kernel_size":4,"gate_lower_bound":-5.0},
               "layer_types":["linear_attention","linear_attention","deepseek_sparse_attention","linear_attention"]}}
            """;
        var cfg = ModelConverter.ParseHfConfig(glm5, new ConvertOptions(), "glm5.json", out var unsupported);
        if (cfg.HiddenSize != 4096 || cfg.NumLayers != 4 || cfg.NumHeads != 64 ||
            cfg.VocabSize != 154880 || cfg.NumExperts != 288 || cfg.NumActiveExperts != 8 ||
            cfg.ExpertFfnDim != 2048 || cfg.Arch != "moe")
            throw new Exception($"text_config parse wrong: dim={cfg.HiddenSize} layers={cfg.NumLayers} heads={cfg.NumHeads} " +
                                $"vocab={cfg.VocabSize} experts={cfg.NumExperts}x{cfg.NumActiveExperts} effn={cfg.ExpertFfnDim} arch={cfg.Arch}");
        if (unsupported.Count != 0)
            throw new Exception($"GLM-5 config must not be refused any more: [{string.Join("; ", unsupported)}]");
        if (cfg.AttnKinds.Length != 4 || cfg.AttnKinds[0] != "kda" || cfg.AttnKinds[2] != "mla")
            throw new Exception($"attn kinds wrong: [{string.Join(',', cfg.AttnKinds)}]");
        if (!cfg.HasMhc || cfg.HcMult != 4 || cfg.HcSinkhornIters != 20 || Math.Abs(cfg.HcEps - 1e-6f) > 1e-12f)
            throw new Exception($"mHC parse wrong: mult={cfg.HcMult} iters={cfg.HcSinkhornIters} eps={cfg.HcEps}");
        if (cfg.LinearHeads != 64 || cfg.LinearHeadDim != 128 || cfg.LinearConvKernel != 4 ||
            Math.Abs(cfg.LinearLowerBound - (-5f)) > 1e-6f)
            throw new Exception($"KDA parse wrong: heads={cfg.LinearHeads} dim={cfg.LinearHeadDim} conv={cfg.LinearConvKernel} lb={cfg.LinearLowerBound}");
        if (cfg.QLoraRank != 1536 || cfg.KvLoraRank != 512 || cfg.QkNopeHeadDim != 256 ||
            cfg.QkRopeHeadDim != 0 || cfg.VHeadDim != 256)
            throw new Exception($"MLA parse wrong: ql={cfg.QLoraRank} kvl={cfg.KvLoraRank} nope={cfg.QkNopeHeadDim} rope={cfg.QkRopeHeadDim} v={cfg.VHeadDim}");
        if (!cfg.MoeSigmoid || Math.Abs(cfg.RoutedScale - 2.5f) > 1e-6f || cfg.NumSharedExperts != 1 ||
            Math.Abs(cfg.SwigluLimit - 10f) > 1e-6f)
            throw new Exception($"GLM MoE parse wrong: sigmoid={cfg.MoeSigmoid} scale={cfg.RoutedScale} shared={cfg.NumSharedExperts} limit={cfg.SwigluLimit}");

        // cross-layer shared DSA indexer remains genuinely unsupported
        const string shared = """
            {"architectures":["Glm5NextForConditionalGeneration"],
             "text_config":{"hidden_size":256,"num_hidden_layers":2,"num_attention_heads":4,
               "vocab_size":128,"q_lora_rank":64,"kv_lora_rank":32,"qk_nope_head_dim":32,
               "v_head_dim":32,"indexer_types":["full","shared"],
               "layer_types":["deepseek_sparse_attention","deepseek_sparse_attention"]}}
            """;
        bool threw = false;
        try { ModelConverter.ConfigFromHfJson(shared, new ConvertOptions(), "shared.json"); }
        catch (NotSupportedException ex) { threw = ex.Message.Contains("shared"); }
        if (!threw) throw new Exception("ConfigFromHfJson did not refuse the shared-indexer checkpoint");

        // name mapping uses per-layer kind context: same self_attn.q_proj name means
        // KDA projections on a KDA layer and GQA wq on a normal layer
        var kinds = new NameMapper.LayerKinds();
        kinds.Kda.Add(0);
        kinds.Mla.Add(1);
        string? m1 = NameMapper.Map(NameMapper.Source.HuggingFace, "model.language_model.layers.0.self_attn.q_proj.weight", new long[] { 8192, 4096 }, 4, kinds, out _);
        if (m1 != "l0.kda.q") throw new Exception($"KDA q mapping wrong: {m1}");
        string? m2 = NameMapper.Map(NameMapper.Source.HuggingFace, "model.language_model.layers.0.self_attn.q_conv1d.weight", new long[] { 8192, 1, 4 }, 4, kinds, out _);
        if (m2 != "l0.kda.qconv") throw new Exception($"KDA conv mapping wrong: {m2}");
        string? m3 = NameMapper.Map(NameMapper.Source.HuggingFace, "model.language_model.layers.0.self_attn.A_log", new long[] { 64 }, 4, kinds, out _);
        if (m3 != "l0.kda.alog") throw new Exception($"KDA A_log mapping wrong: {m3}");
        string? m4 = NameMapper.Map(NameMapper.Source.HuggingFace, "model.language_model.layers.0.hc_attn_fn", new long[] { 24, 16384 }, 4, kinds, out _);
        if (m4 != "l0.hc_attn_fn") throw new Exception($"mHC fn mapping wrong: {m4}");
        string? m5 = NameMapper.Map(NameMapper.Source.HuggingFace, "model.language_model.layers.1.self_attn.q_a_proj.weight", new long[] { 1536, 4096 }, 4, kinds, out _);
        if (m5 != "l1.mla.qa") throw new Exception($"MLA qa mapping wrong: {m5}");
        string? m6 = NameMapper.Map(NameMapper.Source.HuggingFace, "model.language_model.layers.1.self_attn.indexer.wq_b.weight", new long[] { 4096, 1536 }, 4, kinds, out _);
        if (m6 != null) throw new Exception($"indexer must be skipped: {m6}");
        string? m7 = NameMapper.Map(NameMapper.Source.HuggingFace, "model.language_model.layers.2.mlp.gate.e_score_correction_bias", new long[] { 288 }, 4, kinds, out _);
        if (m7 != "l2.router_bias") throw new Exception($"router bias mapping wrong: {m7}");
        string? m8 = NameMapper.Map(NameMapper.Source.HuggingFace, "model.language_model.layers.2.mlp.shared_experts.gate_proj.weight", new long[] { 2048, 4096 }, 4, kinds, out _);
        if (m8 != "l2.sw_gate") throw new Exception($"shared expert mapping wrong: {m8}");
        string? m9 = NameMapper.Map(NameMapper.Source.HuggingFace, "model.language_model.layers.2.mlp.gate.weight_scale_inv", new long[] { 2, 32 }, 4, kinds, out _);
        if (m9 != null) throw new Exception($"fp8 scale companion must not be planned: {m9}");
    }

    /// <summary>
    /// Reproduces the field failure: a .qw created earlier with a degenerate config
    /// (dim=0/heads=0/vocab=0 — e.g. from a nested text_config the old build could not read)
    /// must be healed by --append --config, then completed and run.
    /// </summary>
    public static unsafe void AppendHealsDegenerateConfig()
    {
        string stPath = Path.Combine(Path.GetTempPath(), $"qr_heal_{Guid.NewGuid():N}.safetensors");
        string cfgPath = Path.Combine(Path.GetTempPath(), $"qr_heal_{Guid.NewGuid():N}.json");
        string qwPath = Path.Combine(Path.GetTempPath(), $"qr_heal_{Guid.NewGuid():N}.qw");
        try
        {
            const int dim = 256, ffn = 512, vocab = 128, kvDim = 128, layers = 2;
            var rng = new Random(41);

            // the broken .qw: only tok_emb, config with zeros (what the old build produced
            // when config.json keys were nested under text_config)
            var cfgEl = System.Text.Json.JsonDocument.Parse(
                """{"arch":"llama","layers":2,"dim":0,"heads":0,"vocab":0}""").RootElement;
            using (var b = new ModelStoreBuilder(qwPath, cfgEl, 1 << 20))
            {
                b.Plan("tok_emb", new long[] { vocab, dim }, QuantFormat.F16);
                b.Commit((p, s) => s.Write(F16Tensor(rng, vocab, dim, 0.05f)));
            }

            File.WriteAllText(cfgPath, """
                {"hidden_size":256,"num_hidden_layers":2,"num_attention_heads":4,
                 "num_key_value_heads":2,"intermediate_size":512,"vocab_size":128}
                """);

            var shard = new List<(string, string, long[], byte[])>
            {
                ("model.norm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
            };
            for (int l = 0; l < layers; l++)
            {
                string p = $"model.layers.{l}.";
                shard.Add((p + "input_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)));
                shard.Add((p + "post_attention_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)));
                shard.Add((p + "self_attn.q_proj.weight", "F16", new long[] { dim, dim }, F16Tensor(rng, dim, dim, 0.05f)));
                shard.Add((p + "self_attn.k_proj.weight", "F16", new long[] { kvDim, dim }, F16Tensor(rng, kvDim, dim, 0.05f)));
                shard.Add((p + "self_attn.v_proj.weight", "F16", new long[] { kvDim, dim }, F16Tensor(rng, kvDim, dim, 0.05f)));
                shard.Add((p + "self_attn.o_proj.weight", "F16", new long[] { dim, dim }, F16Tensor(rng, dim, dim, 0.05f)));
                shard.Add((p + "mlp.gate_proj.weight", "F16", new long[] { ffn, dim }, F16Tensor(rng, ffn, dim, 0.05f)));
                shard.Add((p + "mlp.up_proj.weight", "F16", new long[] { ffn, dim }, F16Tensor(rng, ffn, dim, 0.05f)));
                shard.Add((p + "mlp.down_proj.weight", "F16", new long[] { dim, ffn }, F16Tensor(rng, dim, ffn, 0.05f)));
            }
            File.WriteAllBytes(stPath, BuildSafeTensors(shard));

            var rep = ModelConverter.ConvertSafeTensors(new[] { stPath }, cfgPath, qwPath,
                new ConvertOptions(), appendToExisting: true);
            var cfg = rep.Config;
            if (cfg.HiddenSize != dim || cfg.NumHeads != 4 || cfg.VocabSize != vocab || cfg.NumLayers != 2)
                throw new Exception($"config not healed: dim={cfg.HiddenSize} heads={cfg.NumHeads} vocab={cfg.VocabSize}");
            if (rep.TensorCount != 19) throw new Exception($"append added {rep.TensorCount}, expected 19");
            if (cfg.FfnDim != ffn) throw new Exception($"ffn not enriched: {cfg.FfnDim}");

            using (var store = ModelStore.Open(qwPath))
            {
                var (present, missing) = Commands.Completeness(store, cfg);
                if (missing.Count != 0 || present != 2)
                    throw new Exception($"completeness after heal+append: {present}/2 layers, {missing.Count} missing");
                using var eng = new StreamingEngine(store, new EngineOptions { PrefetchBudgetBytes = 16 << 20 });
                CheckFinite(eng.Forward(3, 0)[..32].ToArray());
            }
        }
        finally
        {
            if (File.Exists(stPath)) File.Delete(stPath);
            if (File.Exists(cfgPath)) File.Delete(cfgPath);
            if (File.Exists(qwPath)) File.Delete(qwPath);
        }
    }

    /// <summary>fp8_e4m3fn decode must be exact on the format grid, encode round-trips within
    /// one ulp, and the safetensors reader must apply 128x128 block scales correctly.</summary>
    public static void Fp8E4M3Roundtrip()
    {
        // decode table spot checks (fn variant: max 448, no inf, 0x7F = NaN)
        if (Dtypes.F8E4M3ToFloat(0) != 0f) throw new Exception("0x00 must decode to 0");
        if (Dtypes.F8E4M3ToFloat(0x38) != 1f) throw new Exception("0x38 must decode to 1.0");
        if (Dtypes.F8E4M3ToFloat(0xB8) != -1f) throw new Exception("0xB8 must decode to -1.0");
        if (Math.Abs(Dtypes.F8E4M3ToFloat(0x7E) - 448f) > 1e-3f) throw new Exception("0x7E must decode to 448 (max)");
        if (!float.IsNaN(Dtypes.F8E4M3ToFloat(0x7F))) throw new Exception("0x7F must decode to NaN");
        if (Math.Abs(Dtypes.F8E4M3ToFloat(0x08) - (1f / 64f)) > 1e-9f) throw new Exception("smallest normal must be 2^-6");

        // round-trip precision on a random spread of magnitudes
        // (subnormals |v| < 2^-6 have coarse relative precision by format design — use an
        //  absolute tolerance of half the subnormal spacing there, like torch's cast does)
        var rng = new Random(77);
        for (int i = 0; i < 4096; i++)
        {
            float v = (float)(rng.NextDouble() * 2 - 1) * MathF.Pow(10f, rng.Next(-4, 3));
            byte e = Dtypes.FloatToF8E4M3(v);
            float d = Dtypes.F8E4M3ToFloat(e);
            if (float.IsNaN(v)) continue;
            bool subnormal = MathF.Abs(v) < (1f / 64f);
            float err = subnormal ? MathF.Abs(d - v) : MathF.Abs(d - v) / MathF.Abs(v);
            float tol = subnormal ? 1f / 1024f : 0.063f;
            if (v != 0 && err > tol)
                throw new Exception($"fp8 roundtrip off: {v} -> {d}");
        }

        // block-scale reader: w = fp8 * scale[row/128, col/128]
        const int rows = 256, cols = 128, scRows = 2, scCols = 1;
        var w8 = new byte[rows * cols];
        var scale = new float[scRows * scCols];
        var expect = new float[rows * cols];
        for (int r = 0; r < scRows; r++) scale[r] = 0.25f * (r + 1);
        for (int i = 0; i < rows * cols; i++)
        {
            int row = i / cols, col = i % cols;
            float baseVal = ((i * 37) % 17 - 8) * 0.25f;
            byte e = Dtypes.FloatToF8E4M3(baseVal);
            w8[i] = e;
            expect[i] = Dtypes.F8E4M3ToFloat(e) * scale[row / 128 * scCols + col / 128];
        }
        var scaleBytes = new byte[scRows * scCols * 2];
        for (int i = 0; i < scale.Length; i++)
            BitConverter.GetBytes(Dtypes.FloatToBf16(scale[i])).CopyTo(scaleBytes, i * 2);

        string stPath = Path.Combine(Path.GetTempPath(), $"qr_fp8_{Guid.NewGuid():N}.safetensors");
        try
        {
            File.WriteAllBytes(stPath, BuildSafeTensors(new[]
            {
                ("w.weight", "F8_E4M3", new long[] { rows, cols }, w8),
                ("w.weight_scale_inv", "BF16", new long[] { scRows, scCols }, scaleBytes),
            }));
            using var st = SafeTensorsFile.Open(stPath);
            var w = st.Tensors["w.weight"];
            var sc = st.Tensors["w.weight_scale_inv"];
            var got = new float[rows * cols];
            st.ReadChunkF32(w, 0, got, sc);
            for (int i = 0; i < got.Length; i++)
                if (MathF.Abs(got[i] - expect[i]) > 1e-6f)
                    throw new Exception($"fp8 block-scale mismatch at {i}: {got[i]} vs {expect[i]}");
        }
        finally { if (File.Exists(stPath)) File.Delete(stPath); }
    }

    /// <summary>mHC: Sinkhorn output is (approximately) doubly stochastic; collapse/combine
    /// match the reference formulas.</summary>
    public static unsafe void MhcSinkhornAndCombine()
    {
        int n = 4, iters = 20;
        var m = new float[n * n];
        var rng = new Random(13);
        // Sinkhorn consumes the POSITIVE softmax+bias output (as in HcFinish) — start positive
        for (int i = 0; i < m.Length; i++) m[i] = (float)rng.NextDouble() + 0.05f;
        fixed (float* pm = m)
            Glm5Ops.Sinkhorn(pm, n, iters, 1e-6f);
        for (int i = 0; i < n; i++)
        {
            float rs = 0, cs = 0;
            for (int j = 0; j < n; j++) { rs += m[i * n + j]; cs += m[j * n + i]; }
            if (MathF.Abs(rs - 1f) > 1e-3f || MathF.Abs(cs - 1f) > 1e-3f)
                throw new Exception($"sinkhorn not doubly stochastic: row {i} sum {rs}, col {i} sum {cs}");
        }

        // HcFinish: collapsed must equal sum_i pre_i * streams_i with pre = sigmoid(w*s+b)+eps
        int dim = 8;
        var w = new float[(2 + n) * n];
        var b = new float[(2 + n) * n];
        var s = new[] { 0.7f, 1.3f, 0.9f };
        var streams = new float[n * dim];
        foreach (float[] arr in new[] { w, b, streams })
            for (int i = 0; i < arr.Length; i++) arr[i] = (float)(rng.NextDouble() * 2 - 1);
        var post = new float[n];
        var comb = new float[n * n];
        var collapsed = new float[dim];
        fixed (float* pw = w, pb = b, ps = s, pst = streams, ppost = post, pcomb = comb, pc = collapsed)
            Glm5Ops.HcFinish(pw, pb, ps, n, 1e-6f, iters, pst, dim, ppost, pcomb, pc);

        for (int i = 0; i < n; i++)
        {
            float pre = 1f / (1f + MathF.Exp(-(w[i] * s[0] + b[i]))) + 1e-6f;
            for (int d = 0; d < dim; d++)
            {
                float expect = 0;
                for (int k = 0; k < n; k++)
                    expect += (1f / (1f + MathF.Exp(-(w[k] * s[0] + b[k]))) + 1e-6f) * streams[k * dim + d];
                if (MathF.Abs(collapsed[d] - expect) > 1e-4f)
                    throw new Exception($"HcFinish collapse mismatch at {d}: {collapsed[d]} vs {expect}");
            }
            float postExpect = 2f / (1f + MathF.Exp(-(w[n + i] * s[1] + b[n + i])));
            if (MathF.Abs(post[i] - postExpect) > 1e-5f)
                throw new Exception($"HcFinish post mismatch: {post[i]} vs {postExpect}");
        }

        // comb must come from the THIRD slice of w ([pre | post | comb] layout, offset 2n)
        for (int i = 0; i < n; i++)
        {
            float max = float.MinValue;
            var raw = new float[n];
            for (int j = 0; j < n; j++)
            {
                raw[j] = w[2 * n + i * n + j] * s[2] + b[2 * n + i * n + j];
                if (raw[j] > max) max = raw[j];
            }
            float sum = 0;
            for (int j = 0; j < n; j++) { raw[j] = MathF.Exp(raw[j] - max); sum += raw[j]; }
            // rows of the comb matrix are softmax-normalized BEFORE the +eps and Sinkhorn,
            // so only the relative pattern of a dominant row must survive: check argmax
            int argmaxGot = 0, argmaxExp = 0;
            for (int j = 1; j < n; j++)
            {
                if (comb[i * n + j] > comb[i * n + argmaxGot]) argmaxGot = j;
                if (raw[j] > raw[argmaxExp]) argmaxExp = j;
            }
            if (argmaxGot != argmaxExp)
                throw new Exception($"comb row {i} argmax {argmaxGot} != expected {argmaxExp} (wrong w slice?)");
        }

        // HcCombine: streams[i] = post[i]*blockOut + sum_m comb[m,i]*streams[m]
        var blockOut = new float[dim];
        for (int i = 0; i < dim; i++) blockOut[i] = (float)(rng.NextDouble() * 2 - 1);
        var before = (float[])streams.Clone();
        var tmp = new float[n * dim];
        fixed (float* pst = streams, pbo = blockOut, ppost2 = post, pcomb2 = comb, ptmp = tmp)
            Glm5Ops.HcCombine(pst, pbo, ppost2, pcomb2, n, dim, ptmp);
        for (int i = 0; i < n; i++)
            for (int d = 0; d < dim; d++)
            {
                float expect = post[i] * blockOut[d];
                for (int mm = 0; mm < n; mm++) expect += comb[mm * n + i] * before[mm * dim + d];
                if (MathF.Abs(streams[i * dim + d] - expect) > 1e-4f)
                    throw new Exception($"HcCombine mismatch at stream {i} dim {d}");
            }
    }

    /// <summary>The absorbed MLA path must equal naive expanded K/V attention (exact math).</summary>
    public static unsafe void MlaAbsorbedVsExpanded()
    {
        int heads = 4, kvLora = 8, nope = 4, vdim = 4, seq = 10;
        int perHead = nope + vdim;
        var rng = new Random(29);
        float R() => (float)(rng.NextDouble() * 2 - 1);

        var q = new float[heads * nope];
        var kvb = new float[heads * perHead * kvLora];   // [heads*(nope+v), kvLora] row-major
        var lat = new float[seq * kvLora];
        for (int i = 0; i < q.Length; i++) q[i] = R();
        for (int i = 0; i < kvb.Length; i++) kvb[i] = R();
        for (int i = 0; i < lat.Length; i++) lat[i] = R();

        // ---- naive reference: expand K/V per token, standard attention ----
        var attnRef = new float[heads * vdim];
        float scale = 1f / MathF.Sqrt(nope);
        for (int h = 0; h < heads; h++)
        {
            var scores = new double[seq];
            for (int t = 0; t < seq; t++)
            {
                double dot = 0;
                for (int j = 0; j < nope; j++)
                {
                    double kj = 0;
                    for (int k = 0; k < kvLora; k++)
                        kj += (double)kvb[(h * perHead + j) * kvLora + k] * lat[t * kvLora + k];
                    dot += (double)q[h * nope + j] * kj;
                }
                scores[t] = dot * scale;
            }
            double max = scores.Max();
            double sum = 0;
            for (int t = 0; t < seq; t++) { scores[t] = Math.Exp(scores[t] - max); sum += scores[t]; }
            for (int j = 0; j < vdim; j++)
            {
                double acc = 0;
                for (int t = 0; t < seq; t++)
                {
                    double vj = 0;
                    for (int k = 0; k < kvLora; k++)
                        vj += (double)kvb[(h * perHead + nope + j) * kvLora + k] * lat[t * kvLora + k];
                    acc += scores[t] / sum * vj;
                }
                attnRef[h * vdim + j] = (float)acc;
            }
        }

        // ---- absorbed path (same math the engine uses) ----
        var attnAbs = new float[heads * vdim];
        fixed (float* pq = q, pk = kvb, pl = lat, pa = attnAbs, prow = new float[kvLora], pqt = new float[heads * kvLora], pu = new float[kvLora], psc = new float[seq])
        {
            for (int h = 0; h < heads; h++)
            {
                float* qh2 = pqt + h * kvLora;
                for (int k = 0; k < kvLora; k++) qh2[k] = 0;
                for (int j = 0; j < nope; j++)
                {
                    for (int k = 0; k < kvLora; k++) prow[k] = pk[(h * perHead + j) * kvLora + k];
                    float qv = pq[h * nope + j];
                    for (int k = 0; k < kvLora; k++) qh2[k] += qv * prow[k];
                }
                float* sc = psc;
                for (int t = 0; t < seq; t++) sc[t] = SimdMath.DotF32(qh2, pl + t * kvLora, kvLora) * scale;
                // softmax (max-shifted)
                float mx = float.MinValue;
                for (int t = 0; t < seq; t++) if (sc[t] > mx) mx = sc[t];
                float sm = 0;
                for (int t = 0; t < seq; t++) { sc[t] = MathF.Exp(sc[t] - mx); sm += sc[t]; }
                float inv = 1f / sm;
                for (int t = 0; t < seq; t++) sc[t] *= inv;
                float* uh = pu;
                for (int k = 0; k < kvLora; k++) uh[k] = 0;
                for (int t = 0; t < seq; t++) SimdMath.AxpyF32(pl + t * kvLora, uh, sc[t], kvLora);
                for (int j = 0; j < vdim; j++)
                {
                    for (int k = 0; k < kvLora; k++) prow[k] = pk[(h * perHead + nope + j) * kvLora + k];
                    pa[h * vdim + j] = SimdMath.DotF32(prow, uh, kvLora);
                }
            }
        }

        double worst = 0;
        for (int i = 0; i < attnRef.Length; i++)
            worst = Math.Max(worst, MathF.Abs(attnAbs[i] - attnRef[i]));
        if (worst > 1e-4) throw new Exception($"absorbed MLA diverges from expanded reference: max err {worst:G3}");
    }

    /// <summary>The decode-time conv state step must equal a direct depthwise convolution
    /// over the whole sequence (zero left padding), including the silu.</summary>
    public static unsafe void KdaConvMatchesDirect()
    {
        int channels = 6, kernel = 4, seq = 12;
        var rng = new Random(31);
        var input = new float[seq * channels];
        var w = new float[channels * kernel];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1);

        // reference: out[c, t] = silu(sum_j w[c, j] * x[c, t - K + 1 + j]) with x padded with zeros
        var reference = new float[seq * channels];
        for (int t = 0; t < seq; t++)
            for (int c = 0; c < channels; c++)
            {
                float acc = 0;
                for (int j = 0; j < kernel; j++)
                {
                    int src = t - kernel + 1 + j;
                    if (src >= 0) acc += w[c * kernel + j] * input[src * channels + c];
                }
                reference[t * channels + c] = TransformerOps.Silu(acc);
            }

        // stateful decode: run the step seq times and compare each output
        var state = new float[(kernel - 1) * channels];
        var stepOut = new float[channels];
        fixed (float* pin = input, pw = w, pst = state, pout = stepOut)
        {
            for (int t = 0; t < seq; t++)
            {
                Glm5Ops.CausalConvStep(pin + t * channels, pst, pw, channels, kernel, pout);
                for (int c = 0; c < channels; c++)
                    if (MathF.Abs(stepOut[c] - reference[t * channels + c]) > 1e-5f)
                        throw new Exception($"conv step mismatch at t={t} c={c}: {stepOut[c]} vs {reference[t * channels + c]}");
            }
        }
    }

    // ------------------------------------------------------------------ format conversion tests

    /// <summary>
    /// Pins our GGML dequantizers to the llama.cpp reference semantics with HAND-COMPUTED
    /// blocks (independent of our own quantizers): legacy Q4_0 (18 B, signed delta) / Q4_1 /
    /// Q5_0 / Q5_1, Q8_0, and one hand-built block each for Q2_K / Q3_K / Q4_K / Q6_K
    /// covering the bit-plane ordering, the kmask scale shuffle and get_scale_min_k4.
    /// Also checks the writer-side ggml Q4_0 re-encoder (round trip + negative-delta rule).
    /// </summary>
    public static unsafe void GgmlLegacyCompat()
    {
        void Need(string what, float got, float want, float tol = 1e-6f)
        {
            if (MathF.Abs(got - want) > tol) throw new Exception($"{what}: got {got}, want {want}");
        }

        // ---- Q4_0 (18 B): d = -0.5; qs[0]=0x08 -> e0=(8-8)*d=0, e16=(0-8)*d=4;
        //      qs[1]=0x2A -> e1=(10-8)*d=-1, e17=(2-8)*d=3
        {
            var blk = new byte[18];
            blk[0] = 0x00; blk[1] = 0xB8;                 // f16 -0.5
            blk[2] = 0x08; blk[3] = 0x2A;
            var y = new float[32];
            GgmlDequant.DequantQ4_0(blk, y);
            Need("q4_0 e0", y[0], 0f); Need("q4_0 e16", y[16], 4f);
            Need("q4_0 e1", y[1], -1f); Need("q4_0 e17", y[17], 3f);
        }

        // ---- Q4_1 (20 B): d=0.5, m=1.0; qs[0]=0x12 -> e0=2*0.5+1=2, e16=1*0.5+1=1.5
        {
            var blk = new byte[20];
            blk[0] = 0; blk[1] = 0x38;                    // f16 0.5
            blk[2] = 0; blk[3] = 0x3C;                    // f16 1.0
            blk[4] = 0x12;
            var y = new float[32];
            GgmlDequant.DequantQ4_1(blk, y);
            Need("q4_1 e0", y[0], 2f); Need("q4_1 e16", y[16], 1.5f);
        }

        // ---- Q5_0 (22 B): d=-1; qh bit0=1 (elem0 5th bit), bit16=0; qs[0]=0x05
        //      e0=((5|16)-16)*d=-5, e16=((0|0)-16)*d=16
        {
            var blk = new byte[22];
            blk[0] = 0; blk[1] = 0xBC;                    // f16 -1.0
            blk[2] = 1;                                   // qh: bit0
            blk[6] = 0x05;
            var y = new float[32];
            GgmlDequant.DequantQ5_0(blk, y);
            Need("q5_0 e0", y[0], -5f); Need("q5_0 e16", y[16], 16f);
        }

        // ---- Q5_1 (24 B): d=1, m=1; qh bit0=1; qs[0]=0x05 -> e0=(5|16)+1=22, e16=0+1=1
        {
            var blk = new byte[24];
            blk[0] = 0; blk[1] = 0x3C;
            blk[2] = 0; blk[3] = 0x3C;
            blk[4] = 1;
            blk[8] = 0x05;
            var y = new float[32];
            GgmlDequant.DequantQ5_1(blk, y);
            Need("q5_1 e0", y[0], 22f); Need("q5_1 e16", y[16], 1f);
        }

        // ---- Q8_0 (34 B): d=0.25, qs[0]=-8 -> -2
        {
            var blk = new byte[34];
            blk[0] = 0; blk[1] = 0x34;                    // f16 0.25
            blk[2] = unchecked((byte)-8);
            var y = new float[32];
            BlockDequantizer.DequantQ8_0(blk, y);
            Need("q8_0 e0", y[0], -2f);
        }

        // ---- Q2_K (84 B): d=1, dmin=0.5; scales[0]=0x21 (sc=1,min=2), scales[8]=0x42;
        //      q[0]=0x03 -> e0=1*3-1=2; q[16]=0x02 -> e16=2-1=1; q[32]=0x03 -> e128=2*3-1.5=4.5
        {
            var blk = new byte[84];
            blk[80] = 0; blk[81] = 0x3C;                  // d = 1.0
            blk[82] = 0; blk[83] = 0x38;                  // dmin = 0.5
            blk[0] = 0x21; blk[1] = 0x21; blk[8] = 0x32;  // scales[0]/[1] for elems 0-15/16-31 (sc=1,min=2), scales[8] for 128-143 (sc=2,min=3)
            blk[16] = 0x03; blk[32] = 0x02; blk[48] = 0x03;
            var y = new float[256];
            BlockDequantizer.Dequantize(QuantFormat.Q2_K, blk, y);
            Need("q2_K e0 (scales[0])", y[0], 2f); Need("q2_K e16 (scales[1])", y[16], 1f); Need("q2_K e128 (scales[8])", y[128], 4.5f);
        }

        // ---- Q3_K (110 B): kmask shuffle pin. scales[] is a sequential int8 view over the
        //      four aux words: sc[0] = aux0.b0 = (s[96]&0xF) | ((s[104]&3)<<4) = 1|16 = 17 -> -15;
        //      sc[1] = aux0.b1 = (s[97]&0xF) | ((s[105]&3)<<4) = 2 -> -30. d = 0.25.
        //      q[0]=0x02 (blk[32]), hmask[0] bit0=1      -> e0  = 0.25*(-15)*2 = -7.5
        //      q[16]=0x01 (blk[48]), hmask[16]=0         -> e16 = 0.25*(-30)*(1-4) = 22.5
        {
            var blk = new byte[110];
            blk[108] = 0; blk[109] = 0x34;                // d = 0.25
            blk[96] = 0x01;                               // aux0.b0 low nibble
            blk[97] = 0x02;                               // aux0.b1 low nibble
            blk[104] = 0x21;                              // tmp.b0: 2 high bits -> sc[0] high nibble
            blk[0] = 0x01;                                // hmask[0] bit0 set
            blk[32] = 0x02;                               // q[0]  (qs @32)
            blk[48] = 0x01;                               // q[16]
            var y = new float[256];
            BlockDequantizer.Dequantize(QuantFormat.Q3_K, blk, y);
            Need("q3_K e0", y[0], -7.5f); Need("q3_K e16", y[16], 22.5f);
        }

        // ---- Q4_K (144 B): d=2, dmin=1; get_scale_min_k4(0): sc=scales[0]=3, m=scales[4]=1
        //      -> e0 = 2*3*(q[0]&0xF) - 1*1; get_scale_min_k4(1): sc=scales[1]=2, m=scales[5]=1
        //      -> e32 = 2*2*(q[0]>>4) - 1
        {
            var blk = new byte[144];
            blk[0] = 0; blk[1] = 0x40;                    // d = 2.0
            blk[2] = 0; blk[3] = 0x3C;                    // dmin = 1.0
            blk[4] = 3; blk[5] = 2; blk[8] = 1; blk[9] = 1;
            blk[16] = 0x5F;                               // low=15 (e0), high=5 (e32)
            var y = new float[256];
            BlockDequantizer.Dequantize(QuantFormat.Q4_K, blk, y);
            Need("q4_K e0", y[0], 2 * 3 * 15 - 1); Need("q4_K e32", y[32], 2 * 2 * 5 - 1);
        }

        // ---- Q6_K (210 B): ql[0]=0x21, ql[32]=0x43, qh[0]=0b00011001, scales[0]=2,[2]=1,[4]=3,[6]=-1,
        //      d=0.5 -> q1=17,q2=35,q3=18,q4=4; y0=0.5*2*(17-32)=-15, y32=1.5, y64=-21, y96=14
        {
            var blk = new byte[210];
            blk[208] = 0; blk[209] = 0x38;                // d = 0.5
            blk[0] = 0x21; blk[32] = 0x43;
            blk[128] = 0b00011001;
            blk[192] = 2; blk[194] = 1; blk[196] = 3; blk[198] = unchecked((byte)-1);
            var y = new float[256];
            BlockDequantizer.Dequantize(QuantFormat.Q6_K, blk, y);
            Need("q6_K e0", y[0], -15f); Need("q6_K e32", y[32], 1.5f);
            Need("q6_K e64", y[64], -21f); Need("q6_K e96", y[96], 14f);
        }

        // ---- writer-side ggml Q4_0: round trip + signed-delta convention (max>0 => d<0)
        {
            var rng = new Random(5);
            int n = 32 * 64;
            var x = new float[n];
            for (int i = 0; i < n; i++) x[i] = Synth.NextGaussian(rng) * 0.5f;
            var bytes = new byte[n / 32 * 18];
            GgmlDequant.QuantizeGgmlQ4_0(x, bytes);
            // ggml rule: d = maxSigned / -8 where maxSigned is the SIGNED value of the
            // max-abs element (d is positive exactly when that element is negative)
            float d0 = Dtypes.F16ToFloat((ushort)(bytes[0] | (bytes[1] << 8)));
            float amax = 0, maxSigned = 0;
            for (int i = 0; i < 32; i++) { float a = MathF.Abs(x[i]); if (a > amax) { amax = a; maxSigned = x[i]; } }
            float wantD = Dtypes.F16ToFloat(Dtypes.FloatToF16(maxSigned / -8f));
            if (MathF.Abs(d0 - wantD) > 1e-6f) throw new Exception($"ggml q4_0 delta {d0} != maxSigned/-8 ({wantD})");
            var y = new float[n];
            for (int b = 0; b < n / 32; b++)
                GgmlDequant.DequantQ4_0(bytes.AsSpan(b * 18, 18), y.AsSpan(b * 32, 32));
            double se = 0, seRef = 0;
            for (int i = 0; i < n; i++) { double e = y[i] - x[i]; se += e * e; seRef += (double)x[i] * x[i]; }
            double rms = Math.Sqrt(se / n) / Math.Sqrt(Math.Max(seRef / n, 1e-12));
            if (rms > 0.11) throw new Exception($"ggml q4_0 round trip rel RMS {rms:F3} > 0.11");
        }
    }

    /// <summary>
    /// Full conversion loop: a hand-built GGUF (F32 + true-ggml Q4_0 + Q8_0 + 3-D MoE experts +
    /// BPE tokenizer KVs) -> .qw (exercising the 18-byte Q4_0 import path, expert splitting and
    /// byte-exact Q8_0 passthrough) -> .qw -> GGUF export (verifying metadata restore, 3-D expert
    /// reassembly and byte-exact passthrough) -> re-import (second round must be byte-stable).
    /// </summary>
    public static void GgufQ4ImportAndWriterRoundTrip()
    {
        string ggufPath = Path.Combine(Path.GetTempPath(), $"qr_gg_{Guid.NewGuid():N}.gguf");
        string qwPath = Path.Combine(Path.GetTempPath(), $"qr_gg_{Guid.NewGuid():N}.qw");
        string outGguf = Path.Combine(Path.GetTempPath(), $"qr_gg_{Guid.NewGuid():N}.out.gguf");
        string qw2Path = Path.Combine(Path.GetTempPath(), $"qr_gg_{Guid.NewGuid():N}.qw");
        try
        {
            // ---- build a mini GGUF
            const int dim = 64, ffn = 128, vocab = 32, kvDim = 32;
            var kvs = new List<(string, GgufFile.GgufValueType, object)>
            {
                ("general.architecture", GgufFile.GgufValueType.String, "llama"),
                ("general.name", GgufFile.GgufValueType.String, "mini"),
                ("llama.context_length", GgufFile.GgufValueType.U32, 512u),
                ("llama.embedding_length", GgufFile.GgufValueType.U32, (uint)dim),
                ("llama.block_count", GgufFile.GgufValueType.U32, 1u),
                ("llama.feed_forward_length", GgufFile.GgufValueType.U32, (uint)ffn),
                ("llama.attention.head_count", GgufFile.GgufValueType.U32, 4u),
                ("llama.attention.head_count_kv", GgufFile.GgufValueType.U32, 2u),
                ("llama.attention.layer_norm_rms_epsilon", GgufFile.GgufValueType.F32, 1e-5f),
                ("llama.rope.freq_base", GgufFile.GgufValueType.F32, 10000f),
                ("llama.vocab_size", GgufFile.GgufValueType.U32, (uint)vocab),
                ("tokenizer.ggml.model", GgufFile.GgufValueType.String, "gpt2"),
                ("tokenizer.ggml.tokens", GgufFile.GgufValueType.Array, new[] { "<s>", "hello", "world", "</s>" }),
                ("tokenizer.ggml.merges", GgufFile.GgufValueType.Array, new[] { "he llo", "wo rld" }),
                ("tokenizer.ggml.token_type", GgufFile.GgufValueType.Array, new long[] { 2, 0, 0, 2 }),
                ("tokenizer.ggml.eos_token_id", GgufFile.GgufValueType.U32, 3u),
            };

            // token_embd F32 [dim, vocab] (gguf ne = {dim, vocab})
            var emb = new float[dim * vocab];
            for (int i = 0; i < emb.Length; i++) emb[i] = (i % 7) * 0.125f - 0.375f;
            var embBytes = new byte[emb.Length * 4];
            Buffer.BlockCopy(emb, 0, embBytes, 0, embBytes.Length);

            // attn_q: true ggml Q4_0, ne = {dim, 128} -> 8192 elems, 256 blocks
            var q4 = new byte[256 * 18];
            var q4Expected = new float[128 * dim];
            for (int b = 0; b < 256; b++)
            {
                float d = (b % 2 == 0) ? 0.5f : -0.25f;
                ushort dh = Dtypes.FloatToF16(d);
                q4[b * 18] = (byte)dh; q4[b * 18 + 1] = (byte)(dh >> 8);
                for (int j = 0; j < 16; j++)
                {
                    int lo = (b + j) % 16, hi = (b + j + 5) % 16;
                    q4[b * 18 + 2 + j] = (byte)(lo | (hi << 4));
                    q4Expected[b * 32 + j] = d * (lo - 8);
                    q4Expected[b * 32 + j + 16] = d * (hi - 8);
                }
            }

            // attn_v: Q8_0 ne = {dim, kvDim}
            var v8 = new byte[(dim * kvDim) / 32 * 34];
            new Random(9).NextBytes(v8);

            // experts: ffn_gate 3-D ne = {dim, ffn, 2} in Q8_0
            var exp = new byte[2 * (ffn * dim) / 32 * 34];
            new Random(10).NextBytes(exp);

            var tensors = new List<(string, long[], int, byte[])>
            {
                ("token_embd.weight", new[] { (long)dim, vocab }, 0, embBytes),
                ("blk.0.attn_q.weight", new[] { (long)dim, 128 }, 2, q4),
                ("blk.0.attn_v.weight", new[] { (long)dim, kvDim }, 8, v8),
                ("blk.0.ffn_gate_exps.weight", new[] { (long)dim, ffn, 2L }, 8, exp),
            };
            File.WriteAllBytes(ggufPath, BuildTestGguf(kvs, tensors));

            // ---- gguf -> qw
            var opt = new ConvertOptions { Attention = QuantFormat.Q8_0, Ffn = QuantFormat.Q8_0 };
            ModelConverter.ConvertGguf(ggufPath, qwPath, opt);
            using (var store = ModelStore.Open(qwPath))
            {
                var embQw = store.Get("tok_emb");
                if (embQw.Format != QuantFormat.F16) throw new Exception($"tok_emb format {embQw.Format} (small fp tensor -> f16 expected)");
                var embBack = new float[dim * vocab];
                BlockDequantizer.Dequantize(QuantFormat.F16, embQw.Bytes, embBack);
                for (int i = 0; i < embBack.Length; i++)
                    if (embBack[i] != emb[i]) throw new Exception($"tok_emb[{i}] {embBack[i]} != {emb[i]} (f16 lossless for grid values)");

                var wq = store.Get("l0.wq");
                if (wq.Format != QuantFormat.Q8_0) throw new Exception($"l0.wq format {wq.Format}");
                if (wq.Dims[0] != 128 || wq.Dims[1] != dim) throw new Exception($"l0.wq dims {string.Join('x', wq.Dims)}");
                var wqF = new float[128 * dim];
                BlockDequantizer.Dequantize(QuantFormat.Q8_0, wq.Bytes, wqF);
                double se = 0, seRef = 0;
                for (int i = 0; i < wqF.Length; i++) { double e = wqF[i] - q4Expected[i]; se += e * e; seRef += (double)q4Expected[i] * q4Expected[i]; }
                double rms = Math.Sqrt(se / wqF.Length) / Math.Sqrt(Math.Max(seRef / wqF.Length, 1e-12));
                if (rms > 0.02) throw new Exception($"q4_0 import rel RMS {rms:F3} > 0.02 (8-bit re-encode of a 4-bit grid)");

                if (!store.Has("l0.e0.w_gate") || !store.Has("l0.e1.w_gate")) throw new Exception("experts not split");
                var e0 = store.Get("l0.e0.w_gate");
                if (e0.Dims[0] != ffn || e0.Dims[1] != dim) throw new Exception($"e0 dims {string.Join('x', e0.Dims)}");

                // header must embed the source gguf metadata
                if (!store.ConfigJson.TryGetProperty("gguf_kvs", out _)) throw new Exception("gguf_kvs not embedded in .qw header");
            }

            // ---- qw -> gguf (export)
            GgufExporter.Export(qwPath, outGguf, new GgufExportOptions());
            using (var back = GgufFile.Open(outGguf))
            {
                if (back.Tensors.Count != 4) throw new Exception($"export tensor count {back.Tensors.Count} != 4");
                if (back.Kv("tokenizer.ggml.tokens") is not string[] toks || toks.Length != 4 || toks[1] != "hello")
                    throw new Exception("tokenizer KVs not restored from .qw header");
                if (back.Kv("llama.block_count") is not object || Convert.ToInt64(back.Kv("llama.block_count")!) != 1)
                    throw new Exception("arch KVs not written");

                var vInfo = back.Tensors.First(t => t.Name == "blk.0.attn_v.weight");
                if (vInfo.GgmlType != 8) throw new Exception($"attn_v exported as {GgufFile.GgmlTypeName(vInfo.GgmlType)} (q8_0 passthrough expected)");
                var vData = new byte[vInfo.SizeBytes];
                back.Stream.Position = back.DataStart + vInfo.Offset;
                back.Stream.ReadExactly(vData);
                if (!vData.SequenceEqual(v8)) throw new Exception("q8_0 passthrough not byte-exact through qw->gguf");

                var gate = back.Tensors.First(t => t.Name == "blk.0.ffn_gate_exps.weight");
                if (gate.Dims.Length != 3 || gate.Dims[2] != 2) throw new Exception($"experts not reassembled 3-D: {string.Join('x', gate.Dims)}");
                var gateData = new byte[gate.SizeBytes];
                back.Stream.Position = back.DataStart + gate.Offset;
                back.Stream.ReadExactly(gateData);
                if (!gateData.SequenceEqual(exp)) throw new Exception("expert 3-D reassembly not byte-exact");
            }

            // ---- gguf -> qw again: must be byte-stable for the passthrough tensors
            ModelConverter.ConvertGguf(outGguf, qw2Path, opt);
            using (var s1 = ModelStore.Open(qwPath))
            using (var s2 = ModelStore.Open(qw2Path))
            {
                if (!s1.Get("l0.wv").Bytes.SequenceEqual(s2.Get("l0.wv").Bytes))
                    throw new Exception("second import not byte-stable (wv)");
                if (!s1.Get("l0.wq").Bytes.SequenceEqual(s2.Get("l0.wq").Bytes))
                    throw new Exception("second import not byte-stable (wq)");
            }
        }
        finally
        {
            foreach (var p in new[] { ggufPath, qwPath, outGguf, qw2Path }) if (File.Exists(p)) File.Delete(p);
        }
    }

    /// <summary>
    /// qw -> HuggingFace safetensors export: shard files + index.json + config.json, HF shapes
    /// ([out, in], embedding [vocab, dim]), F16 values bit-equal to the .qw dequant, and
    /// tie_word_embeddings inferred from the missing lm_head.
    /// </summary>
    public static void HfExportFromQw()
    {
        string stPath = Path.Combine(Path.GetTempPath(), $"qr_hf_{Guid.NewGuid():N}.safetensors");
        string qwPath = Path.Combine(Path.GetTempPath(), $"qr_hf_{Guid.NewGuid():N}.qw");
        string outDir = Path.Combine(Path.GetTempPath(), $"qr_hf_{Guid.NewGuid():N}");
        try
        {
            const int dim = 256, ffn = 512, vocab = 128, kvDim = 128;
            var rng = new Random(23);
            var tensors = new List<(string Name, string Dtype, long[] Shape, byte[] Data)>
            {
                ("model.embed_tokens.weight", "F16", new long[] { vocab, dim }, F16Tensor(rng, vocab, dim, 0.05f)),
                ("model.norm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                ("model.layers.0.input_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                ("model.layers.0.post_attention_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                ("model.layers.0.self_attn.q_proj.weight", "F16", new long[] { dim, dim }, F16Tensor(rng, dim, dim, 0.05f)),
                ("model.layers.0.self_attn.k_proj.weight", "F16", new long[] { kvDim, dim }, F16Tensor(rng, kvDim, dim, 0.05f)),
                ("model.layers.0.self_attn.v_proj.weight", "F16", new long[] { kvDim, dim }, F16Tensor(rng, kvDim, dim, 0.05f)),
                ("model.layers.0.self_attn.o_proj.weight", "F16", new long[] { dim, dim }, F16Tensor(rng, dim, dim, 0.05f)),
                ("model.layers.0.mlp.gate_proj.weight", "F16", new long[] { ffn, dim }, F16Tensor(rng, ffn, dim, 0.05f)),
                ("model.layers.0.mlp.up_proj.weight", "F16", new long[] { ffn, dim }, F16Tensor(rng, ffn, dim, 0.05f)),
                ("model.layers.0.mlp.down_proj.weight", "F16", new long[] { dim, ffn }, F16Tensor(rng, dim, ffn, 0.05f)),
            };
            File.WriteAllBytes(stPath, BuildSafeTensors(tensors));
            ModelConverter.ConvertSafeTensors(new[] { stPath }, null, qwPath, new ConvertOptions());

            HfExporter.FromQw(qwPath, outDir, new HfExportOptions { Shards = 2 });

            // ---- files exist
            var shard1 = Path.Combine(outDir, "model-00001-of-00002.safetensors");
            var shard2 = Path.Combine(outDir, "model-00002-of-00002.safetensors");
            if (!File.Exists(shard1) || !File.Exists(shard2)) throw new Exception("shard files not created");

            // ---- config.json
            using var cfgDoc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(outDir, "config.json")));
            var cfgEl = cfgDoc.RootElement;
            foreach (var (key, want) in new (string, object)[]
                     {
                         ("model_type", "llama"), ("hidden_size", 256L), ("num_hidden_layers", 1L),
                         ("num_attention_heads", 4L), ("num_key_value_heads", 2L), ("vocab_size", 128L),
                         ("tie_word_embeddings", true), ("torch_dtype", "float16")
                     })
            {
                if (!cfgEl.TryGetProperty(key, out var el)) throw new Exception($"config.json missing {key}");
                bool ok = el.ValueKind switch
                {
                    System.Text.Json.JsonValueKind.Number => el.GetInt64() == Convert.ToInt64(want),
                    System.Text.Json.JsonValueKind.True => want is bool b && b,
                    System.Text.Json.JsonValueKind.False => want is bool b2 && !b2,
                    System.Text.Json.JsonValueKind.String => el.GetString() == (string)want,
                    _ => false
                };
                if (!ok) throw new Exception($"config.json {key} = {el.GetRawText()}, want {want}");
            }

            // ---- index.json covers everything
            using var idxDoc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(outDir, "model.safetensors.index.json")));
            var weightMap = idxDoc.RootElement.GetProperty("weight_map");
            int mapped = weightMap.EnumerateObject().Count();
            if (mapped != tensors.Count) throw new Exception($"index weight_map has {mapped}, want {tensors.Count}");

            // ---- values: locate each tensor through the index (shard split is size-driven)
            using var store = ModelStore.Open(qwPath);
            string ShardOf(string hfName) => Path.Combine(outDir, weightMap.GetProperty(hfName).GetString()!);

            using var shEmb = SafeTensorsFile.Open(ShardOf("model.embed_tokens.weight"));
            var embInfo = shEmb.Tensors["model.embed_tokens.weight"];
            if (embInfo.Dtype != "F16" || embInfo.Shape[0] != vocab || embInfo.Shape[1] != dim)
                throw new Exception($"embed shape/dtype {embInfo.Dtype} [{string.Join(',', embInfo.Shape)}]");
            var embSrc = new float[vocab * dim];
            BlockDequantizer.Dequantize(QuantFormat.F16, store.Get("tok_emb").Bytes, embSrc);
            var embBack = shEmb.ReadTensorF32(embInfo);
            for (int i = 0; i < embBack.Length; i++)
                if (embBack[i] != embSrc[i]) throw new Exception($"embed value mismatch at {i}: {embBack[i]} vs {embSrc[i]}");

            // ---- q_proj: exported f16 must equal dequant(q4_k) -> f16 bit-for-bit
            using var shQ = SafeTensorsFile.Open(ShardOf("model.layers.0.self_attn.q_proj.weight"));
            var qInfo = shQ.Tensors["model.layers.0.self_attn.q_proj.weight"];
            if (qInfo.Shape[0] != dim || qInfo.Shape[1] != dim) throw new Exception($"q_proj shape [{string.Join(',', qInfo.Shape)}]");
            var qBack = shQ.ReadTensorF32(qInfo);
            var qSrc = new float[dim * dim];
            BlockDequantizer.Dequantize(QuantFormat.Q4_K, store.Get("l0.wq").Bytes, qSrc);
            for (int i = 0; i < qBack.Length; i++)
            {
                float want = Dtypes.F16ToFloat(Dtypes.FloatToF16(qSrc[i]));
                if (qBack[i] != want) throw new Exception($"q_proj value mismatch at {i}: {qBack[i]} vs {want}");
            }
        }
        finally
        {
            if (File.Exists(stPath)) File.Delete(stPath);
            if (File.Exists(qwPath)) File.Delete(qwPath);
            if (Directory.Exists(outDir)) Directory.Delete(outDir, recursive: true);
        }
    }

    /// <summary>gguf -> HF export must generate a working BPE tokenizer.json from the GGUF
    /// metadata (specials from token_type, merges preserved) plus a correct config.json.</summary>
    public static void HfExportFromGguf()
    {
        string ggufPath = Path.Combine(Path.GetTempPath(), $"qr_hfg_{Guid.NewGuid():N}.gguf");
        string outDir = Path.Combine(Path.GetTempPath(), $"qr_hfg_{Guid.NewGuid():N}");
        try
        {
            const int dim = 64, vocab = 32;
            var kvs = new List<(string, GgufFile.GgufValueType, object)>
            {
                ("general.architecture", GgufFile.GgufValueType.String, "qwen2"),
                ("qwen2.context_length", GgufFile.GgufValueType.U32, 512u),
                ("qwen2.embedding_length", GgufFile.GgufValueType.U32, (uint)dim),
                ("qwen2.block_count", GgufFile.GgufValueType.U32, 1u),
                ("qwen2.feed_forward_length", GgufFile.GgufValueType.U32, 128u),
                ("qwen2.attention.head_count", GgufFile.GgufValueType.U32, 4u),
                ("qwen2.attention.head_count_kv", GgufFile.GgufValueType.U32, 2u),
                ("qwen2.attention.layer_norm_rms_epsilon", GgufFile.GgufValueType.F32, 1e-5f),
                ("qwen2.rope.freq_base", GgufFile.GgufValueType.F32, 1000000f),
                ("qwen2.vocab_size", GgufFile.GgufValueType.U32, (uint)vocab),
                ("tokenizer.ggml.model", GgufFile.GgufValueType.String, "gpt2"),
                ("tokenizer.ggml.tokens", GgufFile.GgufValueType.Array, new[] { "<|im_start|>", "hi", "there", "<|im_end|>" }),
                ("tokenizer.ggml.merges", GgufFile.GgufValueType.Array, new[] { "h i", "th ere" }),
                ("tokenizer.ggml.token_type", GgufFile.GgufValueType.Array, new long[] { 2, 0, 0, 2 }),
                ("tokenizer.ggml.eos_token_id", GgufFile.GgufValueType.U32, 3u),
            };
            var emb = new byte[dim * vocab * 2];
            var tensors = new List<(string, long[], int, byte[])>
            {
                ("token_embd.weight", new[] { (long)dim, vocab }, 1, emb),
            };
            File.WriteAllBytes(ggufPath, BuildTestGguf(kvs, tensors));

            HfExporter.FromGguf(ggufPath, outDir, new HfExportOptions());

            using var tj = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(outDir, "tokenizer.json")));
            var model = tj.RootElement.GetProperty("model");
            if (model.GetProperty("type").GetString() != "BPE") throw new Exception("tokenizer.json model type");
            if (model.GetProperty("vocab").GetProperty("hi").GetInt32() != 1) throw new Exception("tokenizer vocab ids");
            if (model.GetProperty("merges").EnumerateArray().First().GetString() != "h i") throw new Exception("tokenizer merges");
            int added = tj.RootElement.GetProperty("added_tokens").EnumerateArray().Count();
            if (added != 2) throw new Exception($"added_tokens {added} != 2 (two CONTROL tokens)");

            using var cfg = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(outDir, "config.json")));
            if (cfg.RootElement.GetProperty("architectures")[0].GetString() != "Qwen2ForCausalLM")
                throw new Exception("architectures not derived from gguf arch");
            if (cfg.RootElement.GetProperty("eos_token_id").GetInt64() != 3) throw new Exception("eos_token_id");

            if (!File.Exists(Path.Combine(outDir, "generation_config.json"))) throw new Exception("generation_config.json missing");
            if (!File.Exists(Path.Combine(outDir, "tokenizer_config.json"))) throw new Exception("tokenizer_config.json missing");
        }
        finally
        {
            if (File.Exists(ggufPath)) File.Delete(ggufPath);
            if (Directory.Exists(outDir)) Directory.Delete(outDir, recursive: true);
        }
    }

    // ------------------------------------------------------------------ gguf builder (tests)

    private static void GwTestString(BinaryWriter bw, string s)
    {
        var b = System.Text.Encoding.UTF8.GetBytes(s);
        bw.Write((ulong)b.Length);
        bw.Write(b);
    }

    private static byte[] BuildTestGguf(
        List<(string Key, GgufFile.GgufValueType Type, object Value)> kvs,
        List<(string Name, long[] Ne, int GgmlType, byte[] Data)> tensors)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);
        bw.Write('G'); bw.Write('G'); bw.Write('U'); bw.Write('F');
        bw.Write(3u);
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)kvs.Count);
        foreach (var (key, type, value) in kvs)
        {
            GwTestString(bw, key);
            bw.Write((uint)type);
            switch (type)
            {
                case GgufFile.GgufValueType.String: GwTestString(bw, (string)value); break;
                case GgufFile.GgufValueType.U32: bw.Write(Convert.ToUInt32(value)); break;
                case GgufFile.GgufValueType.F32: bw.Write(Convert.ToSingle(value)); break;
                case GgufFile.GgufValueType.Array when value is string[] a:
                    bw.Write((uint)GgufFile.GgufValueType.String);
                    bw.Write((ulong)a.Length);
                    foreach (var s in a) GwTestString(bw, s);
                    break;
                case GgufFile.GgufValueType.Array when value is long[] a:
                    bw.Write((uint)GgufFile.GgufValueType.I32);
                    bw.Write((ulong)a.Length);
                    foreach (var l in a) bw.Write((int)l);
                    break;
                default: throw new ArgumentException($"test gguf kv {key}");
            }
        }
        // tensor infos with 32-byte-aligned data offsets
        var offsets = new List<long>();
        long off = 0;
        foreach (var t in tensors)
        {
            GwTestString(bw, t.Name);
            bw.Write(t.Ne.Length);
            foreach (var d in t.Ne) bw.Write((ulong)d);
            bw.Write((uint)t.GgmlType);
            long aligned = (off + 31) / 32 * 32;
            bw.Write((ulong)aligned);
            offsets.Add(aligned);
            off = aligned + t.Data.Length;
        }
        long dataStart = (bw.BaseStream.Position + 31) / 32 * 32;
        while (bw.BaseStream.Position < dataStart) bw.Write((byte)0);
        for (int i = 0; i < tensors.Count; i++)
        {
            while (bw.BaseStream.Position - dataStart < offsets[i]) bw.Write((byte)0);
            bw.Write(tensors[i].Data);
        }
        bw.Flush();
        return ms.ToArray();
    }

    private static byte[] F16Tensor(Random rng, long rows, long cols, float scale)
    {
        long n = rows * cols;
        var f = new float[n];
        for (long i = 0; i < n; i++) f[i] = Synth.NextGaussian(rng) * scale;
        var b = new byte[n * 2];
        for (long i = 0; i < n; i++) BitConverter.GetBytes(Dtypes.FloatToF16(f[i])).CopyTo(b, i * 2);
        return b;
    }

    private static byte[] F32Tensor(Random rng, long n)
    {
        var f = new float[n];
        for (long i = 0; i < n; i++) f[i] = Synth.NextGaussian(rng);
        var b = new byte[n * 4];
        Buffer.BlockCopy(f, 0, b, 0, b.Length);
        return b;
    }

    /// <summary>
    /// K-quants (256-wide blocks) need every row to be a whole number of blocks — the dot
    /// kernels walk blocks along rows. An ffn of 128 with the default q4_k used to write a
    /// broken model (NaN logits at inference); the converter now falls back to q8_0 (or f32)
    /// for such rows and prints a note.
    /// </summary>
    public static void KQuantRowFallback()
    {
        // direct contract
        if (ModelConverter.SafeQuantForRows(QuantFormat.Q4_K, 256) != QuantFormat.Q4_K)
            throw new Exception("aligned row must keep q4_k");
        if (ModelConverter.SafeQuantForRows(QuantFormat.Q4_K, 128) != QuantFormat.Q8_0)
            throw new Exception("128-wide row must fall back to q8_0");
        if (ModelConverter.SafeQuantForRows(QuantFormat.Q6_K, 100) != QuantFormat.F32)
            throw new Exception("row not divisible by 32 must fall back to f32");
        if (ModelConverter.SafeQuantForRows(QuantFormat.F16, 100) != QuantFormat.F16)
            throw new Exception("fp formats must be untouched");
        if (ModelConverter.SafeQuantForRows(QuantFormat.TRI, 128) != QuantFormat.TRI)
            throw new Exception("TRI (64-wide block) must accept 128-wide rows");

        // integration: ffn=128 model with default (q4_k) options converts safe and runs finite
        string stPath = Path.Combine(Path.GetTempPath(), $"qr_kq_{Guid.NewGuid():N}.safetensors");
        string qwPath = Path.Combine(Path.GetTempPath(), $"qr_kq_{Guid.NewGuid():N}.qw");
        try
        {
            const int dim = 256, ffn = 128, vocab = 128, kvDim = 128;   // ffn % 256 != 0
            var rng = new Random(41);
            var tensors = new List<(string Name, string Dtype, long[] Shape, byte[] Data)>
            {
                ("model.embed_tokens.weight", "F16", new long[] { vocab, dim }, F16Tensor(rng, vocab, dim, 0.05f)),
                ("model.norm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                ("model.layers.0.input_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                ("model.layers.0.post_attention_layernorm.weight", "F32", new long[] { dim }, F32Tensor(rng, dim)),
                ("model.layers.0.self_attn.q_proj.weight", "F16", new long[] { dim, dim }, F16Tensor(rng, dim, dim, 0.05f)),
                ("model.layers.0.self_attn.k_proj.weight", "F16", new long[] { kvDim, dim }, F16Tensor(rng, kvDim, dim, 0.05f)),
                ("model.layers.0.self_attn.v_proj.weight", "F16", new long[] { kvDim, dim }, F16Tensor(rng, kvDim, dim, 0.05f)),
                ("model.layers.0.self_attn.o_proj.weight", "F16", new long[] { dim, dim }, F16Tensor(rng, dim, dim, 0.05f)),
                ("model.layers.0.mlp.gate_proj.weight", "F16", new long[] { ffn, dim }, F16Tensor(rng, ffn, dim, 0.05f)),
                ("model.layers.0.mlp.up_proj.weight", "F16", new long[] { ffn, dim }, F16Tensor(rng, ffn, dim, 0.05f)),
                ("model.layers.0.mlp.down_proj.weight", "F16", new long[] { dim, ffn }, F16Tensor(rng, dim, ffn, 0.05f)),
            };
            File.WriteAllBytes(stPath, BuildSafeTensors(tensors));

            ModelConverter.ConvertSafeTensors(new[] { stPath }, null, qwPath, new ConvertOptions());
            using var store = ModelStore.Open(qwPath);
            if (store.Get("l0.w_gate").Format != QuantFormat.Q4_K)
                throw new Exception("w_gate (256-wide rows) should stay q4_k");
            var down = store.Get("l0.w_down");
            if (down.Format != QuantFormat.Q8_0)
                throw new Exception($"w_down (128-wide rows) should fall back to q8_0, got {down.Format}");

            using var eng = new StreamingEngine(store, new EngineOptions { PrefetchBudgetBytes = 0 });
            CheckFinite(eng.Forward(3, 0)[..16].ToArray());
            CheckFinite(eng.Forward(9, 1)[..16].ToArray());
        }
        finally
        {
            if (File.Exists(stPath)) File.Delete(stPath);
            if (File.Exists(qwPath)) File.Delete(qwPath);
        }
    }

    /// <summary>
    /// Regression: ModelStoreAppender.Plan computes 64-byte-aligned offsets, but Commit used to
    /// write tensor bodies back-to-back — every tensor planned after one whose byte size is not
    /// a multiple of 64 landed BEFORE its claimed offset (silent corruption; surfaced by
    /// GLM-5.3-Flash's 12-byte hc_*_scale / 32-byte hc_*_base tensors). Commit now seeks to each
    /// planned offset. This test appends odd-sized tensors in two chained waves and verifies the
    /// bytes + 64-byte alignment contract through ModelStore.
    /// </summary>
    public static unsafe void AppenderOffsetAlignment()
    {
        string path = Path.Combine(Path.GetTempPath(), $"qr_al_{Guid.NewGuid():N}.qw");
        try
        {
            static byte[] Pattern(long size, int seed)
            {
                var b = new byte[size];
                for (int i = 0; i < size; i++) b[i] = (byte)((i * 31 + seed) % 251 + 1);
                return b;
            }

            var cfg = new ModelConfig { NumLayers = 1, HiddenSize = 8, NumHeads = 1, VocabSize = 8 };
            using var cfgDoc = System.Text.Json.JsonDocument.Parse(cfg.ToJson());
            using (var b = new ModelStoreBuilder(path, cfgDoc.RootElement.Clone(), 1 << 20))
            {
                b.Plan("tok_emb", new long[] { 8, 8 }, QuantFormat.F16);   // 128 B
                b.Commit((p, s) => s.Write(Pattern(p.SizeBytes, 1)));
            }

            var expected = new Dictionary<string, byte[]>();
            int seed = 100;
            // wave 1: odd-sized tensors (12/32/8 B — none a multiple of 64) around an aligned one
            using (var app = new ModelStoreAppender(path))
            {
                foreach (var (name, dims, fmt) in new (string, long[], QuantFormat)[]
                {
                    ("l0.hc_attn_scale", new long[] { 3 }, QuantFormat.F32),      // 12 B
                    ("l0.hc_attn_base",  new long[] { 8 }, QuantFormat.F32),      // 32 B
                    ("l0.w_gate",        new long[] { 256, 128 }, QuantFormat.Q8_0),
                    ("l0.kda.alog",      new long[] { 2 }, QuantFormat.F32),      // 8 B
                })
                {
                    app.Plan(name, dims, fmt);
                    long n = 1; foreach (var d in dims) n *= d;
                    expected[name] = Pattern(Dtypes.TensorBytes(n, fmt), seed++);
                }
                app.Commit((p, s) => s.Write(expected[p.Name]));
            }

            // wave 2: chained append on top of an already-appended file
            using (var app = new ModelStoreAppender(path))
            {
                app.Plan("l1.hc_ffn_scale", new long[] { 3 }, QuantFormat.F32);
                expected["l1.hc_ffn_scale"] = Pattern(12, seed);
                app.Commit((p, s) => s.Write(expected[p.Name]));
            }

            using var store = ModelStore.Open(path);
            long baseOff = store.OffsetOf(store.Get("tok_emb"));
            foreach (var (name, want) in expected)
            {
                var t = store.Get(name);
                var got = t.Bytes.ToArray();
                if (got.Length != want.Length || !got.AsSpan().SequenceEqual(want))
                    throw new Exception($"tensor '{name}': data at its claimed offset does not match what was appended");
                if ((store.OffsetOf(t) - baseOff) % 64 != 0)
                    throw new Exception($"tensor '{name}': offset not 64-byte aligned relative to data start");
            }
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>Serializes tensors into the safetensors container format (8-byte header len + JSON + data).</summary>
    private static byte[] BuildSafeTensors(IEnumerable<(string Name, string Dtype, long[] Shape, byte[] Data)> tensors)
    {
        long off = 0;
        var parts = new List<string>();
        var data = new List<byte>();
        foreach (var (name, dtype, shape, bytes) in tensors)
        {
            var shapeJson = "[" + string.Join(",", shape) + "]";
            parts.Add($"\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shapeJson},\"data_offsets\":[{off},{off + bytes.Length}]}}");
            data.AddRange(bytes);
            off += bytes.Length;
        }
        string json = "{" + string.Join(",", parts) + "}";
        while (json.Length % 8 != 0) json += " ";   // safetensors headers are 8-byte aligned
        var header = System.Text.Encoding.UTF8.GetBytes(json);
        var buf = new byte[8 + header.Length + data.Count];
        BitConverter.GetBytes((long)header.Length).CopyTo(buf, 0);
        header.CopyTo(buf, 8);
        data.CopyTo(buf, 8 + header.Length);
        return buf;
    }

    public static unsafe void EngineDense() => RunEngine(moe: false);

    public static unsafe void EngineMoe() => RunEngine(moe: true);

    private static unsafe void RunEngine(bool moe)
    {
        string tmp = Path.Combine(Path.GetTempPath(), $"qr_eng_{Guid.NewGuid():N}.qw");
        int dim = 256, layers = 2, heads = 4, ffn = 512, vocab = 256, experts = moe ? 4 : 0;
        var rng = new Random(5);
        try
        {
            Synth.Write(tmp, dim, layers, heads, ffn, vocab, experts, moe ? 2 : 0);
            using var store = ModelStore.Open(tmp);
            using var eng = new StreamingEngine(store, new EngineOptions { PrefetchBudgetBytes = 16 << 20 });
            var logits0 = eng.Forward(3, 0)[..32].ToArray();   // copy: buffer reused next call
            CheckFinite(logits0);
            var logits1 = eng.Forward(9, 1)[..32].ToArray();
            CheckFinite(logits1);
            float d = 0;
            for (int i = 0; i < logits0.Length; i++) d += Math.Abs(logits0[i] - logits1[i]);
            if (d == 0) throw new Exception("logits identical across positions (suspicious)");
            var gen = eng.Generate(new[] { 3, 9 }, 4, new Sampler { Temperature = 0f });
            if (gen.Length == 0) throw new Exception("no tokens generated");
        }
        finally { if (File.Exists(tmp)) File.Delete(tmp); }
    }

    private static void CheckFinite(float[] v)
    {
        for (int i = 0; i < v.Length; i++)
            if (!float.IsFinite(v[i])) throw new Exception($"non-finite logits at {i}");
    }

    /// <summary>End-to-end GLM-5.x forward: synthesize a hybrid model (mHC + KDA:MLA 3:1 +
    /// GLM sigmoid-MoE + shared expert), then run the real engine over it. Verifies the engine
    /// picks the GLM-5 path, logits stay finite across positions, greedy generation is
    /// deterministic, ResetConversation restores the exact same sequence (KDA recurrent state +
    /// MLA latent cache reset), and a fresh instance reproduces it too.</summary>
    public static unsafe void EngineGlm5Hybrid()
    {
        string tmp = Path.Combine(Path.GetTempPath(), $"qr_g5_{Guid.NewGuid():N}.qw");
        try
        {
            Synth.WriteGlm5(tmp, 256, 4, 8, 512, 512, experts: 4, active: 2, hcMult: 2, seed: 11);
            using var store = ModelStore.Open(tmp);
            var cfg = ModelConfig.FromJson(store.ConfigJson);
            if (!cfg.IsGlm5 || !cfg.HasKda || !cfg.HasMla)
                throw new Exception("config did not describe a GLM-5 hybrid");
            if (cfg.AttnKinds[0] != "kda" || cfg.AttnKinds[3] != "mla")
                throw new Exception("unexpected attention pattern");
            using var eng = new StreamingEngine(store, new EngineOptions { PrefetchBudgetBytes = 0 });
            if (eng.Kv != null || eng.MlaCache == null)
                throw new Exception("engine did not select the GLM-5 path");

            int[] prompt = { 3, 1, 4, 1, 5, 9, 2, 6 };
            CheckFinite(eng.Forward(prompt[0], 0)[..32].ToArray());
            for (int p = 1; p < prompt.Length; p++)
                CheckFinite(eng.Forward(prompt[p], p)[..32].ToArray());

            // cold start (the manual forwards above warm the KDA state): reset, then generate
            eng.ResetConversation();
            int[] run1 = eng.Generate(prompt, 12, new Sampler { Temperature = 0f });
            if (run1.Length == 0) throw new Exception("no tokens generated");
            foreach (var t in run1)
                if ((uint)t >= (uint)cfg.VocabSize) throw new Exception($"token {t} out of range");

            // state reset: KDA recurrence + MLA cache must rewind to the exact same trajectory
            eng.ResetConversation();
            int[] run2 = eng.Generate(prompt, 12, new Sampler { Temperature = 0f });
            if (!run1.SequenceEqual(run2))
                throw new Exception("generation after ResetConversation diverged");

            // fresh instance reproduces the same sequence (no hidden cross-instance state)
            using (var store2 = ModelStore.Open(tmp))
            using (var eng2 = new StreamingEngine(store2, new EngineOptions { PrefetchBudgetBytes = 0 }))
            {
                int[] run3 = eng2.Generate(prompt, 12, new Sampler { Temperature = 0f });
                if (!run1.SequenceEqual(run3)) throw new Exception("fresh engine diverged");
                if (eng2.MlaCache!.SeqLen < prompt.Length)
                    throw new Exception("MLA latent cache did not fill");
                if (eng2.AttentionStateBytes <= eng2.MlaCache.BytesUsed)
                    throw new Exception("KDA recurrent state not resident");
            }
        }
        finally { if (File.Exists(tmp)) File.Delete(tmp); }
    }

    /// <summary>Greedy speculative decoding must produce EXACTLY the target's plain greedy
    /// trajectory — even with a completely unrelated draft model (mismatches, corrections, KDA
    /// snapshot/restore and cache rollbacks all get exercised).</summary>
    public static unsafe void SpeculativeGreedyEquivalence()
    {
        string tgt = Path.Combine(Path.GetTempPath(), $"qr_sp_t_{Guid.NewGuid():N}.qw");
        string drf = Path.Combine(Path.GetTempPath(), $"qr_sp_d_{Guid.NewGuid():N}.qw");
        try
        {
            Synth.Write(tgt, 256, 2, 4, 512, 512, 0, 0, seed: 3, fmt: QuantFormat.Q8_0);
            Synth.Write(drf, 128, 2, 4, 256, 512, 0, 0, seed: 99, fmt: QuantFormat.Q8_0);
            using var ts = ModelStore.Open(tgt);
            using var ds = ModelStore.Open(drf);
            using var target = new StreamingEngine(ts, new EngineOptions { PrefetchBudgetBytes = 0 });
            using var draft = new StreamingEngine(ds, new EngineOptions { PrefetchBudgetBytes = 0 });
            int[] prompt = { 5, 2, 7, 3 };

            int[] reference = target.Generate(prompt, 16, new Sampler { Temperature = 0f });
            if (reference.Length == 0) throw new Exception("reference generation empty");

            target.ResetConversation();
            var runner = new SpeculativeRunner(target, draft) { DraftTokens = 4 };
            int[] spec = runner.Generate(prompt, 16, new Sampler { Temperature = 0f });
            if (!reference.SequenceEqual(spec))
                throw new Exception($"speculative trajectory diverged: [{string.Join(',', reference)}] vs [{string.Join(',', spec)}]");
            if (runner.LastStats.Drafted == 0) throw new Exception("draft never proposed");
            if (runner.LastStats.Rounds == 0) throw new Exception("no speculative rounds ran");
        }
        finally { if (File.Exists(tgt)) File.Delete(tgt); if (File.Exists(drf)) File.Delete(drf); }
    }

    public static void BpeRoundTrip()
    {
        string tmp = Path.Combine(Path.GetTempPath(), $"qr_tok_{Guid.NewGuid():N}.json");
        try
        {
            File.WriteAllText(tmp, Synth.MiniTokenizerJson());
            var tok = BpeTokenizer.Load(tmp);
            // merges: h+e -> he, l+l -> ll   ("hello" -> he|ll|o)
            int[] ids = tok.Encode("hello");
            if (ids.Length != 3) throw new Exception($"expected 3 tokens [he,ll,o], got {string.Join(',', ids)}");
            string back = tok.Decode(ids);
            if (back != "hello") throw new Exception($"round trip gave '{back}'");
            string rt2 = tok.Decode(tok.Encode("Hello wörld 123 ??!\n  tabs\t"));
            if (rt2 != "Hello wörld 123 ??!\n  tabs\t") throw new Exception($"unicode round trip gave '{rt2}'");
        }
        finally { if (File.Exists(tmp)) File.Delete(tmp); }
    }

    public static void KvCache()
    {
        int kvDim = 32, layers = 1, headDim = 16, heads = 2, kvHeads = 2;
        var cache = new PagedKvCache(layers, heads, kvHeads, headDim, 600);
        var k = Enumerable.Range(0, kvDim).Select(i => (float)Math.Sin(i)).ToArray();
        var v = Enumerable.Range(0, kvDim).Select(i => (float)Math.Cos(i)).ToArray();
        for (int pos = 0; pos < 600; pos++) cache.Append(0, pos, k, v);
        if (cache.SeqLen != 600) throw new Exception("seqlen");
        if (cache.BytesUsed != 3L * PagedKvCache.PageTokens * kvDim * 4 * 2) throw new Exception($"bytes {cache.BytesUsed}");  // 600 tok -> 3 pages
        unsafe
        {
            var q = new float[heads * headDim];
            var attn = new float[heads * headDim];
            fixed (float* qp = q, ap = attn)
            {
                cache.Attention(0, 599, qp, ap, 1f / 4f);
                // q=0 -> all scores equal -> uniform softmax -> attn == mean(V) == V (constant V)
                for (int i = 0; i < attn.Length; i++)
                    if (Math.Abs(attn[i] - v[i % kvDim]) > 1e-4f)
                        throw new Exception($"zero-query attn[{i}]={attn[i]} != meanV {v[i % kvDim]}");
                for (int i = 0; i < q.Length; i++) q[i] = 1f;
                cache.Attention(0, 599, qp, ap, 1f / 4f);
                for (int i = 0; i < attn.Length; i++)
                    if (!float.IsFinite(attn[i])) throw new Exception("non-finite attention out");
                // uniform q over constant V => attn == V value of that head slot
                if (Math.Abs(attn[0] - v[0]) > 1e-4) throw new Exception($"attn[0]={attn[0]} expected {v[0]}");
            }
        }
    }

    public static void ParamBuckets()
    {
        // one representative per bucket per naming scheme: canonical .qw / GGUF / HuggingFace
        var cases = new (string Name, ParamReport.B Want)[]
        {
            // canonical .qw names
            ("tok_emb", ParamReport.B.Embedding), ("lm_head", ParamReport.B.Head),
            ("final_norm", ParamReport.B.Norms), ("l0.attn_norm", ParamReport.B.Norms),
            ("l0.wq", ParamReport.B.Attention), ("l0.wqkv", ParamReport.B.Attention),
            ("l5.mla.qa", ParamReport.B.Attention), ("l5.mla.qanorm", ParamReport.B.Norms),
            ("l6.kda.q", ParamReport.B.LinearAttn), ("l6.kda.onorm", ParamReport.B.Norms),
            ("l0.w_gate", ParamReport.B.Ffn), ("l0.router", ParamReport.B.Router),
            ("l0.e3.w_gate", ParamReport.B.RoutedExperts), ("l0.sw_gate", ParamReport.B.SharedExperts),
            ("l0.hc_attn_fn", ParamReport.B.Misc),
            // GGUF names
            ("token_embd.weight", ParamReport.B.Embedding), ("output.weight", ParamReport.B.Head),
            ("blk.0.attn_norm.weight", ParamReport.B.Norms), ("blk.0.attn_q.weight", ParamReport.B.Attention),
            ("blk.0.ffn_gate.weight", ParamReport.B.Ffn), ("blk.0.ffn_gate_inp.weight", ParamReport.B.Router),
            ("blk.0.ffn_gate_exps.weight", ParamReport.B.RoutedExperts),
            ("blk.0.ffn_gate_shexp.weight", ParamReport.B.SharedExperts),
            ("blk.0.output_norm.weight", ParamReport.B.Norms), ("blk.0.ffn_down.bias", ParamReport.B.Ffn),
            // HuggingFace names
            ("model.embed_tokens.weight", ParamReport.B.Embedding), ("lm_head.weight", ParamReport.B.Head),
            ("model.norm.weight", ParamReport.B.Norms),
            ("model.layers.0.input_layernorm.weight", ParamReport.B.Norms),
            ("model.layers.0.post_attention_layernorm.weight", ParamReport.B.Norms),
            ("model.layers.0.self_attn.q_proj.weight", ParamReport.B.Attention),
            ("model.layers.0.self_attn.q_a_proj.weight", ParamReport.B.Attention),
            ("model.layers.0.self_attn.kv_b_proj.weight", ParamReport.B.Attention),
            ("model.layers.0.self_attn.o_proj.weight", ParamReport.B.Attention),
            ("model.layers.0.mlp.gate_proj.weight", ParamReport.B.Ffn),
            ("model.layers.0.mlp.up_proj.weight", ParamReport.B.Ffn),
            ("model.layers.0.mlp.down_proj.weight", ParamReport.B.Ffn),
            ("model.layers.0.mlp.gate.weight", ParamReport.B.Router),
            ("model.layers.0.block_sparse_moe.gate.weight", ParamReport.B.Router),
            ("model.layers.0.mlp.gate.e_score_correction_bias", ParamReport.B.Router),
            ("model.layers.0.mlp.experts.3.gate_proj.weight", ParamReport.B.RoutedExperts),
            ("model.layers.0.mlp.shared_expert.gate_proj.weight", ParamReport.B.SharedExperts),
        };
        foreach (var (name, want) in cases)
        {
            var got = ParamReport.Bucket(name);
            if (got != want) throw new Exception($"Bucket('{name}') = {got}, want {want}");
        }

        // Count(): hand-checked totals
        var tensors = new (string, long[])[]
        {
            ("tok_emb", new long[] { 128, 64 }),   // 8192
            ("lm_head", new long[] { 128, 64 }),   // 8192
            ("l0.wq", new long[] { 64, 64 }),      // 4096
            ("l0.wo", new long[] { 64, 64 }),      // 4096
            ("l0.w_gate", new long[] { 172, 64 }), // 11008
            ("l0.attn_norm", new long[] { 64 }),   // 64
        };
        var s = ParamReport.Count(tensors);
        long total = 8192L * 2 + 4096L * 2 + 11008 + 64;   // 35648
        if (s.Total != total) throw new Exception($"Total {s.Total} != {total}");
        if (s.Embedding != 8192 || s.Head != 8192) throw new Exception("emb/head buckets wrong");
        if (s.Attention != 8192 || s.Ffn != 11008 || s.Norms != 64) throw new Exception("attn/ffn/norm buckets wrong");
        if (s.NonEmbedding != total - 16384) throw new Exception($"NonEmbedding {s.NonEmbedding}");
        if (ParamReport.LayersSeen(new[] { "l3.wq", "l3.wo", "l11.router", "tok_emb" }) != 2)
            throw new Exception("LayersSeen wrong");
        if (ParamReport.LayersSeen(new[] { "blk.3.attn_q", "blk.7.attn_v" }) != 2)
            throw new Exception("LayersSeen gguf wrong");
        if (ParamReport.LayersSeen(new[] { "model.layers.9.mlp.gate_proj.weight" }) != 1)
            throw new Exception("LayersSeen hf wrong");
    }

    public static void ParamMoeAndHuman()
    {
        // 100M toy model with a 64M routed-expert pool, 2 of 8 experts active per token
        var s = new ParamReport.Stats { Total = 100_000_000, RoutedExperts = 64_000_000 };
        if (ParamReport.ActivePerToken(s, 8, 2) != 52_000_000)
            throw new Exception($"active {ParamReport.ActivePerToken(s, 8, 2)} != 52,000,000");
        // dense guard: no experts -> everything is active
        if (ParamReport.ActivePerToken(new ParamReport.Stats { Total = 123 }, 0, 0) != 123)
            throw new Exception("dense active-per-token wrong");

        var human = new (string Want, long N)[]
        {
            ("671B", 671_032_832_512L), ("7.62B", 7_618_378_752L), ("410.5M", 410_523_648L),
            ("2.8T", 2_800_000_000_000L), ("1M", 1_000_000L), ("999", 999L),
        };
        foreach (var (want, n) in human)
        {
            string got = ParamReport.Human(n);
            if (got != want) throw new Exception($"Human({n}) = '{got}', want '{want}'");
        }
    }

    public static void GgufEmbeddedTokenizer()
    {
        // 1. mini GGUF KVs (the bytes a .qw header carries under gguf_kvs) -> parse -> build
        //    tokenizer.json -> load -> encode/decode round trip
        var cfg = new ModelConfig { HiddenSize = 64, NumLayers = 1, NumHeads = 4, FfnDim = 128, VocabSize = 300, MaxCtx = 512 };
        string b64 = System.Text.Json.Nodes.JsonNode.Parse(Synth.MiniGgufKvsJson(cfg))!["b64"]!.ToString();
        var kvs = GgufFile.ParseKvSection(Convert.FromBase64String(b64));
        if (kvs["tokenizer.ggml.tokens"] is not string[] toksArr || toksArr.Length != 258)
            throw new Exception($"tokens KVs wrong: {(kvs["tokenizer.ggml.tokens"] as string[])?.Length ?? -1}");
        if (kvs["tokenizer.ggml.merges"] is not string[] mrgArr || mrgArr.Length != 2)
            throw new Exception("merges KVs wrong");

        string json = HfExporter.BuildTokenizerJson(kvs) ?? throw new Exception("BPE tokenizer not built from KVs");
        var tok = BpeTokenizer.LoadJson(System.Text.Encoding.UTF8.GetBytes(json));
        // merges h+e, l+l — same split the file-based mini tokenizer gives
        int[] ids = tok.Encode("hello");
        if (ids.Length != 3) throw new Exception($"expected [he,ll,o], got {string.Join(',', ids)}");
        if (tok.Decode(ids) != "hello") throw new Exception($"round trip gave '{tok.Decode(ids)}'");
        if (tok.Decode(tok.Encode("Hello wörld 123 ??!\n")) != "Hello wörld 123 ??!\n")
            throw new Exception("unicode round trip failed");

        // 2. the same KVs embedded in an actual .qw header -> loaded without any external file
        string tmp = Path.Combine(Path.GetTempPath(), $"qr_emb_{Guid.NewGuid():N}.qw");
        try
        {
            Synth.Write(tmp, 64, 1, 4, 128, 300, 0, 0);
            using var store = ModelStore.Open(tmp);
            if (!store.ConfigJson.TryGetProperty("gguf_kvs", out _)) throw new Exception("synth .qw has no gguf_kvs");
            var tok2 = GgufRun.LoadTokenizerFromStore(store, out string reason);
            if (tok2 == null) throw new Exception($"embedded tokenizer not loaded: {reason}");
            if (tok2.Decode(tok2.Encode("hello")) != "hello") throw new Exception(".qw-embedded round trip failed");
        }
        finally { if (File.Exists(tmp)) File.Delete(tmp); }
    }

    public static unsafe void GgufRunFromGguf()
    {
        // the `run model.gguf` path: complete model -> gguf (tokenizer embedded) -> one-time
        // as-is .qw cache conversion -> embedded tokenizer -> engine generation -> cache reuse
        string dir = Path.Combine(Path.GetTempPath(), $"qr_grun_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        string qwOrig = Path.Combine(dir, "m.qw");
        string gguf = Path.Combine(dir, "m.gguf");
        try
        {
            Synth.Write(qwOrig, 256, 1, 4, 512, 300, 0, 0);          // dim 256 keeps Q4_K rows block-aligned
            GgufExporter.Export(qwOrig, gguf, new GgufExportOptions());

            var (cache, converted) = GgufRun.EnsureQwFromGguf(gguf);
            if (!converted || cache != gguf + ".qw") throw new Exception($"cache path/convert flag wrong: {cache}");
            var (_, reused) = GgufRun.EnsureQwFromGguf(gguf);
            if (reused) throw new Exception("cache not reused on the second run");

            byte[] wqOrig;
            using (var s0 = ModelStore.Open(qwOrig)) wqOrig = s0.Get("l0.wq").Bytes.ToArray();

            using var s1 = ModelStore.Open(cache);
            var wq = s1.Get("l0.wq");
            if (wq.Format != QuantFormat.Q4_K) throw new Exception($"l0.wq became {wq.Format} (as-is passthrough expected)");
            if (!wq.Bytes.ToArray().SequenceEqual(wqOrig)) throw new Exception("Q4_K weights not byte-exact through qw -> gguf -> qw");
            if (!s1.ConfigJson.TryGetProperty("gguf_kvs", out _)) throw new Exception("cache .qw lost the embedded gguf_kvs");

            var tok = GgufRun.LoadTokenizerFromStore(s1, out string reason);
            if (tok == null) throw new Exception($"embedded tokenizer: {reason}");
            if (tok.Decode(tok.Encode("hello world")) != "hello world") throw new Exception("encode/decode round trip");

            using var eng = new StreamingEngine(s1, new EngineOptions { PrefetchBudgetBytes = 8 << 20 });
            var gen = eng.Generate(tok.Encode("hello"), 4, new Sampler { Temperature = 0f });
            if (gen.Length != 4) throw new Exception($"generated {gen.Length} tokens");
            foreach (var id in gen) if (id is < 0 or >= 300) throw new Exception($"generated id {id} outside vocab");
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, true); }
    }
}
