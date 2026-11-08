using System.Text.Json;
using QuantRunner.Core.Memory;
using QuantRunner.Core.Quantization;

namespace QuantRunner.Core.Importers;

/// <summary>
/// Options for requantizing an existing .qw into a smaller .qw.
/// Per-group targets follow the same routing as quantize (--fmt/--qattn/--qffn/--qemb/--qhead);
/// a null group target keeps the source format of that group untouched.
/// </summary>
public sealed class RequantOptions
{
    public QuantFormat? Attention;                 // wq/wk/wv/wo + KDA q/k/v/o + MLA qa/qb/kva/kvb/o
    public QuantFormat? Ffn;                       // w_gate/w_up/w_down + shared experts
    public QuantFormat? Emb;                       // tok_emb
    public QuantFormat? Head;                      // lm_head
    public long MinQuantElements = 4096;           // tensors smaller stay as-is
    public bool DryRun;                            // compute the plan, write nothing

    public List<string>? Exclude;                  // canonical names never touched
    public List<string>? Only;                     // requant only these names

    // ---- smart mode: per-tensor formats from an `analyze` report ----
    public string? FromReport;                     // report.json produced by `analyze -o report.json`
    public double MaxRelL2 = 0.35;                 // quality gate: smallest-bpw format whose relL2 fits
}

/// <summary>One tensor's requant plan row.</summary>
public sealed record RequantPlanRow(
    string Name, long[] Dims, QuantFormat From, QuantFormat To, string Action, string? Reason)
{
    public long FromBytes => Dtypes.TensorBytes(NumElements, From);
    public long ToBytes => Dtypes.TensorBytes(NumElements, To);
    public long NumElements { get { long n = 1; foreach (var d in Dims) n *= d; return n; } }
}

public sealed record RequantResult(
    IReadOnlyList<RequantPlanRow> Rows, long SourceBytes, long OutputBytes, double Seconds)
{
    public int Requantized => Rows.Count(r => r.Action == "requant");
    public int Passthrough => Rows.Count(r => r.Action == "passthrough");
    public int Kept => Rows.Count(r => r.Action == "keep");
}

/// <summary>
/// Requantizes an existing .qw model into a smaller .qw — "quantize the quantized model".
/// Every selected tensor is dequantized block by block and re-encoded with a smaller
/// target format in one streaming pass; the source is never materialized, so a 2.8T
/// model requantizes in RAM-similar space to quantizing it the first time.
///
/// What is preserved:
///   - the header config is copied verbatim (same arch, same tensor set, same names);
///   - tensors whose target format equals their source format are copied BYTE-EXACT
///     (passthrough — a same-format round trip would add nothing but rounding noise);
///   - norms/router/sensitive groups (KDA gate/beta, mHC fn) are never touched;
///   - 1-d tensors and sub-min-elems tensors are never touched;
///   - a target format whose block size does not divide the row length is skipped with
///     a note (the original row-aligned format beats a forced q8_0/f32 fallback).
///
/// Smart mode (--from-report): `analyze` measures per-tensor relL2 for a ladder of
/// formats on the real checkpoint weights. Requant picks, per tensor, the SMALLEST-bpw
/// format whose relL2 fits the --max-rell2 gate and is smaller than the source —
/// so quality spend goes exactly where the weights can afford it (heavy-tail FFN
/// tensors often keep B1X; attention usually stops at TRI/Q4_K).
/// </summary>
public static class Requantizer
{
    public static RequantResult Run(string srcPath, string dstPath, RequantOptions opt)
    {
        if (!File.Exists(srcPath)) throw new FileNotFoundException($"source .qw not found: {srcPath}");
        if (!opt.DryRun && File.Exists(dstPath))
            throw new IOException($"{dstPath} already exists — delete it or pick another -o");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var store = ModelStore.Open(srcPath);

        var report = opt.FromReport != null ? LoadReport(opt.FromReport) : null;
        var router = new ConvertOptions();                    // group routing only — Pick is never used
        var notes = new List<string>();

        // ---- plan every tensor ----
        var rows = new List<RequantPlanRow>(store.Tensors.Count);
        foreach (var t in store.Tensors.Values)
        {
            var (fmt, action, reason) = Decide(t, opt, router, report, notes);
            rows.Add(new RequantPlanRow(t.Name, t.Dims, t.Format, fmt, action, reason));
        }
        long srcBytes = rows.Sum(r => r.FromBytes);
        long dstBytes = rows.Sum(r => r.ToBytes);

        // ---- write ----
        if (!opt.DryRun)
        {
            var cfg = JsonDocument.Parse(store.ConfigJson.GetRawText()).RootElement.Clone();
            using var builder = new ModelStoreBuilder(dstPath, cfg);
            foreach (var r in rows) builder.Plan(r.Name, r.Dims, r.To);
            builder.Commit((plan, stream) => WriteTensor(store.Get(plan.Name), plan.Format, stream));
            foreach (var n in notes) Console.WriteLine($"  note     : {n}");
        }
        sw.Stop();
        return new RequantResult(rows, srcBytes, dstBytes, sw.Elapsed.TotalSeconds);
    }

    // ------------------------------------------------------------------ policy

    private static (QuantFormat To, string Action, string? Reason) Decide(
        TensorView t, RequantOptions opt, ConvertOptions router,
        Dictionary<string, List<(QuantFormat f, double bpw, double relL2)>>? report, List<string> notes)
    {
        string name = t.Name;
        if (opt.Only is { Count: > 0 } && !opt.Only.Contains(name)) return (t.Format, "keep", "not in --only");
        if (opt.Exclude is { Count: > 0 } && opt.Exclude.Contains(name)) return (t.Format, "keep", "excluded");

        // smart mode: the report decides for tensors it measured (never grows the file)
        if (report != null && report.TryGetValue(name, out var cands))
        {
            double srcBpw = Bpw(t.Format);
            var pick = cands.Where(c => c.relL2 <= opt.MaxRelL2 && c.bpw < srcBpw && RowSafe(c.f, t))
                            .OrderBy(c => c.bpw)
                            .FirstOrDefault();
            if (pick.f != default)
                return (pick.f, "requant", $"report: relL2 {pick.relL2:F3} @ {pick.bpw:F2} bpw");
            return (t.Format, "keep", "no reported format fits the quality gate");
        }

        var g = router.Group(name);
        QuantFormat? target = g switch
        {
            ConvertOptions.QuantGroup.Attention => opt.Attention,
            ConvertOptions.QuantGroup.Ffn => opt.Ffn,
            ConvertOptions.QuantGroup.Emb => opt.Emb,
            ConvertOptions.QuantGroup.Head => opt.Head,
            _ => null,                                          // Norm / Sensitive: never touched
        };

        if (target == null) return (t.Format, "keep", g is ConvertOptions.QuantGroup.Norm or ConvertOptions.QuantGroup.Sensitive ? "protected group" : null);
        QuantFormat dst = target.Value;
        if (dst == t.Format) return (t.Format, "passthrough", null);
        if (!dst.IsQuantized() && t.Format.IsFp()) return (t.Format, "keep", "fp to fp");   // no win
        if (t.NumElements < opt.MinQuantElements) return (t.Format, "keep", $"small ({t.NumElements} elems)");
        if (t.Dims.Length < 2) return (t.Format, "keep", "1-d tensor");
        if (!RowSafe(dst, t))
        {
            notes.Add($"{name}: {dst} block does not divide {t.Dims[^1]} columns — kept {t.Format}");
            return (t.Format, "keep", "row alignment");
        }
        return (dst, "requant", null);
    }

    private static bool RowSafe(QuantFormat f, TensorView t)
    {
        if (t.Dims.Length < 2) return false;
        return t.Dims[^1] % Dtypes.BlockSpec(f).blockSize == 0;
    }

    private static double Bpw(QuantFormat f)
    {
        var (bs, bb) = Dtypes.BlockSpec(f);
        return bb * 8.0 / bs;
    }

    // ------------------------------------------------------------------ report parsing

    private static Dictionary<string, List<(QuantFormat, double, double)>> LoadReport(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var map = new Dictionary<string, List<(QuantFormat, double, double)>>();
        foreach (var te in doc.RootElement.GetProperty("tensors").EnumerateArray())
        {
            string name = te.GetProperty("name").GetString() ?? "";
            var list = new List<(QuantFormat, double, double)>();
            foreach (var fe in te.GetProperty("formats").EnumerateArray())
            {
                try
                {
                    var f = ConvertOptions.ParseQuant(fe.GetProperty("format").GetString()!.ToLowerInvariant());
                    list.Add((f, fe.GetProperty("bpw").GetDouble(), fe.GetProperty("rel_l2").GetDouble()));
                }
                catch (ArgumentException) { /* unknown format name in report — skip it */ }
            }
            map[name] = list;
        }
        return map;
    }

    // ------------------------------------------------------------------ streaming writer

    /// <summary>Byte-exact passthrough for kept tensors; dequant → requant in 1M-element
    /// chunks (a multiple of every block size: 32 / 64 / 256) for downgraded ones.</summary>
    private static unsafe void WriteTensor(TensorView t, QuantFormat dst, Stream s)
    {
        if (dst == t.Format)
        {
            long n = t.SizeBytes, done = 0;
            var buf = new byte[1 << 20];
            while (done < n)
            {
                int take = (int)Math.Min(buf.Length, n - done);
                new ReadOnlySpan<byte>(t.Ptr + done, take).CopyTo(buf);
                s.Write(buf, 0, take);
                done += take;
            }
            return;
        }

        var (sbs, sbb) = Dtypes.BlockSpec(t.Format);
        long elems = t.NumElements;
        const long CHUNK = 1 << 20;
        var f32 = new float[CHUNK];
        var srcBuf = new byte[CHUNK / sbs * sbb];
        var dstBuf = new byte[Dtypes.TensorBytes(CHUNK, dst)];
        long doneE = 0;
        while (doneE < elems)
        {
            long take = Math.Min(CHUNK, elems - doneE);
            int sb = (int)(take / sbs * sbb);
            new ReadOnlySpan<byte>(t.Ptr + doneE / sbs * sbb, sb).CopyTo(srcBuf);
            BlockDequantizer.Dequantize(t.Format, srcBuf.AsSpan(0, sb), f32.AsSpan(0, (int)take));
            long ob = Dtypes.TensorBytes(take, dst);
            BlockQuantizer.Quantize(dst, f32.AsSpan(0, (int)take), dstBuf.AsSpan(0, (int)ob));
            s.Write(dstBuf, 0, (int)ob);
            doneE += take;
        }
    }
}
