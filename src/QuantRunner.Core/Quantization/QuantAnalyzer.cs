using System.Buffers.Binary;
using System.Globalization;
using System.Text.Json;
using QuantRunner.Core.Importers;
using QuantRunner.Core.Models;

namespace QuantRunner.Core.Quantization;

/// <summary>A single format's measured error on one tensor.</summary>
public sealed record FormatQuality(
    QuantFormat Format, double Bpw, double RelL2, double SnrDb)
{
    public override string ToString() => $"{Format,-8} {Bpw,6:F3} bpw  relL2 {RelL2,8:F5}  snr {SnrDb,7:F2} dB";
}

/// <summary>All format results for one tensor (subsampled measurement).</summary>
public sealed record TensorQuality(string Name, long[] Dims, long TotalElements, long MeasuredElements,
    IReadOnlyList<FormatQuality> Results);

/// <summary>
/// Measures how much quality survives extreme-low-bit quantization, per tensor, without
/// running the model: every tensor is quantized to each candidate format in memory,
/// dequantized, and scored by relative Frobenius error ||W - Q||_F / ||W||_F and
/// SNR = 10*log10(||W||^2 / ||W-Q||^2). Weight-only L2 error is the standard proxy the
/// extreme-quantization literature reports alongside perplexity (BiLLM, OneBit, QuIP#...);
/// it ranks formats consistently and needs no calibration data.
///
/// Large tensors are subsampled by rows spread evenly across the tensor (rows keep block
/// alignment intact), so a 4.6B-parameter FFN shard scores in seconds.
/// </summary>
public static class QuantAnalyzer
{
    public static readonly QuantFormat[] DefaultFormats =
        { QuantFormat.Q4_K, QuantFormat.Q3_K, QuantFormat.TRI, QuantFormat.B1X, QuantFormat.T1E, QuantFormat.B1, QuantFormat.T1E_LO };

    public const long DefaultMaxElems = 1 << 20;   // per tensor measurement budget

    public static double BitsPerWeight(QuantFormat f)
    {
        var (bs, bb) = Dtypes.BlockSpec(f);
        return bb * 8.0 / bs;
    }

    /// <summary>Quantize -> dequantize -> score one buffer. tol returns relative L2 + SNR dB.</summary>
    public static (double relL2, double snrDb) Score(QuantFormat fmt, ReadOnlySpan<float> w)
    {
        int n = w.Length;
        var (bs, bb) = Dtypes.BlockSpec(fmt);
        if (n % bs != 0) throw new ArgumentException($"element count {n} not block-aligned for {fmt}");
        var q = new byte[n / bs * bb];
        BlockQuantizer.Quantize(fmt, w, q);
        var dq = new float[n];
        BlockDequantizer.Dequantize(fmt, q, dq);

        double se = 0, seRef = 0;
        for (int i = 0; i < n; i++)
        {
            double d = dq[i] - w[i];
            se += d * d;
            seRef += (double)w[i] * w[i];
        }
        double relL2 = Math.Sqrt(se / Math.Max(seRef, 1e-30));
        double snrDb = se > 0 ? 10.0 * Math.Log10(seRef / se) : 300.0;
        return (relL2, snrDb);
    }

    /// <summary>Score every format on a row-spread subsample of a 2-D (or 1-D) tensor.</summary>
    public static List<FormatQuality> AnalyzeTensor(string name, long[] dims, ReadOnlySpan<float> full,
        IReadOnlyList<QuantFormat> formats, long maxElems = DefaultMaxElems)
    {
        long total = 1; foreach (var d in dims) total *= d;
        ReadOnlySpan<float> sample = total <= maxElems ? full : Subsample(full, dims, maxElems, out long measured);
        var list = new List<FormatQuality>(formats.Count);
        foreach (var f in formats)
        {
            if (sample.Length % Dtypes.BlockSpec(f).blockSize != 0) continue;   // skip non-alignable sample
            var (relL2, snr) = Score(f, sample);
            list.Add(new FormatQuality(f, BitsPerWeight(f), relL2, snr));
        }
        _ = name;
        return list;
    }

    /// <summary>Pick up to maxElems elements as evenly spread whole rows (2-D) or one contiguous
    /// prefix-aligned chunk (1-D). Returns the sample; measured = element count.</summary>
    public static float[] Subsample(ReadOnlySpan<float> full, long[] dims, long maxElems, out long measured)
    {
        if (dims.Length >= 2)
        {
            long rows = dims[0], cols = dims[^1];
            long inner = 1; for (int i = 1; i < dims.Length - 1; i++) inner *= dims[i];
            long rowElems = inner * cols;
            long take = Math.Min(rows, Math.Max(1, maxElems / rowElems));
            var sample = new float[take * rowElems];
            for (long r = 0; r < take; r++)
            {
                long srcRow = rows > take ? r * rows / take : r;   // even spread incl. edges
                full.Slice((int)(srcRow * rowElems), (int)rowElems).CopyTo(sample.AsSpan((int)(r * rowElems)));
            }
            measured = sample.Length;
            return sample;
        }
        long n = Math.Min(full.Length, maxElems);
        var prefix = full[..(int)n].ToArray();
        measured = n;
        return prefix;
    }

    // ------------------------------------------------------------------ file driver

    /// <summary>Analyze every (mapped, block-alignable) tensor of a .safetensors file.</summary>
    public static List<TensorQuality> AnalyzeFile(string path, IReadOnlyList<QuantFormat> formats,
        long maxElems = DefaultMaxElems, long minElems = 4096, Action<string>? log = null)
    {
        using var st = SafeTensorsFile.Open(path);
        var results = new List<TensorQuality>();
        foreach (var (rawName, info) in st.Tensors)
        {
            long total = 1; foreach (var d in info.Shape) total *= d;
            if (total < minElems) continue;
            var mapped = NameMapper.Map(NameMapper.Source.HuggingFace, rawName, info.Shape, int.MaxValue, out _)
                         ?? rawName;
            long rowElems = info.Shape.Length >= 2 ? info.Shape[^1] : total;
            var buf = new float[Math.Min(total, Math.Max(maxElems, rowElems))];
            try
            {
                if (total <= buf.Length)
                {
                    st.ReadChunkF32(info, 0, buf.AsSpan(0, (int)total));
                    var dims = info.Shape.ToArray();
                    var res = AnalyzeTensor(mapped, dims, buf.AsSpan(0, (int)total), formats, maxElems);
                    results.Add(new TensorQuality(mapped, dims, total, total, res));
                }
                else
                {
                    var sample = SubsampleRowSpread(st, info, total, maxElems, out long measured);
                    var dims = new[] { measured / rowElems, rowElems };
                    var res = AnalyzeTensor(mapped, dims, sample, formats, maxElems);
                    results.Add(new TensorQuality(mapped, info.Shape.ToArray(), total, measured, res));
                }
            }
            catch (ArgumentException ex)
            {
                log?.Invoke($"  skip {mapped}: {ex.Message}");
            }
            if (log != null) log($"  measured {mapped} ({total:N0} elems)");
        }
        return results;
    }

    /// <summary>Row-spread subsample reading only the needed row ranges from disk.</summary>
    private static float[] SubsampleRowSpread(SafeTensorsFile st, SafeTensorsFile.TensorInfo info,
        long total, long maxElems, out long measured)
    {
        long rows = info.Shape[0];
        long rowElems = total / rows;
        long take = Math.Min(rows, Math.Max(1, maxElems / rowElems));
        var sample = new float[take * rowElems];
        var row = new float[rowElems];
        for (long r = 0; r < take; r++)
        {
            long srcRow = rows > take ? r * rows / take : r;
            st.ReadChunkF32(info, srcRow * rowElems, row, null);
            row.CopyTo(sample.AsSpan((int)(r * rowElems)));
        }
        measured = sample.Length;
        return sample;
    }

    // ------------------------------------------------------------------ output

    public static string ToJson(IEnumerable<TensorQuality> results)
    {
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = true }))
        {
            w.WriteStartObject();
            w.WritePropertyName("tensors");
            w.WriteStartArray();
            foreach (var t in results)
            {
                w.WriteStartObject();
                w.WriteString("name", t.Name);
                w.WriteString("dims", string.Join('x', t.Dims));
                w.WriteNumber("elements", t.TotalElements);
                w.WriteNumber("measured", t.MeasuredElements);
                w.WritePropertyName("formats");
                w.WriteStartArray();
                foreach (var r in t.Results)
                {
                    w.WriteStartObject();
                    w.WriteString("format", r.Format.ToString());
                    w.WriteNumber("bpw", Math.Round(r.Bpw, 4));
                    w.WriteNumber("rel_l2", double.Round(r.RelL2, 6));
                    w.WriteNumber("snr_db", double.Round(r.SnrDb, 2));
                    w.WriteEndObject();
                }
                w.WriteEndArray();
                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteEndObject();
        }
        return System.Text.Encoding.UTF8.GetString(ms.ToArray());
    }

    /// <summary>Console table: per-tensor rows then a per-format summary (mean / p90 rel-L2).</summary>
    public static void PrintReport(IReadOnlyList<TensorQuality> results, int top)
    {
        var formats = results.Count > 0 ? results[0].Results.Select(r => r.Format).ToList() : new List<QuantFormat>();

        Console.WriteLine();
        Console.WriteLine($"{"tensor",-40} {"elems",14}  " + string.Join("  ", formats.Select(f => $"{f,-8}")));
        Console.WriteLine(new string('-', 40 + 16 + formats.Count * 11));
        foreach (var t in results.Take(top))
        {
            var cells = formats.Select(f =>
            {
                var r = t.Results.FirstOrDefault(x => x.Format == f);
                return r == null ? "      -  " : $"{r.RelL2,8:F5}";
            });
            string elems = t.MeasuredElements == t.TotalElements ? $"{t.TotalElements:N0}" : $"{t.MeasuredElements:N0}/{t.TotalElements:N0}";
            Console.WriteLine($"{t.Name,-40} {elems,14}  " + string.Join("  ", cells));
        }
        if (results.Count > top) Console.WriteLine($"... and {results.Count - top} more tensors");

        Console.WriteLine();
        Console.WriteLine($"{"format",-10} {"bpw",8} {"mean relL2",12} {"p90 relL2",12} {"mean SNR dB",13} {"best tensor",-24}");
        foreach (var f in formats)
        {
            var errs = results.Select(t => t.Results.FirstOrDefault(r => r.Format == f))
                              .Where(r => r != null).Select(r => r!.RelL2).OrderBy(v => v).ToList();
            if (errs.Count == 0) continue;
            double mean = errs.Average();
            double p90 = errs[Math.Min(errs.Count - 1, (int)Math.Floor(errs.Count * 0.9))];
            double meanSnr = results.Select(t => t.Results.FirstOrDefault(r => r.Format == f))
                                    .Where(r => r != null).Average(r => r!.SnrDb);
            var best = results.OrderBy(t => t.Results.FirstOrDefault(r => r.Format == f)?.RelL2 ?? double.MaxValue).First();
            Console.WriteLine($"{f,-10} {BitsPerWeight(f),8:F3} {mean,12:F5} {p90,12:F5} {meanSnr,13:F2} {best.Name,-24}");
        }
    }
}
