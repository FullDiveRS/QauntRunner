using System.Diagnostics;
using System.Text.Json;
using QuantRunner.Core.Memory;
using QuantRunner.Core.Tokenizers;

namespace QuantRunner.Core.Importers;

/// <summary>
/// Support for running GGUF models directly. The engine runs from .qw (mmap + our kernels),
/// so `run model.gguf` converts once into a cached .qw beside the GGUF — with every tensor
/// kept in its own format (layout-identical ggml types copy byte-exact, so a Q4_K GGUF
/// produces a byte-identical Q4_K .qw) — and reuses that cache on every later run. The
/// tokenizer comes from inside the model: GGUF always carries tokenizer metadata, and a .qw
/// imported from GGUF embeds it (gguf_kvs) — no external tokenizer.json needed.
/// </summary>
public static class GgufRun
{
    /// <summary>Cache path for a runnable .qw converted from a GGUF: model.gguf -> model.gguf.qw.</summary>
    public static string CachePath(string ggufPath) => ggufPath + ".qw";

    /// <summary>Make sure a runnable .qw exists for <paramref name="ggufPath"/> (converts once,
    /// keeping the source formats; reuses the cache when it is newer than the GGUF). Returns the
    /// .qw path and whether a conversion happened.</summary>
    public static (string QwPath, bool Converted) EnsureQwFromGguf(string ggufPath)
    {
        string qwPath = CachePath(ggufPath);
        if (File.Exists(qwPath) && File.GetLastWriteTimeUtc(qwPath) >= File.GetLastWriteTimeUtc(ggufPath))
            return (qwPath, false);

        Console.Error.WriteLine($"  gguf       : converting once to a runnable .qw cache ({qwPath})");
        Console.Error.WriteLine("               formats are kept as-is — layout-identical tensors copy byte-exact");
        var sw = Stopwatch.StartNew();
        var opt = new ConvertOptions { KeepSourceFormats = true };
        var rep = ModelConverter.ConvertGguf(ggufPath, qwPath, opt);
        sw.Stop();
        Console.Error.WriteLine($"               {rep.TensorCount} tensors, {rep.TotalBytes / 1048576.0:F1} MiB " +
                                 $"in {sw.Elapsed.TotalSeconds:F1}s — later runs skip this step");
        return (qwPath, true);
    }

    /// <summary>Load the tokenizer embedded in a .qw header (gguf_kvs, stored when the .qw was
    /// imported from a GGUF) and build a byte-level BPE tokenizer from it. Returns null with a
    /// reason when there is no embedded tokenizer or it is sentencepiece (SPM vocabularies
    /// cannot be reconstructed — pass --tokenizer for those).</summary>
    public static BpeTokenizer? LoadTokenizerFromStore(ModelStore store, out string reason)
    {
        reason = "";
        if (!store.ConfigJson.TryGetProperty("gguf_kvs", out var kvsEl) ||
            kvsEl.ValueKind != JsonValueKind.Object ||
            !kvsEl.TryGetProperty("b64", out var b64El) || b64El.ValueKind != JsonValueKind.String)
        {
            reason = "no embedded tokenizer (this .qw was not imported from a GGUF)";
            return null;
        }

        byte[] kvBytes;
        try { kvBytes = Convert.FromBase64String(b64El.GetString()!); }
        catch (FormatException) { reason = "embedded gguf_kvs is corrupt (bad base64)"; return null; }

        Dictionary<string, object> kvs;
        try { kvs = GgufFile.ParseKvSection(kvBytes); }
        catch (Exception ex) { reason = $"embedded gguf_kvs could not be parsed: {ex.Message}"; return null; }

        string? tokenizerJson = HfExporter.BuildTokenizerJson(kvs);
        if (tokenizerJson == null)
        {
            string model = kvs.TryGetValue("tokenizer.ggml.model", out var m) && m is string ms ? ms : "(none)";
            reason = model is "gpt2" or "bpe"
                ? "embedded tokenizer has no vocabulary"
                : $"embedded tokenizer is '{model}' (sentencepiece) — its vocab file cannot be rebuilt, pass --tokenizer";
            return null;
        }

        try { return BpeTokenizer.LoadJson(System.Text.Encoding.UTF8.GetBytes(tokenizerJson)); }
        catch (Exception ex) { reason = $"embedded tokenizer failed to load: {ex.Message}"; return null; }
    }
}
