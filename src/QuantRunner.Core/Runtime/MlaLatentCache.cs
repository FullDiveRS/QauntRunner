namespace QuantRunner.Core.Runtime;

/// <summary>
/// Paged cache for MLA (multi-latent attention) layers: stores the COMPRESSED kv latent
/// (kv_lora_rank [+ qk_rope_head_dim]) per token instead of expanded per-head K/V.
/// That is the entire point of MLA — e.g. GLM-5.3-Flash caches 512 floats per token per
/// layer where a 64-head GQA model would cache 16384. The absorbed-attention math that
/// consumes these rows lives in StreamingEngine (it needs the kv_b projection weights).
/// </summary>
public sealed unsafe class MlaLatentCache
{
    public const int PageTokens = PagedKvCache.PageTokens;

    private readonly int _latentDim;
    private readonly int _maxSeq;
    private readonly float[][][] _pages;   // [layer][page][PageTokens * latentDim]
    public int LatentDim => _latentDim;
    public int Layers { get; }
    public int SeqLen { get; private set; }
    public long BytesUsed { get; private set; }

    public MlaLatentCache(int layers, int latentDim, int maxSeq)
    {
        Layers = layers; _latentDim = latentDim; _maxSeq = maxSeq;
        _pages = new float[layers][][];
        for (int l = 0; l < layers; l++) _pages[l] = Array.Empty<float[]>();
    }

    private void Ensure(int layer, int tokens)
    {
        int pages = (tokens + PageTokens - 1) / PageTokens;
        var lp = _pages[layer];
        if (lp.Length >= pages) return;
        int newLen = Math.Max(pages, lp.Length * 2);
        var np = new float[newLen][];
        Array.Copy(lp, np, lp.Length);
        for (int i = lp.Length; i < pages; i++)
        {
            np[i] = new float[PageTokens * _latentDim];
            BytesUsed += (long)PageTokens * _latentDim * 4;
        }
        _pages[layer] = np;
    }

    public void Append(int layer, int pos, ReadOnlySpan<float> latent)
    {
        Ensure(layer, pos + 1);
        int off = (pos % PageTokens) * _latentDim;
        latent.CopyTo(_pages[layer][pos / PageTokens].AsSpan(off, _latentDim));
        if (pos + 1 > SeqLen) SeqLen = pos + 1;
    }

    public void Reset() => SeqLen = 0;

    /// <summary>Rewind the logical length (speculative decoding rollback). Pages keep stale
    /// bytes; every position is rewritten before it is read again.</summary>
    public void Truncate(int len) { if (len < SeqLen) SeqLen = len; }

    /// <summary>Number of pages currently held for a layer.</summary>
    public int PageCount(int layer) => _pages[layer].Length;

    /// <summary>Raw page contents [PageTokens * latentDim] for the attention loop.</summary>
    public float[] Page(int layer, int page) => _pages[layer][page];

    public int MaxSeq => _maxSeq;
}
