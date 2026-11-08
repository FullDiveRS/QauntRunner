using QuantRunner.Core.Kernels;

namespace QuantRunner.Core.Runtime;

/// <summary>
/// Paged KV cache: K/V are stored in fixed-size pages of <see cref="PageTokens"/> tokens that are
/// allocated on demand, so RAM grows with the *actual* sequence length instead of max context.
/// A 2.8T MoE with GQA therefore streams attention state the same way it streams weights.
/// Also owns the fused attention math: scores (dot of query against every cached K row),
/// softmax, and the weighted sum over V — all single-pass over the pages.
/// </summary>
public sealed unsafe class PagedKvCache
{
    public const int PageTokens = 256;

    private readonly int _kvDim;
    private readonly float[][][] _k;   // [layer][page][PageTokens * kvDim]
    private readonly float[][][] _v;
    private readonly float[] _scores;  // scratch [Heads * maxSeq] laid out per head
    private readonly int _maxSeq;

    public int Heads { get; }
    public int KvHeads { get; }
    public int HeadDim { get; }
    public int Layers { get; }
    public int SeqLen { get; private set; }
    public long BytesUsed { get; private set; }

    public PagedKvCache(int layers, int heads, int kvHeads, int headDim, int maxSeq)
    {
        Layers = layers; Heads = heads; KvHeads = kvHeads; HeadDim = headDim; _maxSeq = maxSeq;
        _kvDim = kvHeads * headDim;
        _k = new float[layers][][];
        _v = new float[layers][][];
        for (int l = 0; l < layers; l++) { _k[l] = Array.Empty<float[]>(); _v[l] = Array.Empty<float[]>(); }
        _scores = new float[(long)heads * Math.Max(maxSeq, 1)];
    }

    public long BytesPerToken => (long)_kvDim * 4 * 2 * Layers;

    private void Ensure(int layer, int tokens)
    {
        int pages = (tokens + PageTokens - 1) / PageTokens;
        var kp = _k[layer];
        if (kp.Length >= pages) return;

        int newLen = Math.Max(pages, kp.Length * 2);
        var nk = new float[newLen][];
        var nv = new float[newLen][];
        Array.Copy(kp, nk, kp.Length);
        Array.Copy(_v[layer], nv, _v[layer].Length);
        for (int i = kp.Length; i < pages; i++)
        {
            nk[i] = new float[PageTokens * _kvDim];
            nv[i] = new float[PageTokens * _kvDim];
            BytesUsed += PageTokens * _kvDim * 4L * 2;
        }
        _k[layer] = nk;
        _v[layer] = nv;
    }

    public void Append(int layer, int pos, ReadOnlySpan<float> k, ReadOnlySpan<float> v)
    {
        Ensure(layer, pos + 1);
        int off = (pos % PageTokens) * _kvDim;
        k.CopyTo(_k[layer][pos / PageTokens].AsSpan(off, _kvDim));
        v.CopyTo(_v[layer][pos / PageTokens].AsSpan(off, _kvDim));
        if (pos + 1 > SeqLen) SeqLen = pos + 1;
    }

    /// <summary>Reset sequence position between generation requests (pages are reused, not freed).</summary>
    public void Reset() => SeqLen = 0;

    /// <summary>Rewind the logical length (speculative decoding rollback). Pages keep stale
    /// bytes; every position is rewritten before it is read again.</summary>
    public void Truncate(int len) { if (len < SeqLen) SeqLen = len; }

    /// <summary>
    /// Full attention for one token at position <paramref name="pos"/>:
    /// scores = softmax(q·K^T * scale), attn = scores·V, written to <paramref name="attn"/>
    /// [Heads * HeadDim]. GQA mapping: query head h attends kv head (h * KvHeads / Heads).
    /// </summary>
    public void Attention(int layer, int pos, float* q, float* attn, float scale)
    {
        int headDim = HeadDim, kvDim = _kvDim, kvHeads = KvHeads, heads = Heads;
        int seq = pos + 1;

        fixed (float* scBase = _scores)
        {
            // ---- pass 1: scores[head, p] = dot(q_head, K[p, kvHead]) * scale ----
            int p = 0;
            var kPages = _k[layer];
            for (int page = 0; page < kPages.Length && p < seq; page++)
            {
                fixed (float* kp = kPages[page])
                {
                    int end = Math.Min(seq, (page + 1) * PageTokens);
                    for (; p < end; p++)
                    {
                        int kvOff = (p % PageTokens) * kvDim;
                        for (int h = 0; h < heads; h++)
                        {
                            int kvh = (int)((long)h * kvHeads / heads);
                            scBase[(long)h * _maxSeq + p] =
                                SimdMath.DotF32(q + (long)h * headDim, kp + kvOff + kvh * headDim, headDim) * scale;
                        }
                    }
                }
            }

            // ---- pass 2: softmax per head, then attn = sum_p w_p * V[p] ----
            var vPages = _v[layer];
            for (int h = 0; h < heads; h++)
            {
                float* sc = scBase + (long)h * _maxSeq;
                TransformerOps.Softmax(sc, seq);
                float* dst = attn + (long)h * headDim;
                for (int i = 0; i < headDim; i++) dst[i] = 0;

                int kvh = (int)((long)h * kvHeads / heads);
                p = 0;
                for (int page = 0; page < vPages.Length && p < seq; page++)
                {
                    fixed (float* vp = vPages[page])
                    {
                        int end = Math.Min(seq, (page + 1) * PageTokens);
                        for (; p < end; p++)
                        {
                            int kvOff = (p % PageTokens) * kvDim + kvh * headDim;
                            SimdMath.AxpyF32(vp + kvOff, dst, sc[p], headDim);
                        }
                    }
                }
            }
        }
    }
}
