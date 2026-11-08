using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using QuantRunner.Core.Kernels;
using QuantRunner.Core.Memory;
using QuantRunner.Core.Models;
using QuantRunner.Core.Quantization;

namespace QuantRunner.Core.Runtime;

/// <summary>Engine tuning knobs.</summary>
public sealed class EngineOptions
{
    /// <summary>Per-layer staging budget for the async prefetch pipeline. 0 disables prefetch (pure mmap).</summary>
    public long PrefetchBudgetBytes { get; set; } = 1L << 30;
    /// <summary>Hard context cap (KV pages are never allocated beyond this).</summary>
    public int MaxCtx { get; set; } = -1;
}

public sealed record GenerationStats(
    int PromptTokens, int GeneratedTokens, double PromptSeconds, double GenSeconds,
    double TokensPerSecond, long PrefetchBytes, double PrefetchWaitMs, long KvBytes);

/// <summary>
/// Streaming token engine: runs any Llama-family (dense or Mixtral-style MoE) model from a
/// .qw store one layer at a time. The full model is NEVER resident — weights for layer N+1
/// are prefetched from disk on a background task while layer N computes; activations, KV
/// pages and at most one layer of weights live in RAM. Same memory profile at 3B and at 2.8T.
/// </summary>
public sealed unsafe class StreamingEngine : IDisposable
{
    public ModelConfig Config { get; }
    public ModelStore Store { get; }
    public PagedKvCache? Kv { get; }                 // legacy GQA path (null on the GLM-5.x path)
    public MlaLatentCache? MlaCache { get; }         // GLM-5.x path: compressed kv-latent pages
    public PrefetchPipeline? Prefetch { get; }

    private readonly int _dim, _nQ, _nK, _headDim, _ffn, _expFfn, _vocab;
    private readonly float[] _x, _xn, _q, _k, _v, _attn, _ffnG, _ffnU, _logits, _normW, _router;
    private readonly float[] _cosSin;
    private readonly int _maxCtx;

    // ---- GLM-5.x working set (mHC streams + KDA/MLA scratch) ----
    private float[]? _streams, _hcFlat, _hcW, _hcBase, _hcScale, _hcPost, _hcComb,
        _collapsed, _blockOut;
    private float[]? _kdaMixed, _kdaG, _kdaBeta, _kdaGate, _kdaOut, _kdaTmp;
    private float[]? _mlaQa, _mlaQ, _mlaCkv, _mlaQtilde, _mlaU, _mlaScores, _mlaRow, _mlaAttn;
    private readonly float[]? _routerBias;
    private KdaLayerState[]? _kdaStates;
    private float[]? _kdaSnap;                       // KDA state snapshot (speculative rollback)
    private int[] _kdaSlot = Array.Empty<int>();     // layer -> kda-state slot (-1 = not kda)

    private sealed class KdaLayerState
    {
        public bool Loaded;                            // constant weights dequanted once
        public float[] ConvW = Array.Empty<float>();   // [3*qkvDim * K] dequanted conv taps
        public float[] Dt = Array.Empty<float>();      // [qkvDim] dt_bias
        public float[] ALog = Array.Empty<float>();    // [heads] A_log
        public float[] ONorm = Array.Empty<float>();   // [headDim] o_norm weight
        public float[] Conv = Array.Empty<float>();    // [(K-1) * 3*qkvDim] ring windows
        public float[] S = Array.Empty<float>();       // [heads * dk * dv] recurrent state
    }

    /// <summary>Resolved weight reference: pointer into mmap or the prefetch staging buffer.</summary>
    private readonly struct WRef
    {
        public readonly byte* Ptr;
        public readonly QuantFormat Fmt;
        public WRef(byte* p, QuantFormat f) { Ptr = p; Fmt = f; }
    }

    public StreamingEngine(ModelStore store, EngineOptions? opt = null)
    {
        Store = store;
        var o = opt ?? new EngineOptions();
        Config = ModelConfig.FromJson(store.ConfigJson);

        _dim = Config.HiddenSize;
        _headDim = Config.EffectiveHeadDim;
        _nQ = Config.NumHeads * _headDim;
        _nK = Config.EffectiveKvHeads * _headDim;
        _ffn = (int)Config.FfnDim;
        _expFfn = (int)Config.EffectiveExpertFfn;
        _vocab = Config.VocabSize;
        _maxCtx = o.MaxCtx > 0 ? o.MaxCtx : Math.Min(Config.MaxCtx, 32768);

        bool glm5 = Config.IsGlm5;
        if (!glm5)
        {
            Kv = new PagedKvCache(Config.NumLayers, Config.NumHeads, Config.EffectiveKvHeads, _headDim, _maxCtx);
            MlaCache = null;
        }
        else
        {
            Kv = null;
            int latentDim = Config.KvLoraRank + Config.QkRopeHeadDim;
            MlaCache = Config.HasMla ? new MlaLatentCache(Config.NumLayers, latentDim, _maxCtx) : null;
        }
        if (o.PrefetchBudgetBytes > 0)
            Prefetch = new PrefetchPipeline(store, o.PrefetchBudgetBytes);

        long ffnMax = Math.Max(_ffn, Config.IsMoe ? _expFfn : 0);
        _x = new float[_dim];
        _xn = new float[_dim];
        _q = new float[_nQ];
        _k = new float[_nK];
        _v = new float[_nK];
        _attn = new float[_nQ];
        _ffnG = new float[ffnMax];
        _ffnU = new float[ffnMax];
        _logits = new float[_vocab];
        _normW = new float[_dim];
        _router = new float[Math.Max(1, Config.NumExperts)];
        _cosSin = new float[Math.Max(_headDim, 1)];
        _routerBias = glm5 && Config.MoeSigmoid && Config.NumExperts > 0
            ? new float[Config.NumExperts] : null;

        if (glm5) InitGlm5Buffers();
    }

    private void InitGlm5Buffers()
    {
        int n = Math.Max(1, Config.HcMult);
        int dim = _dim;
        int hcW = (2 + n) * n;
        _streams = new float[n * dim];
        _hcFlat = new float[n * dim];
        _hcW = new float[hcW];
        _hcBase = new float[hcW];
        _hcScale = new float[3];
        _hcPost = new float[n];
        _hcComb = new float[n * n];
        _collapsed = new float[dim];
        _blockOut = new float[dim];

        if (Config.HasKda)
        {
            int qkv = Config.LinearHeads * Config.LinearHeadDim;
            _kdaMixed = new float[3 * qkv];
            _kdaG = new float[qkv];
            _kdaBeta = new float[Config.LinearHeads];
            _kdaGate = new float[qkv];
            _kdaOut = new float[qkv];
            _kdaTmp = new float[Math.Max(qkv, Config.LinearHeadDim)];

            _kdaSlot = new int[Config.NumLayers];
            var states = new List<KdaLayerState>();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (Config.AttnKind(l) != "kda") { _kdaSlot[l] = -1; continue; }
                _kdaSlot[l] = states.Count;
                int kernel = Math.Max(1, Config.LinearConvKernel);
                states.Add(new KdaLayerState
                {
                    ConvW = new float[3 * qkv * kernel],
                    Dt = new float[qkv],
                    ALog = new float[Config.LinearHeads],
                    ONorm = new float[Config.LinearHeadDim],
                    Conv = new float[(kernel - 1) * 3 * qkv],
                    S = new float[(long)Config.LinearHeads * Config.LinearHeadDim * Config.LinearHeadDim]
                });
            }
            _kdaStates = states.ToArray();
        }

        if (Config.HasMla)
        {
            int qkDim = Config.QkNopeHeadDim + Config.QkRopeHeadDim;
            _mlaQa = new float[Config.QLoraRank];
            _mlaQ = new float[(long)Config.NumHeads * qkDim];
            _mlaCkv = new float[Config.KvLoraRank + Config.QkRopeHeadDim];
            _mlaQtilde = new float[(long)Config.NumHeads * Config.KvLoraRank];
            _mlaU = new float[Config.KvLoraRank];
            _mlaScores = new float[(long)Config.NumHeads * Math.Max(_maxCtx, 1)];
            _mlaRow = new float[Config.KvLoraRank];
            _mlaAttn = new float[(long)Config.NumHeads * Config.VHeadDim];
        }
    }

    // ---------------------------------------------------------------- tensor helpers

    private TensorView T(string name) => Store.Get(name);

    /// <summary>Resolve tensor pointer — waits on the prefetch pipeline when the layer is staged.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private WRef W(string name, int layer)
    {
        var t = Store.Get(name);
        byte* p = t.Ptr;
        if (Prefetch != null) p = Prefetch.Acquire(name, layer);
        return new WRef(p, t.Format);
    }

    /// <summary>Dequantize a whole 1-D tensor (norm weights, router) into dst.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Dequant1D(string name, float* dst)
    {
        var t = Store.Get(name);
        BlockDequantizer.Dequantize(t.Format, new ReadOnlySpan<byte>(t.Ptr, (int)t.SizeBytes),
            new Span<float>(dst, (int)t.NumElements));
    }

    // ---------------------------------------------------------------- forward

    /// <summary>Run one token at position <paramref name="pos"/>; the returned logits are
    /// valid only until the next Forward/Generate call.</summary>
    public float[] Forward(int token, int pos)
    {
        if (pos >= _maxCtx) throw new InvalidOperationException($"position {pos} exceeds context {_maxCtx}");
        return Config.IsGlm5 ? ForwardGlm5(token, pos) : ForwardLegacy(token, pos);
    }

    private unsafe float[] ForwardLegacy(int token, int pos)
    {
        var kv = Kv!;
        fixed (float* px = _x, pxn = _xn, pq = _q, pk = _k, pv = _v, pa = _attn,
               pg = _ffnG, pu = _ffnU, plo = _logits, pnw = _normW, pr = _router, pcs = _cosSin)
        {
            // ---- embedding row (streams straight off the mmap) ----
            var emb = Store.Get("tok_emb");
            BlockDequantizer.Dequantize(emb.Format, new ReadOnlySpan<byte>(emb.RowPtr(token), (int)emb.RowBytes()),
                new Span<float>(px, _dim));

            // kick off prefetch for layer 0 (no-op when disabled)
            Prefetch?.Begin(0);

            float eps = Config.RmsEps;
            int layers = Config.NumLayers;

            for (int l = 0; l < layers; l++)
            {
                // ---- attention block ----
                TransformerOps.RmsNorm(px, null, pxn, _dim, eps);
                Dequant1D($"l{l}.attn_norm", pnw);
                TransformerOps.RmsNorm(px, pnw, pxn, _dim, eps);

                var wq = W($"l{l}.wq", l);
                var wk = W($"l{l}.wk", l);
                var wv = W($"l{l}.wv", l);
                DotKernels.MatVecParallel(wq.Fmt, wq.Ptr, _nQ, _dim, pxn, pq);
                DotKernels.MatVecParallel(wk.Fmt, wk.Ptr, _nK, _dim, pxn, pk);
                DotKernels.MatVecParallel(wv.Fmt, wv.Ptr, _nK, _dim, pxn, pv);

                // RoPE (cos/sin row for this position, shared by all heads)
                TransformerOps.BuildCosSinInto(pos, _headDim, Config.RopeTheta, _cosSin);
                TransformerOps.Rope(pq, pk, pcs, _nQ, _nK, _headDim);

                kv.Append(l, pos, new ReadOnlySpan<float>(pk, _nK), new ReadOnlySpan<float>(pv, _nK));

                float scale = 1f / MathF.Sqrt(_headDim);
                kv.Attention(l, pos, pq, pa, scale);

                // out projection + residual
                var wo = W($"l{l}.wo", l);
                DotKernels.MatVecParallel(wo.Fmt, wo.Ptr, _dim, _nQ, pa, pxn);
                SimdMath.AxpyF32(pxn, px, 1f, _dim);

                // ---- FFN block ----
                Dequant1D($"l{l}.ffn_norm", pnw);
                TransformerOps.RmsNorm(px, pnw, pxn, _dim, eps);

                if (Config.IsMoe) MoeFfn(l, pxn, px, pg, pu, pr);
                else DenseFfn(l, pxn, px, pg, pu);

                // start streaming the next layer while this one finishes computing
                if (l + 1 < layers) Prefetch?.Begin(l + 1);
            }

            // ---- head ----
            Dequant1D("final_norm", pnw);
            TransformerOps.RmsNorm(px, pnw, pxn, _dim, eps);

            if (Store.Has("lm_head"))
            {
                var wh = W("lm_head", -1);
                DotKernels.MatVecParallel(wh.Fmt, wh.Ptr, _vocab, _dim, pxn, plo);
            }
            else
            {
                var te = Store.Get("tok_emb");   // tied embeddings
                DotKernels.MatVecParallel(te.Format, te.Ptr, _vocab, _dim, pxn, plo);
            }

            return _logits;
        }
    }

    // ================================================================ GLM-5.x forward
    // (mHC hyper-connections + KDA linear attention + MLA/DSA attention + GLM-MoE)

    private unsafe float[] ForwardGlm5(int token, int pos)
    {
        int n = Math.Max(1, Config.HcMult), dim = _dim;
        fixed (float* px = _x, pxn = _xn, plo = _logits, pnw = _normW,
               pstreams = _streams, pflat = _hcFlat, pw = _hcW, pbase = _hcBase, pscale = _hcScale,
               ppost = _hcPost, pcomb = _hcComb, pcollapsed = _collapsed, pout = _blockOut)
        {
            // ---- embedding row, replicated across the mHC streams ----
            var emb = Store.Get("tok_emb");
            BlockDequantizer.Dequantize(emb.Format, new ReadOnlySpan<byte>(emb.RowPtr(token), (int)emb.RowBytes()),
                new Span<float>(px, dim));
            for (int i = 0; i < n; i++) SimdMath.CopyF32(px, pstreams + i * dim, dim);
            Prefetch?.Begin(0);
            for (int l = 0; l < Config.NumLayers; l++)
            {
                // ---- attention sub-block (wrapped in mHC) ----
                HcStep($"l{l}.hc_attn", l, pstreams, pcollapsed, ppost, pcomb);
                Dequant1D($"l{l}.attn_norm", pnw);
                TransformerOps.RmsNorm(pcollapsed, pnw, pxn, dim, Config.RmsEps);

                string kind = Config.AttnKind(l);
                if (kind == "kda") KdaLayer(l, pxn, pos, pout);
                else if (kind == "mla") MlaLayer(l, pxn, pos, pout);
                else throw new NotSupportedException(
                    $"layer {l}: GQA attention inside a GLM-5.x (mHC) model is not supported");
                Glm5Ops.HcCombine(pstreams, pout, ppost, pcomb, n, dim, pflat);

                // ---- FFN sub-block (wrapped in mHC) ----
                HcStep($"l{l}.hc_ffn", l, pstreams, pcollapsed, ppost, pcomb);
                Dequant1D($"l{l}.ffn_norm", pnw);
                TransformerOps.RmsNorm(pcollapsed, pnw, pxn, dim, Config.RmsEps);

                if (Store.Has($"l{l}.router"))
                {
                    fixed (float* pg = _ffnG, pu = _ffnU, pr = _router)
                    {
                        if (Config.MoeSigmoid) GlmMoeFfn(l, pxn, pout, pr, pg, pu);
                        else LegacyMoeInto(l, pxn, pout, pr, pg, pu);
                    }
                }
                else
                {
                    fixed (float* pg = _ffnG, pu = _ffnU)
                        DenseFfnOut(l, pxn, pout, pg, pu);
                }
                Glm5Ops.HcCombine(pstreams, pout, ppost, pcomb, n, dim, pflat);

                if (l + 1 < Config.NumLayers) Prefetch?.Begin(l + 1);
            }

            // ---- head: unweighted mean over streams, final norm, lm_head ----
            for (int d = 0; d < dim; d++) px[d] = 0;
            for (int i = 0; i < n; i++) SimdMath.AxpyF32(pstreams + i * dim, px, 1f / n, dim);
            Dequant1D("final_norm", pnw);
            TransformerOps.RmsNorm(px, pnw, pxn, dim, Config.RmsEps);

            if (Store.Has("lm_head"))
            {
                var wh = W("lm_head", -1);
                DotKernels.MatVecParallel(wh.Fmt, wh.Ptr, _vocab, dim, pxn, plo);
            }
            else
            {
                var te = Store.Get("tok_emb");
                DotKernels.MatVecParallel(te.Format, te.Ptr, _vocab, dim, pxn, plo);
            }
            return _logits;
        }
    }

    /// <summary>mHC block entry: computes post/comb from the current streams and collapses
    /// them into the block input (Glm5NextTextHyperConnection.forward).</summary>
    private unsafe void HcStep(string prefix, int layer, float* streams, float* collapsed, float* post, float* comb)
    {
        int n = Math.Max(1, Config.HcMult), dim = _dim, nd = n * dim;
        fixed (float* pflat = _hcFlat, pw = _hcW, pbase = _hcBase, pscale = _hcScale)
        {
            Glm5Ops.UnweightedRmsNorm(streams, pflat, nd, Config.RmsEps);
            var wf = W($"{prefix}_fn", layer);
            DotKernels.MatVecParallel(wf.Fmt, wf.Ptr, (2 + n) * n, nd, pflat, pw);
            Dequant1D($"{prefix}_base", pbase);
            Dequant1D($"{prefix}_scale", pscale);
            Glm5Ops.HcFinish(pw, pbase, pscale, n, Config.HcEps, Math.Max(1, Config.HcSinkhornIters),
                streams, dim, post, comb, collapsed);
        }
    }

    // ---------------------------------------------------------------- KDA layer

    private unsafe void KdaLayer(int l, float* xn, int pos, float* out0)
    {
        int H = Config.LinearHeads, dk = Config.LinearHeadDim, qkv = H * dk;
        int kernel = Math.Max(1, Config.LinearConvKernel);
        var st = _kdaStates![_kdaSlot[l]];

        // constant per-layer weights (dt_bias, A_log, o_norm, conv taps) — dequant once
        if (!st.Loaded)
        {
            // torch runs one depthwise conv over the concatenated [q|k|v] channels, so the
            // full tap matrix is the row-concatenation of the three per-projection convs
            int qkvKernel = qkv * kernel;
            var qcv = Store.Get($"l{l}.kda.qconv");
            BlockDequantizer.Dequantize(qcv.Format, qcv.Bytes, st.ConvW.AsSpan(0, qkvKernel));
            var kcv = Store.Get($"l{l}.kda.kconv");
            BlockDequantizer.Dequantize(kcv.Format, kcv.Bytes, st.ConvW.AsSpan(qkvKernel, qkvKernel));
            var vcv = Store.Get($"l{l}.kda.vconv");
            BlockDequantizer.Dequantize(vcv.Format, vcv.Bytes, st.ConvW.AsSpan(2 * qkvKernel, qkvKernel));
            var dtv = Store.Get($"l{l}.kda.dt");
            BlockDequantizer.Dequantize(dtv.Format, dtv.Bytes, st.Dt);
            var alogv = Store.Get($"l{l}.kda.alog");
            BlockDequantizer.Dequantize(alogv.Format, alogv.Bytes, st.ALog);
            var onv = Store.Get($"l{l}.kda.onorm");
            BlockDequantizer.Dequantize(onv.Format, onv.Bytes, st.ONorm);
            st.Loaded = true;
        }

        fixed (float* pmixed = _kdaMixed, pg = _kdaG, pbeta = _kdaBeta, pgate = _kdaGate,
               pkvout = _kdaOut, ptmp = _kdaTmp, pconv = st.Conv, pstate = st.S,
               pconvw = st.ConvW, pdt = st.Dt, palog = st.ALog, pon = st.ONorm)
        {
            // 1) q/k/v projections into one contiguous [3*qkv] mixed buffer
            var wq = W($"l{l}.kda.q", l);
            var wk = W($"l{l}.kda.k", l);
            var wv = W($"l{l}.kda.v", l);
            DotKernels.MatVecParallel(wq.Fmt, wq.Ptr, qkv, _dim, xn, pmixed);
            DotKernels.MatVecParallel(wk.Fmt, wk.Ptr, qkv, _dim, xn, pmixed + qkv);
            DotKernels.MatVecParallel(wv.Fmt, wv.Ptr, qkv, _dim, xn, pmixed + 2 * qkv);

            // 2) causal depthwise short conv (+ silu) over all 3*qkv channels, then shift window
            Glm5Ops.CausalConvStep(pmixed, pconv, pconvw, 3 * qkv, kernel, pmixed);

            // 3) forget gate g and beta
            var wfa = W($"l{l}.kda.fa", l);
            var wfb = W($"l{l}.kda.fb", l);
            DotKernels.MatVecParallel(wfa.Fmt, wfa.Ptr, dk, _dim, xn, ptmp);          // [dk]
            DotKernels.MatVecParallel(wfb.Fmt, wfb.Ptr, qkv, dk, ptmp, pg);           // [qkv]
            Glm5Ops.ForgetGate(pg, pdt, palog, H, dk, Config.LinearLowerBound, pg);
            var wbb = W($"l{l}.kda.b", l);
            DotKernels.MatVecParallel(wbb.Fmt, wbb.Ptr, H, _dim, xn, pbeta);
            for (int h = 0; h < H; h++) pbeta[h] = Glm5Ops.Sigmoid(pbeta[h]);

            // 4) KDA recurrence per head (state persists across tokens)
            for (int h = 0; h < H; h++)
                Glm5Ops.KdaStep(
                    q: pmixed + h * dk,
                    k: pmixed + qkv + h * dk,
                    v: pmixed + 2 * qkv + h * dk,
                    g: pg + h * dk,
                    beta: pbeta[h],
                    state: pstate + (long)h * dk * dk,
                    dk: dk, dv: dk,
                    o: pkvout + h * dk);

            // 5) gated output norm + o projection
            var wga = W($"l{l}.kda.ga", l);
            var wgb = W($"l{l}.kda.gb", l);
            DotKernels.MatVecParallel(wga.Fmt, wga.Ptr, dk, _dim, xn, ptmp);          // [dk]
            DotKernels.MatVecParallel(wgb.Fmt, wgb.Ptr, qkv, dk, ptmp, pgate);        // [qkv]
            for (int h = 0; h < H; h++)
            {
                float* oh = pkvout + h * dk;
                // RMSNormGated: fp32 norm over head_dim, * weight, * sigmoid(gate)
                double ss = 0;
                for (int d = 0; d < dk; d++) ss += (double)oh[d] * oh[d];
                float inv = 1f / MathF.Sqrt((float)(ss / dk) + Config.RmsEps);
                for (int d = 0; d < dk; d++)
                    oh[d] = oh[d] * inv * pon[d] * Glm5Ops.Sigmoid(pgate[h * dk + d]);
            }
            var wo = W($"l{l}.kda.o", l);
            DotKernels.MatVecParallel(wo.Fmt, wo.Ptr, _dim, qkv, pkvout, out0);
        }
    }

    // ---------------------------------------------------------------- MLA layer

    /// <summary>
    /// Multi-latent attention (GLM-5.3-Flash / DeepSeek style) with the weight-absorption
    /// trick against the compressed latent cache — no expanded K/V is ever materialized:
    ///   score_h(t) = (W_k,h^T q_h) · c_t          (k absorbed into the query)
    ///   out_h      = W_v,h ( sum_t w_t · c_t )    (linearity of the value projection)
    /// </summary>
    private unsafe void MlaLayer(int l, float* xn, int pos, float* out0)
    {
        int heads = Config.NumHeads;
        int qLora = Config.QLoraRank, kvLora = Config.KvLoraRank;
        int nope = Config.QkNopeHeadDim, rope = Config.QkRopeHeadDim, vdim = Config.VHeadDim;
        int qkDim = nope + rope;
        float eps = Config.RmsEps;
        var cache = MlaCache!;

        fixed (float* pqa = _mlaQa, pq = _mlaQ, pckv = _mlaCkv, pqt = _mlaQtilde,
               pu = _mlaU, psc = _mlaScores, prow = _mlaRow, pattn = _mlaAttn, pnw = _normW, pcs = _cosSin)
        {
            // ---- query path: x -> q_a -> norm -> q_b -> [heads, qkDim] ----
            var wqa = W($"l{l}.mla.qa", l);
            DotKernels.MatVecParallel(wqa.Fmt, wqa.Ptr, qLora, _dim, xn, pqa);
            Dequant1D($"l{l}.mla.qanorm", pnw);
            TransformerOps.RmsNorm(pqa, pnw, pqa, qLora, eps);
            var wqb = W($"l{l}.mla.qb", l);
            DotKernels.MatVecParallel(wqb.Fmt, wqb.Ptr, heads * qkDim, qLora, pqa, pq);

            // ---- kv path: x -> kv_a -> norm(first kvLora) -> latent [kvLora | rope] ----
            var wkva = W($"l{l}.mla.kva", l);
            DotKernels.MatVecParallel(wkva.Fmt, wkva.Ptr, kvLora + rope, _dim, xn, pckv);
            Dequant1D($"l{l}.mla.kvanorm", pnw);
            TransformerOps.RmsNorm(pckv, pnw, pckv, kvLora, eps);   // rope tail untouched

            // optional shared rope key (DeepSeek-V2/V3; GLM-5.3-Flash has rope=0 / NoPE)
            if (rope > 0)
            {
                TransformerOps.BuildCosSinInto(pos, rope, Config.RopeTheta, new Span<float>(pcs, rope));
                for (int off = 0; off < heads * qkDim; off += qkDim)
                    RopeSegments(pq + off + nope, pcs, rope);
                RopeSegments(pckv + kvLora, pcs, rope);
            }
            cache.Append(l, pos, new ReadOnlySpan<float>(pckv, kvLora + rope));

            // ---- absorbed attention against the latent cache ----
            var wkvb = W($"l{l}.mla.kvb", l);
            int perHead = nope + vdim;
            float scale = 1f / MathF.Sqrt(qkDim);
            int seq = pos + 1;

            // pass 1: absorb k into the queries, then score every head against every cached latent
            string kvbName = $"l{l}.mla.kvb";
            var kvbView = Store.Get(kvbName);
            for (int h = 0; h < heads; h++)
            {
                float* qh = pq + h * qkDim;
                float* qh2 = pqt + h * kvLora;
                for (int k = 0; k < kvLora; k++) qh2[k] = 0;
                for (int j = 0; j < nope; j++)
                {
                    BlockDequantizer.Dequantize(wkvb.Fmt,
                        new ReadOnlySpan<byte>(kvbView.RowPtr((long)h * perHead + j), (int)kvbView.RowBytes()),
                        new Span<float>(prow, kvLora));
                    float qv = qh[j];
                    for (int k = 0; k < kvLora; k++) qh2[k] += qv * prow[k];
                }

                int p = 0;
                float* sc = psc + (long)h * cache.MaxSeq;
                for (int page = 0; page < cache.PageCount(l) && p < seq; page++)
                {
                    fixed (float* lp = cache.Page(l, page))
                    {
                        int end = Math.Min(seq, (page + 1) * MlaLatentCache.PageTokens);
                        for (; p < end; p++)
                        {
                            float* ct = lp + (p % MlaLatentCache.PageTokens) * (kvLora + rope);
                            float s = SimdMath.DotF32(qh2, ct, kvLora);
                            if (rope > 0) s += SimdMath.DotF32(qh + nope, ct + kvLora, rope);
                            sc[p] = s * scale;
                        }
                    }
                }
                TransformerOps.Softmax(sc, seq);
            }

            // pass 2: per head, u = sum_t w_t c_t ; out_h = W_v,h u  (row dots)
            for (int h = 0; h < heads; h++)
            {
                float* sc = psc + (long)h * cache.MaxSeq;
                float* uh = pu;
                for (int k = 0; k < kvLora; k++) uh[k] = 0;
                int p = 0;
                for (int page = 0; page < cache.PageCount(l) && p < seq; page++)
                {
                    fixed (float* lp = cache.Page(l, page))
                    {
                        int end = Math.Min(seq, (page + 1) * MlaLatentCache.PageTokens);
                        for (; p < end; p++)
                        {
                            float* ct = lp + (p % MlaLatentCache.PageTokens) * (kvLora + rope);
                            SimdMath.AxpyF32(ct, uh, sc[p], kvLora);
                        }
                    }
                }
                float* oh = pattn + h * vdim;
                for (int j = 0; j < vdim; j++)
                {
                    BlockDequantizer.Dequantize(wkvb.Fmt,
                        new ReadOnlySpan<byte>(kvbView.RowPtr((long)h * perHead + nope + j), (int)kvbView.RowBytes()),
                        new Span<float>(prow, kvLora));
                    oh[j] = SimdMath.DotF32(prow, uh, kvLora);
                }
            }

            var wo = W($"l{l}.mla.o", l);
            DotKernels.MatVecParallel(wo.Fmt, wo.Ptr, _dim, heads * vdim, pattn, out0);
        }
    }

    /// <summary>Rotate-half rope on one rope-length segment (shared MLA rope key or per-head q tail).</summary>
    private static unsafe void RopeSegments(float* v, float* cosSin, int rope)
    {
        int half = rope / 2;
        for (int i = 0; i < half; i++)
        {
            float c = cosSin[i], s = cosSin[half + i];
            float x0 = v[i], x1 = v[i + half];
            v[i] = x0 * c - x1 * s;
            v[i + half] = x0 * s + x1 * c;
        }
    }

    // ---------------------------------------------------------------- FFNs

    /// <summary>Legacy dense FFN with residual add (Llama-style path).</summary>
    private void DenseFfn(int l, float* xn, float* x, float* g, float* u)
    {
        var wg = W($"l{l}.w_gate", l);
        var wu = W($"l{l}.w_up", l);
        var wd = W($"l{l}.w_down", l);
        DotKernels.MatVecParallel(wg.Fmt, wg.Ptr, _ffn, _dim, xn, g);
        DotKernels.MatVecParallel(wu.Fmt, wu.Ptr, _ffn, _dim, xn, u);
        Glm5Ops.LimitedSwiglu(g, u, _ffn, Config.SwigluLimit);   // limit 0 -> plain silu*up
        DotKernels.MatVecParallel(wd.Fmt, wd.Ptr, _dim, _ffn, g, xn);
        SimdMath.AxpyF32(xn, x, 1f, _dim);
    }

    /// <summary>Dense FFN writing only the block output (no residual — GLM-5.x path adds
    /// it through the mHC combine). Uses the limited SwiGLU when the config sets one.</summary>
    private void DenseFfnOut(int l, float* xn, float* out0, float* g, float* u)
    {
        var wg = W($"l{l}.w_gate", l);
        var wu = W($"l{l}.w_up", l);
        var wd = W($"l{l}.w_down", l);
        DotKernels.MatVecParallel(wg.Fmt, wg.Ptr, _ffn, _dim, xn, g);
        DotKernels.MatVecParallel(wu.Fmt, wu.Ptr, _ffn, _dim, xn, u);
        Glm5Ops.LimitedSwiglu(g, u, _ffn, Config.SwigluLimit);
        DotKernels.MatVecParallel(wd.Fmt, wd.Ptr, _dim, _ffn, g, out0);
    }

    /// <summary>Legacy Mixtral-style MoE with residual add (softmax router, llama.cpp top-k norm).</summary>
    private void MoeFfn(int l, float* xn, float* x, float* g, float* u, float* router)
    {
        int nExp = Config.NumExperts, nAct = Math.Min(Config.NumActiveExperts, 8);
        int ffn = _expFfn;

        // first_k_dense_replace-style models (GLM-5.x / DeepSeek / GLM-4.5): the first few
        // layers of a MoE model are dense — fall back per layer when no router exists
        if (!Store.Has($"l{l}.router"))
        {
            DenseFfn(l, xn, x, g, u);
            return;
        }

        // router is an [n_experts, dim] matrix: logits = router · xn (tiny tensor on mmap)
        var rt = Store.Get($"l{l}.router");
        DotKernels.MatVecParallel(rt.Format, rt.Ptr, nExp, _dim, xn, router);
        float max = float.MinValue;
        for (int i = 0; i < nExp; i++) if (router[i] > max) max = router[i];
        float sum = 0;
        for (int i = 0; i < nExp; i++) { router[i] = MathF.Exp(router[i] - max); sum += router[i]; }
        float inv = 1f / sum;
        for (int i = 0; i < nExp; i++) router[i] *= inv;

        // top-k selection, renormalized over the selected experts (llama.cpp style)
        Span<int> sel = stackalloc int[8];
        Span<float> selW = stackalloc float[8];
        SelectTopK(router, nExp, nAct, sel, selW);

        for (int e = 0; e < nAct; e++)
        {
            int exp = sel[e];
            float wgt = selW[e];
            var wg = W($"l{l}.e{exp}.w_gate", l);
            var wu = W($"l{l}.e{exp}.w_up", l);
            var wd = W($"l{l}.e{exp}.w_down", l);
            DotKernels.MatVecParallel(wg.Fmt, wg.Ptr, ffn, _dim, xn, g);
            DotKernels.MatVecParallel(wu.Fmt, wu.Ptr, ffn, _dim, xn, u);
            TransformerOps.SiluMulInPlace(g, u, ffn);
            DotKernels.MatVecParallel(wd.Fmt, wd.Ptr, _dim, ffn, g, u);
            SimdMath.AxpyF32(u, x, wgt, _dim);
        }
    }

    /// <summary>Legacy Mixtral-style MoE writing the summed expert output (no residual add).</summary>
    private void LegacyMoeInto(int l, float* xn, float* out0, float* router, float* g, float* u)
    {
        int nExp = Config.NumExperts, nAct = Math.Min(Config.NumActiveExperts, 8);
        int ffn = _expFfn;

        // first_k_dense_replace-style models (GLM-5.x / DeepSeek / GLM-4.5): the first few
        // layers of a MoE model are dense — fall back per layer when no router exists
        if (!Store.Has($"l{l}.router"))
        {
            DenseFfnOut(l, xn, out0, g, u);
            return;
        }

        // router is an [n_experts, dim] matrix: logits = router · xn (tiny tensor on mmap)
        var rt = Store.Get($"l{l}.router");
        DotKernels.MatVecParallel(rt.Format, rt.Ptr, nExp, _dim, xn, router);
        float max = float.MinValue;
        for (int i = 0; i < nExp; i++) if (router[i] > max) max = router[i];
        float sum = 0;
        for (int i = 0; i < nExp; i++) { router[i] = MathF.Exp(router[i] - max); sum += router[i]; }
        float inv = 1f / sum;
        for (int i = 0; i < nExp; i++) router[i] *= inv;

        // top-k selection, renormalized over the selected experts (llama.cpp style)
        Span<int> sel = stackalloc int[8];
        Span<float> selW = stackalloc float[8];
        SelectTopK(router, nExp, nAct, sel, selW);

        for (int d = 0; d < _dim; d++) out0[d] = 0;
        for (int e = 0; e < nAct; e++)
        {
            int exp = sel[e];
            float wgt = selW[e];
            var wg = W($"l{l}.e{exp}.w_gate", l);
            var wu = W($"l{l}.e{exp}.w_up", l);
            var wd = W($"l{l}.e{exp}.w_down", l);
            DotKernels.MatVecParallel(wg.Fmt, wg.Ptr, ffn, _dim, xn, g);
            DotKernels.MatVecParallel(wu.Fmt, wu.Ptr, ffn, _dim, xn, u);
            TransformerOps.SiluMulInPlace(g, u, ffn);
            DotKernels.MatVecParallel(wd.Fmt, wd.Ptr, _dim, ffn, g, u);
            SimdMath.AxpyF32(u, out0, wgt, _dim);
        }
    }

    /// <summary>
    /// GLM-5.x MoE (Glm5NextTextTopkRouter + Experts + shared expert):
    /// sigmoid scores, noaux_tc choice bias, top-k on (score + bias), weights renormalized
    /// over the selected set and scaled by routed_scaling_factor; limited SwiGLU experts;
    /// the shared expert is added unconditionally with weight 1.
    /// </summary>
    private unsafe void GlmMoeFfn(int l, float* xn, float* out0, float* router, float* g, float* u)
    {
        int nExp = Config.NumExperts, nAct = Math.Min(Config.NumActiveExperts, 8);
        int ffn = _expFfn;
        float limit = Config.SwigluLimit;

        // router logits (fp32 path per moe_router_dtype)
        var rt = Store.Get($"l{l}.router");
        DotKernels.MatVecParallel(rt.Format, rt.Ptr, nExp, _dim, xn, router);

        // scores = sigmoid(logits); choice = score + e_score_correction_bias
        float* scores = stackalloc float[512];
        float* choice = stackalloc float[512];
        nExp = Math.Min(nExp, 512);
        if (_routerBias != null)
        {
            fixed (float* pb = _routerBias)
            {
                var bv = Store.Get($"l{l}.router_bias");
                BlockDequantizer.Dequantize(bv.Format, bv.Bytes, new Span<float>(pb, nExp));
                for (int i = 0; i < nExp; i++)
                {
                    scores[i] = Glm5Ops.Sigmoid(router[i]);
                    choice[i] = scores[i] + pb[i];
                }
            }
        }
        else
        {
            for (int i = 0; i < nExp; i++)
            {
                scores[i] = Glm5Ops.Sigmoid(router[i]);
                choice[i] = scores[i];
            }
        }

        // top-k by choice (n_group = 1: no group masking), weights from raw scores
        Span<int> sel = stackalloc int[8];
        SelectTopKIndices(choice, nExp, nAct, sel);
        float wsum = 0;
        for (int e = 0; e < nAct; e++) wsum += scores[sel[e]];
        float inv2 = 1f / (wsum + 1e-20f);

        for (int d = 0; d < _dim; d++) out0[d] = 0;
        for (int e = 0; e < nAct; e++)
        {
            int exp = sel[e];
            float wgt = scores[exp] * inv2 * Config.RoutedScale;
            var wg = W($"l{l}.e{exp}.w_gate", l);
            var wu = W($"l{l}.e{exp}.w_up", l);
            var wd = W($"l{l}.e{exp}.w_down", l);
            DotKernels.MatVecParallel(wg.Fmt, wg.Ptr, ffn, _dim, xn, g);
            DotKernels.MatVecParallel(wu.Fmt, wu.Ptr, ffn, _dim, xn, u);
            Glm5Ops.LimitedSwiglu(g, u, ffn, limit);
            DotKernels.MatVecParallel(wd.Fmt, wd.Ptr, _dim, ffn, g, u);
            SimdMath.AxpyF32(u, out0, wgt, _dim);
        }

        // shared expert (always on, weight 1)
        if (Config.NumSharedExperts > 0 && Store.Has($"l{l}.sw_gate"))
        {
            var wg = W($"l{l}.sw_gate", l);
            var wu = W($"l{l}.sw_up", l);
            var wd = W($"l{l}.sw_down", l);
            DotKernels.MatVecParallel(wg.Fmt, wg.Ptr, ffn, _dim, xn, g);
            DotKernels.MatVecParallel(wu.Fmt, wu.Ptr, ffn, _dim, xn, u);
            Glm5Ops.LimitedSwiglu(g, u, ffn, limit);
            DotKernels.MatVecParallel(wd.Fmt, wd.Ptr, _dim, ffn, g, u);
            SimdMath.AxpyF32(u, out0, 1f, _dim);
        }
    }

    /// <summary>Top-k indices only (GLM router: weights come from the raw scores afterwards).</summary>
    private static unsafe void SelectTopKIndices(float* v, int n, int k, Span<int> sel)
    {
        var idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => v[b].CompareTo(v[a]));
        for (int i = 0; i < k && i < n && i < sel.Length; i++) sel[i] = idx[i];
    }

    private static unsafe void SelectTopK(float* v, int n, int k, Span<int> sel, Span<float> selW)
    {
        var idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => v[b].CompareTo(v[a]));
        float sum = 0;
        for (int i = 0; i < k; i++) sum += v[idx[i]];
        if (sum <= 0) sum = 1;
        for (int i = 0; i < k; i++) { sel[i] = idx[i]; selW[i] = v[idx[i]] / sum; }
    }

    // ---------------------------------------------------------------- generation

    public GenerationStats LastStats { get; private set; } = new(0, 0, 0, 0, 0, 0, 0, 0);

    /// <summary>Let orchestrators (e.g. SpeculativeRunner) publish their own stats block.</summary>
    public void PublishStats(GenerationStats stats) => LastStats = stats;

    /// <summary>
    /// Stream-generate. <paramref name="onToken"/>(tokenId, indexSoFar, tokensPerSec) fires per
    /// generated token while the loop keeps running — true token streaming, not batch output.
    /// </summary>
    public int[] Generate(int[] prompt, int maxNewTokens, Sampler sampler, Action<int, int, double>? onToken = null)
    {
        var sw = Stopwatch.StartNew();
        var recent = new List<int>(prompt.Length + maxNewTokens);
        recent.AddRange(prompt);

        int pos = 0;
        float[] logits = _logits;
        foreach (var t in prompt)
        {
            logits = Forward(t, pos);
            pos++;
            if (pos >= _maxCtx) break;
        }
        double promptSecs = sw.Elapsed.TotalSeconds;
        double promptTokPerSec = prompt.Length / Math.Max(promptSecs, 1e-9);

        var generated = new List<int>(maxNewTokens);
        long genStart = sw.ElapsedTicks;
        double tokPerSec = 0;

        for (int i = 0; i < maxNewTokens; i++)
        {
            if (pos >= _maxCtx) break;
            int next = sampler.Sample(logits, CollectionsMarshal.AsSpan(recent));
            if (next == 0) break;   // EOS convention (id 0)

            generated.Add(next);
            recent.Add(next);
            onToken?.Invoke(next, generated.Count, tokPerSec);

            if (i + 1 < maxNewTokens && pos < _maxCtx)
            {
                logits = Forward(next, pos);
                pos++;
                long dt = sw.ElapsedTicks - genStart;
                tokPerSec = generated.Count / (dt / (double)Stopwatch.Frequency);
            }
        }

        sw.Stop();
        LastStats = new GenerationStats(prompt.Length, generated.Count, promptSecs, promptTokPerSec,
            tokPerSec, Prefetch?.BytesStreamed ?? 0,
            Prefetch != null ? Prefetch.WaitTicks * 1000.0 / Stopwatch.Frequency : 0,
            AttentionStateBytes);
        return generated.ToArray();
    }

    /// <summary>RAM used by attention state: KV pages (legacy), MLA latent pages and KDA states.</summary>
    public long AttentionStateBytes =>
        (Kv?.BytesUsed ?? 0) + (MlaCache?.BytesUsed ?? 0) +
        (_kdaStates?.Sum(s => (long)s.Conv.Length + s.S.Length + s.ConvW.Length + s.Dt.Length + s.ALog.Length + s.ONorm.Length) ?? 0) * 4;

    public void ResetConversation()
    {
        Kv?.Reset();
        MlaCache?.Reset();
        if (_kdaStates != null)
            foreach (var s in _kdaStates)
            {
                Array.Clear(s.Conv);
                Array.Clear(s.S);
            }
    }

    /// <summary>Hard context cap the engine enforces in Forward/Generate.</summary>
    public int MaxCtx => _maxCtx;

    // ---- speculative decoding state management ----

    /// <summary>Copy all KDA recurrent states (S matrices + conv windows) into an internal
    /// scratch buffer. Returns the number of floats saved (0 for models without KDA layers).</summary>
    public long SnapshotRecurrentState()
    {
        if (_kdaStates == null) return 0;
        long need = 0;
        foreach (var s in _kdaStates) need += s.S.Length + s.Conv.Length;
        if (_kdaSnap == null || _kdaSnap.Length < need) _kdaSnap = new float[need];
        long off = 0;
        foreach (var s in _kdaStates)
        {
            Array.Copy(s.S, 0, _kdaSnap, off, s.S.Length); off += s.S.Length;
            Array.Copy(s.Conv, 0, _kdaSnap, off, s.Conv.Length); off += s.Conv.Length;
        }
        return off;
    }

    /// <summary>Restore what SnapshotRecurrentState saved (same engine instance, same layer set).</summary>
    public void RestoreRecurrentState()
    {
        if (_kdaStates == null) return;
        long off = 0;
        foreach (var s in _kdaStates)
        {
            Array.Copy(_kdaSnap!, off, s.S, 0, s.S.Length); off += s.S.Length;
            Array.Copy(_kdaSnap!, off, s.Conv, 0, s.Conv.Length); off += s.Conv.Length;
        }
    }

    /// <summary>Rewind the attention caches to <paramref name="pos"/> tokens (KDA state is
    /// handled by Snapshot/RestoreRecurrentState). Stale page bytes are never read before
    /// being overwritten — truncation is O(1).</summary>
    public void TruncateTo(int pos)
    {
        Kv?.Truncate(pos);
        MlaCache?.Truncate(pos);
    }

    public void Dispose()
    {
        Prefetch?.Dispose();
        Store.Dispose();
    }
}
