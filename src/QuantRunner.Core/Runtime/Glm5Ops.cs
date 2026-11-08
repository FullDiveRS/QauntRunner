using System.Runtime.CompilerServices;
using QuantRunner.Core.Kernels;
using QuantRunner.Core.Quantization;

namespace QuantRunner.Core.Runtime;

/// <summary>
/// Pure math for the GLM-5.x family forward pass (validated against transformers'
/// modeling_glm5_next.py): manifold-constrained hyper-connections (mHC), Kimi delta
/// attention (KDA) recurrent step, short causal conv, and limited SwiGLU.
/// The MLA absorbed-attention loop lives in StreamingEngine (it owns the latent cache);
/// the reusable pieces are here.
/// </summary>
public static unsafe class Glm5Ops
{
    public const float L2NormEps = 1e-6f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Softplus(float v) => v > 20f ? v : MathF.Log(1f + MathF.Exp(v));

    // ------------------------------------------------------------------ norms

    /// <summary>RMSNorm without a weight (mHC input_norm): normalizes over the whole
    /// flattened N*D stream vector, exactly like Glm5NextTextUnweightedRMSNorm.</summary>
    public static void UnweightedRmsNorm(float* x, float* dst, int n, float eps)
    {
        double ss = 0;
        for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
        if (Environment.GetEnvironmentVariable("QR_TRACE") == "1" && n == 256)
        {
            double sm = 0;
            for (int i = 0; i < n; i++) sm += x[i];
            Console.WriteLine($"[trace] unorm n={n} ss={ss:F6} x0={x[0]:F6} x127={x[127]:F6} x128={x[128]:F6} x255={x[255]:F6} sum={sm:F6}");
        }
        float inv = 1f / MathF.Sqrt((float)(ss / n) + eps);
        for (int i = 0; i < n; i++) dst[i] = x[i] * inv;
    }

    /// <summary>FLA-style L2 norm: x / sqrt(sum(x^2) + eps) over the vector.</summary>
    public static void L2Norm(float* v, int n, float eps = L2NormEps)
    {
        double ss = 0;
        for (int i = 0; i < n; i++) ss += (double)v[i] * v[i];
        float inv = 1f / MathF.Sqrt((float)ss + eps);
        for (int i = 0; i < n; i++) v[i] *= inv;
    }

    // ------------------------------------------------------------------ mHC

    /// <summary>
    /// Sinkhorn-Knopp normalization of an N x N row-major matrix toward doubly stochastic,
    /// matching Glm5NextTextHyperConnection: first column-normalize, then
    /// (iters-1) x [row-normalize, column-normalize]. eps guards empty sums.
    /// </summary>
    public static void Sinkhorn(float* m, int n, int iters, float eps)
    {
        // column sums (dim=-2): for each column j, divide column by its sum
        for (int j = 0; j < n; j++)
        {
            float s = 0;
            for (int i = 0; i < n; i++) s += m[i * n + j];
            float inv = 1f / (s + eps);
            for (int i = 0; i < n; i++) m[i * n + j] *= inv;
        }
        for (int it = 0; it < iters - 1; it++)
        {
            for (int i = 0; i < n; i++)   // rows (dim=-1)
            {
                float s = 0;
                for (int j = 0; j < n; j++) s += m[i * n + j];
                float inv = 1f / (s + eps);
                for (int j = 0; j < n; j++) m[i * n + j] *= inv;
            }
            for (int j = 0; j < n; j++)   // cols (dim=-2)
            {
                float s = 0;
                for (int i = 0; i < n; i++) s += m[i * n + j];
                float inv = 1f / (s + eps);
                for (int i = 0; i < n; i++) m[i * n + j] *= inv;
            }
        }
    }

    /// <summary>
    /// Second half of the mHC block: turn the projected weights into pre/post/comb and
    /// collapse the input streams into the block input. (The fn matvec is done by the
    /// caller through DotKernels so quantized formats work.)
    /// </summary>
    /// <param name="w">projected weights [(2+N)*N] (fn @ normalized streams)</param>
    /// <param name="b">base bias [(2+N)*N]</param>
    /// <param name="s">scale [3] (pre, post, comb)</param>
    /// <param name="streams">input residual streams [N, D]</param>
    /// <param name="post">out [N]</param>
    /// <param name="comb">out [N*N] (row-major, Sinkhorn-normalized)</param>
    /// <param name="collapsed">out [D]: sum_i pre[i] * streams[i]</param>
    public static void HcFinish(float* w, float* b, float* s, int n, float eps, int sinkhornIters,
        float* streams, int dim, float* post, float* comb, float* collapsed)
    {
        for (int i = 0; i < n; i++)
        {
            float pre = Sigmoid(w[i] * s[0] + b[i]) + eps;
            post[i] = 2f * Sigmoid(w[n + i] * s[1] + b[n + i]);
            // comb rows: softmax over j (dim=-1), then + eps.
            // layout: w = [pre (n) | post (n) | comb (n*n)] — comb starts at 2n!
            float* row = comb + i * n;
            float max = float.MinValue;
            for (int j = 0; j < n; j++)
            {
                float v = w[2 * n + i * n + j] * s[2] + b[2 * n + i * n + j];
                row[j] = v;
                if (v > max) max = v;
            }
            float sum = 0;
            for (int j = 0; j < n; j++) { row[j] = MathF.Exp(row[j] - max); sum += row[j]; }
            float inv = 1f / sum;
            for (int j = 0; j < n; j++) row[j] = row[j] * inv + eps;
        }
        Sinkhorn(comb, n, sinkhornIters, eps);
        // collapsed = sum_i pre[i] * streams[i]
        for (int d = 0; d < dim; d++) collapsed[d] = 0f;
        for (int i = 0; i < n; i++)
        {
            float pre = Sigmoid(w[i] * s[0] + b[i]) + eps;
            float* src = streams + i * dim;
            for (int d = 0; d < dim; d++) collapsed[d] += pre * src[d];
        }
    }

    /// <summary>
    /// Apply a block's output back onto the residual streams:
    /// streams[i] = post[i] * blockOut + sum_m comb[m, i] * streams[m]
    /// (equivalent to torch.matmul(comb.transpose(-1, -2), residual)).
    /// tmp must be N*D floats of scratch — the update reads the ORIGINAL streams, so it
    /// cannot be done strictly in place.
    /// </summary>
    public static void HcCombine(float* streams, float* blockOut, float* post, float* comb, int n, int dim, float* tmp)
    {
        for (int i = 0; i < n; i++)
        {
            float* dst = tmp + i * dim;
            float p = post[i];
            for (int d = 0; d < dim; d++)
            {
                float acc = p * blockOut[d];
                for (int m = 0; m < n; m++) acc += comb[m * n + i] * streams[m * dim + d];
                dst[d] = acc;
            }
        }
        for (int i = 0; i < n * dim; i++) streams[i] = tmp[i];
    }

    // ------------------------------------------------------------------ KDA

    /// <summary>
    /// One decode step of the Kimi delta attention recurrence (recurrent_kimi_delta_attention,
    /// fine-grained gating): state[h] = state[h] * exp(g) ; delta = (v - state^T k) * beta ;
    /// state += k delta^T ; o = state^T q. q and k are L2-normalized in place, q is scaled by
    /// 1/sqrt(dk) — both match the reference (which casts to fp32 first, as we do).
    /// </summary>
    /// <param name="state">[dk, dv] row-major, updated in place</param>
    public static void KdaStep(float* q, float* k, float* v, float* g, float beta,
        float* state, int dk, int dv, float* o)
    {
        L2Norm(q, dk);
        L2Norm(k, dk);
        float scale = 1f / MathF.Sqrt(dk);
        for (int i = 0; i < dk; i++) q[i] *= scale;

        // decay per (k-dim) channel: state[j, :] *= exp(g[j])
        for (int j = 0; j < dk; j++)
        {
            float e = MathF.Exp(g[j]);
            float* row = state + j * dv;
            for (int s = 0; s < dv; s++) row[s] *= e;
        }
        // kv_mem = state^T k  (sum over k-dim), delta = (v - kv_mem) * beta
        for (int s = 0; s < dv; s++)
        {
            float mem = 0;
            for (int j = 0; j < dk; j++) mem += state[j * dv + s] * k[j];
            float delta = (v[s] - mem) * beta;
            // rank-1 update + read out in the same pass: state[j,s] += k[j]*delta; o[s] = sum_j state[j,s]*q[j]
            float acc = 0;
            for (int j = 0; j < dk; j++)
            {
                float nv = state[j * dv + s] + k[j] * delta;
                state[j * dv + s] = nv;
                acc += nv * q[j];
            }
            o[s] = acc;
        }
    }

    /// <summary>
    /// Decode-time causal depthwise conv1d step (causal_conv1d_update + silu):
    /// out[c] = silu(sum_j w[c, j] * xs[j]) with xs = [conv window (K-1) | input].
    /// The window is shifted afterwards. Weights are float32 [channels, K].
    /// </summary>
    public static void CausalConvStep(float* input, float* convState, float* w,
        int channels, int kernel, float* output)
    {
        int stateLen = kernel - 1;
        for (int c = 0; c < channels; c++)
        {
            // window: state[c, 0..stateLen-1] then input[c] — sum taps in time order
            float acc = 0;
            float* win = convState + c * stateLen;
            float* wc = w + c * kernel;
            for (int j = 0; j < stateLen; j++) acc += wc[j] * win[j];
            acc += wc[stateLen] * input[c];
            float cur = input[c];
            output[c] = TransformerOps.Silu(acc);
            // shift window: drop oldest, append current
            for (int j = 0; j < stateLen - 1; j++) win[j] = win[j + 1];
            if (stateLen > 0) win[stateLen - 1] = cur;
        }
    }

    /// <summary>Forget gate (Glm5NextTextForgetGate) — GLM bounded-sigmoid or Kimi softplus path.
    /// fbOut has dt_bias added already; decay = exp(A_log[head]). Writes g[head * headDim + d].</summary>
    public static void ForgetGate(float* fbOut, float* dtBias, float* aLog, int heads, int headDim,
        float lowerBound, float* g)
    {
        int qkvDim = heads * headDim;
        for (int h = 0; h < heads; h++)
        {
            float decay = MathF.Exp(aLog[h]);
            for (int d = 0; d < headDim; d++)
            {
                float x = fbOut[h * headDim + d] + dtBias[h * headDim + d];
                g[h * headDim + d] = lowerBound != 0
                    ? lowerBound * Sigmoid(decay * x)
                    : -decay * Softplus(x);
            }
        }
        _ = qkvDim;
    }

    /// <summary>Limited SwiGLU (GLM-5.x): g = silu(clamp_max(g, limit)) * clamp(u, -limit, limit). limit &lt;= 0: plain.</summary>
    public static void LimitedSwiglu(float* g, float* u, int n, float limit)
    {
        if (limit <= 0)
        {
            TransformerOps.SiluMulInPlace(g, u, n);
            return;
        }
        for (int i = 0; i < n; i++)
        {
            float gv = g[i] > limit ? limit : g[i];
            float uv = u[i]; uv = uv > limit ? limit : (uv < -limit ? -limit : uv);
            g[i] = TransformerOps.Silu(gv) * uv;
        }
    }
}
