using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace QuantRunner.Core.Kernels;

/// <summary>Element-wise transformer ops, SIMD-optimized where safe.</summary>
public static unsafe class TransformerOps
{
    // ------------------------------- RMSNorm -------------------------------

    public static void RmsNorm(float* x, float* weights, float* output, int n, float eps)
    {
        float ss = 0;
        int i = 0;
        if (Vector256.IsHardwareAccelerated && n >= 8)
        {
            var acc = Vector256<float>.Zero;
            for (; i + 8 <= n; i += 8)
            {
                var vf = Vector256.Load(x + i);
                acc = Vector256.Add(acc, Vector256.Multiply(vf, vf));
            }
            ss = SimdMath.HSum(acc);
        }
        for (; i < n; i++) ss += x[i] * x[i];
        float scale = 1f / MathF.Sqrt(ss / n + eps);
        i = 0;
        if (Vector256.IsHardwareAccelerated && n >= 8)
        {
            var vs = Vector256.Create(scale);
            for (; i + 8 <= n; i += 8)
                Vector256.Store(Vector256.Multiply(Vector256.Load(x + i), vs), output + i);
        }
        for (; i < n; i++) output[i] = x[i] * scale;
        if (weights != null)
        {
            for (i = 0; i < n; i++) output[i] *= weights[i];
        }
    }

    // ------------------------------- RoPE -------------------------------

    /// <summary>
    /// cos/sin table row for one position: cos[headDim/2] then sin[headDim/2].
    /// Computed once per position (shared across all layers and tokens).
    /// </summary>
    public static float[] BuildCosSin(long pos, int headDim, float theta)
    {
        var row = new float[headDim];
        BuildCosSinInto(pos, headDim, theta, row);
        return row;
    }

    /// <summary>In-place variant writing into a caller-owned buffer (no per-token allocation).</summary>
    public static void BuildCosSinInto(long pos, int headDim, float theta, Span<float> row)
    {
        int half = headDim / 2;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Pow(theta, -2f * i / headDim);
            float ang = pos * freq;
            row[i] = MathF.Cos(ang);
            row[half + i] = MathF.Sin(ang);
        }
    }

    /// <summary>Rotate-half RoPE (llama-style) in place on q [nQ] and k [nK] using a precomputed cos/sin row.</summary>
    public static void Rope(float* q, float* k, float* cosSin, int nQ, int nK, int headDim)
    {
        int half = headDim / 2;
        for (int off = 0; off < nQ; off += headDim)
            RopeHead(q + off, cosSin, half);
        for (int off = 0; off < nK; off += headDim)
            RopeHead(k + off, cosSin, half);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RopeHead(float* v, float* cosSin, int half)
    {
        for (int i = 0; i < half; i++)
        {
            float c = cosSin[i], s = cosSin[half + i];
            float x0 = v[i], x1 = v[i + half];
            v[i] = x0 * c - x1 * s;
            v[i + half] = x0 * s + x1 * c;
        }
    }

    // ------------------------------- Softmax -------------------------------

    public static void Softmax(float* x, int n)
    {
        float max = float.MinValue;
        for (int i = 0; i < n; i++) if (x[i] > max) max = x[i];
        float sum = 0;
        for (int i = 0; i < n; i++) { x[i] = MathF.Exp(x[i] - max); sum += x[i]; }
        float inv = 1f / sum;
        int j = 0;
        if (Vector256.IsHardwareAccelerated && n >= 8)
        {
            var vi = Vector256.Create(inv);
            for (; j + 8 <= n; j += 8)
                Vector256.Store(Vector256.Multiply(Vector256.Load(x + j), vi), x + j);
        }
        for (; j < n; j++) x[j] *= inv;
    }

    // ------------------------------- SwiGLU: g = silu(g) * u -------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Silu(float v) => v / (1f + MathF.Exp(-v));

    public static void SiluMulInPlace(float* g, float* u, int n)
    {
        for (int i = 0; i < n; i++)
        {
            float gv = g[i];
            g[i] = gv / (1f + MathF.Exp(-gv)) * u[i];
        }
    }
}
