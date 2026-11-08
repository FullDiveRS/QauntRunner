using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using QuantRunner.Core.Quantization;

namespace QuantRunner.Core.Kernels;

/// <summary>
/// Fused dequantize + dot-product kernels: y[r] = sum_c W[r,c] * x[c] computed directly on
/// compressed block bytes without materializing floats (the AirLLM killer: no fp16 round-trip,
/// no torch tensor allocation, single pass over the streamed weight bytes).
/// </summary>
public static unsafe class DotKernels
{
    // ------------------------------- scalar accumulators (per row) -------------------------------

    private static readonly ThreadLocal<float[]> Buf = new(() => new float[512]);

    // ------------------------------- public row dot -------------------------------

    /// <summary>Dot product of one weight row (compressed bytes) with vector x. cols must be block-aligned.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotRow(QuantFormat fmt, byte* w, float* x, int cols)
    {
        switch (fmt)
        {
            case QuantFormat.F32: return DotF32(w, x, cols);
            case QuantFormat.F16: return DotF16(w, x, cols);
            case QuantFormat.BF16: return DotBf16(w, x, cols);
            case QuantFormat.Q4_0: return DotQ4_0(w, x, cols);
            case QuantFormat.Q8_0: return DotQ8_0(w, x, cols);
            case QuantFormat.Q2_K: return DotQ2_K(w, x, cols);
            case QuantFormat.Q3_K: return DotQ3_K(w, x, cols);
            case QuantFormat.Q4_K: return DotQ4_K(w, x, cols);
            case QuantFormat.Q5_K: return DotQ5_K(w, x, cols);
            case QuantFormat.Q6_K: return DotQ6_K(w, x, cols);
            case QuantFormat.TRI: return DotTri(w, x, cols);
            case QuantFormat.I8_BLK: return DotI8Blk(w, x, cols);
            case QuantFormat.B1: return DotB1(w, x, cols);
            case QuantFormat.B1X: return DotB1X(w, x, cols);
            case QuantFormat.T1E: return DotT1E(w, x, cols, 34);
            case QuantFormat.T1E_LO: return DotT1E(w, x, cols, 26);
            default: throw new NotSupportedException(fmt.ToString());
        }
    }

    // ------------------------------- fp formats -------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotF32(byte* w, float* x, int cols)
    {
        return SimdMath.DotF32((float*)w, x, cols);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotF16(byte* w, float* x, int cols)
    {
        if (Vector256.IsHardwareAccelerated && BitConverter.IsLittleEndian && cols >= 8)
        {
            var acc = Vector256<float>.Zero;
            int i = 0;
            ushort* s = (ushort*)w;
            for (; i + 8 <= cols; i += 8)
            {
                var vh = Vector128.LoadUnsafe(ref s[i]);            // 8 halves
                // REAL f16 -> f32 (the bits<<16 trick is only valid for BF16!)
                var (u0, u1) = Vector128.Widen(vh);                 // 4+4 zero-extended uints
                var f0 = HalfBitsToFloat(u0);
                var f1 = HalfBitsToFloat(u1);
                var vf = Vector256.Create(f0, f1);
                acc = Vector256.Add(acc, Vector256.Multiply(vf, Vector256.Load(x + i)));
            }
            float sum = SimdMath.HSum(acc);
            for (; i < cols; i++) sum += Dtypes.F16ToFloat(s[i]) * x[i];
            return sum;
        }
        ushort* sp = (ushort*)w;
        float ss = 0;
        for (int i = 0; i < cols; i++) ss += Dtypes.F16ToFloat(sp[i]) * x[i];
        return ss;
    }

    /// <summary>Exact fp16 -> fp32 lane conversion: normals via exponent re-bias (+112),
    /// subnormals via integer->float scale, inf/NaN fall out of the normal formula.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> HalfBitsToFloat(Vector128<uint> u)
    {
        var sign = (u & Vector128.Create(0x8000u)) << 16;
        var exp = (u >> 10) & Vector128.Create(0x1Fu);
        var man = (u & Vector128.Create(0x3FFu)) << 13;
        var norBits = sign | ((exp + Vector128.Create(112u)) << 23) | man;
        // subnormals (exp == 0): value = raw mantissa * 2^-24, sign preserved
        var sub = Vector128.ConvertToSingle(u & Vector128.Create(0x3FFu)) * Vector128.Create(5.9604645e-8f);
        var subBits = sub.AsUInt32() | sign;
        var isSub = Vector128.Equals(exp, Vector128<uint>.Zero);
        var bits = (norBits & ~isSub) | (subBits & isSub);
        return bits.AsSingle();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotBf16(byte* w, float* x, int cols)
    {
        if (Vector256.IsHardwareAccelerated && BitConverter.IsLittleEndian && cols >= 8)
        {
            var acc = Vector256<float>.Zero;
            int i = 0;
            ushort* s = (ushort*)w;
            for (; i + 8 <= cols; i += 8)
            {
                var vh = Vector256.LoadUnsafe(ref *s, (nuint)i).GetLower();
                var (u0, u1) = Vector128.Widen(vh);
                var b0 = u0.AsUInt32() << 16;
                var b1 = u1.AsUInt32() << 16;
                var vf = Vector256.Create(b0.AsSingle(), b1.AsSingle());
                var vx = Vector256.Load(x + i);
                acc = Vector256.Add(acc, Vector256.Multiply(vf, vx));
            }
            float sum = SimdMath.HSum(acc);
            for (; i < cols; i++) sum += Dtypes.Bf16ToFloat(s[i]) * x[i];
            return sum;
        }
        ushort* sp = (ushort*)w;
        float ss = 0;
        for (int i = 0; i < cols; i++) ss += Dtypes.Bf16ToFloat(sp[i]) * x[i];
        return ss;
    }

    // ------------------------------- Q4_0: y = d*(q-8) -------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotQ4_0(byte* w, float* x, int cols)
    {
        int nb = cols / 32;
        Span<float> lo = Buf.Value; Span<float> hi = Buf.Value.AsSpan(16);
        bool vec = Vector256.IsHardwareAccelerated;
        float sum = 0f;
        for (int b = 0; b < nb; b++, w += 20, x += 32)
        {
            float d = Dtypes.F16ToFloat((ushort)(w[0] | (w[1] << 8)));
            ReadOnlySpan<byte> qs = new(w + 2, 16);
            SimdMath.UnpackNibbles32(qs, lo, hi);
            float dotQ = 0, sumX = 0;
            if (vec)
            {
                var accQ = Vector256<float>.Zero;
                var accX = Vector256<float>.Zero;
                ref float xr = ref *x;
                var vq0 = Vector256.LoadUnsafe(ref lo[0]);
                var vq1 = Vector256.LoadUnsafe(ref lo[8]);
                var vq2 = Vector256.LoadUnsafe(ref hi[0]);
                var vq3 = Vector256.LoadUnsafe(ref hi[8]);
                var vx0 = Vector256.LoadUnsafe(ref xr, 0);
                var vx1 = Vector256.LoadUnsafe(ref xr, 8);
                var vx2 = Vector256.LoadUnsafe(ref xr, 16);
                var vx3 = Vector256.LoadUnsafe(ref xr, 24);
                accQ = Vector256.Add(accQ, Vector256.Multiply(vq0, vx0));
                accQ = Vector256.Add(accQ, Vector256.Multiply(vq1, vx1));
                accQ = Vector256.Add(accQ, Vector256.Multiply(vq2, vx2));
                accQ = Vector256.Add(accQ, Vector256.Multiply(vq3, vx3));
                accX = Vector256.Add(accX, vx0);
                accX = Vector256.Add(accX, vx1);
                accX = Vector256.Add(accX, vx2);
                accX = Vector256.Add(accX, vx3);
                dotQ = SimdMath.HSum(accQ);
                sumX = SimdMath.HSum(accX);
            }
            else
            {
                for (int i = 0; i < 16; i++)
                {
                    dotQ += lo[i] * x[i] + hi[i] * x[i + 16];
                    sumX += x[i] + x[i + 16];
                }
            }
            sum += d * dotQ - 8f * d * sumX;
        }
        return sum;
    }

    // ------------------------------- Q8_0: y = d*q -------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotQ8_0(byte* w, float* x, int cols)
    {
        int nb = cols / 32;
        float sum = 0f;
        Span<float> qf = Buf.Value;
        for (int b = 0; b < nb; b++, w += 34, x += 32)
        {
            float d = Dtypes.F16ToFloat((ushort)(w[0] | (w[1] << 8)));
            float dot;
            if (Vector256.IsHardwareAccelerated)
            {
                dot = DotI8Bytes(w + 2, x, 32, qf);
            }
            else
            {
                dot = 0;
                for (int i = 0; i < 32; i++) dot += (sbyte)w[2 + i] * x[i];
            }
            sum += d * dot;
        }
        return sum;
    }

    /// <summary>int8 bytes -> float dot with x[0..n).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotI8Bytes(byte* q, float* x, int n, Span<float> qf)
    {
        if (Vector256.IsHardwareAccelerated && BitConverter.IsLittleEndian)
        {
            var acc = Vector256<float>.Zero;
            int i = 0;
            for (; i + 16 <= n; i += 16)
            {
                Vector128<byte> b = Vector128.Load(q + i);
                var (s0, s1) = Vector128.Widen(b.AsSByte());       // short x8 + x8 (sign-extended)
                var (i0, i1) = Vector128.Widen(s0);
                var (i2, i3) = Vector128.Widen(s1);
                var v0 = Vector256.Create(Vector128.ConvertToSingle(i0), Vector128.ConvertToSingle(i1));
                var v1 = Vector256.Create(Vector128.ConvertToSingle(i2), Vector128.ConvertToSingle(i3));
                var vx0 = Vector256.Load(x + i);
                var vx1 = Vector256.Load(x + i + 8);
                acc = Vector256.Add(acc, Vector256.Multiply(v0, vx0));
                acc = Vector256.Add(acc, Vector256.Multiply(v1, vx1));
            }
            float sum = SimdMath.HSum(acc);
            for (; i < n; i++) sum += (sbyte)q[i] * x[i];
            return sum;
        }
        float ss = 0;
        for (int i = 0; i < n; i++) ss += (sbyte)q[i] * x[i];
        return ss;
    }

    // ------------------------------- Q2_K: y = d*sc*q - dmin*m (bit-plane layout) -------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotQ2_K(byte* w, float* x, int cols)
    {
        int nb = cols / 256;
        float sum = 0f;
        Span<float> qf = Buf.Value; // 256 floats
        for (int b = 0; b < nb; b++, w += 84, x += 256)
        {
            float d = Dtypes.F16ToFloat((ushort)(w[80] | (w[81] << 8)));
            float dmin = Dtypes.F16ToFloat((ushort)(w[82] | (w[83] << 8)));
            byte* sc = w, q = w + 16;
            for (int sub = 0; sub < 16; sub++)
            {
                float a = d * (sc[sub] & 0xF);
                float m = dmin * (sc[sub] >> 4);
                int r = sub * 16;
                float dsub = 0;
                for (int i = 0; i < 16; i++)
                {
                    int wI = r + i;
                    int shift = 2 * ((wI % 128) / 32);
                    int qv = (q[32 * (wI / 128) + (wI % 128) % 32] >> shift) & 3;
                    dsub += qv * x[wI];
                }
                sum += a * dsub - m * SubSum(x + r, 16);
            }
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SubSum(float* x, int n)
    {
        float s = 0;
        for (int i = 0; i < n; i++) s += x[i];
        return s;
    }

    // ------------------------------- Q3_K: y = d*(sc-32)*q3 (bit-plane layout) -------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotQ3_K(byte* w, float* x, int cols)
    {
        int nb = cols / 256;
        float sum = 0f;
        Span<int> sc = stackalloc int[16];
        for (int b = 0; b < nb; b++, w += 110, x += 256)
        {
            float d = Dtypes.F16ToFloat((ushort)(w[108] | (w[109] << 8)));
            byte* hm = w, q = w + 32, packed = w + 96;
            uint a0 = (uint)(packed[0] | (packed[1] << 8) | (packed[2] << 16) | (packed[3] << 24));
            uint a1 = (uint)(packed[4] | (packed[5] << 8) | (packed[6] << 16) | (packed[7] << 24));
            uint a2 = (uint)(packed[8] | (packed[9] << 8) | (packed[10] << 16) | (packed[11] << 24));
            uint tmp = a2;
            uint r2 = ((a0 >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
            uint r3 = ((a1 >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
            uint r0 = (a0 & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);
            uint r1 = (a1 & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);
            sc[0] = (sbyte)(byte)r0 - 32; sc[1] = (sbyte)(byte)(r0 >> 8) - 32; sc[2] = (sbyte)(byte)(r0 >> 16) - 32; sc[3] = (sbyte)(byte)(r0 >> 24) - 32;
            sc[4] = (sbyte)(byte)r1 - 32; sc[5] = (sbyte)(byte)(r1 >> 8) - 32; sc[6] = (sbyte)(byte)(r1 >> 16) - 32; sc[7] = (sbyte)(byte)(r1 >> 24) - 32;
            sc[8] = (sbyte)(byte)r2 - 32; sc[9] = (sbyte)(byte)(r2 >> 8) - 32; sc[10] = (sbyte)(byte)(r2 >> 16) - 32; sc[11] = (sbyte)(byte)(r2 >> 24) - 32;
            sc[12] = (sbyte)(byte)r3 - 32; sc[13] = (sbyte)(byte)(r3 >> 8) - 32; sc[14] = (sbyte)(byte)(r3 >> 16) - 32; sc[15] = (sbyte)(byte)(r3 >> 24) - 32;
            for (int sub = 0; sub < 16; sub++)
            {
                float a = d * sc[sub];
                int r = sub * 16;
                float dsub = 0;
                for (int i = 0; i < 16; i++)
                {
                    int wI = r + i;
                    int shift = 2 * ((wI % 128) / 32);
                    int qv = (q[32 * (wI / 128) + (wI % 128) % 32] >> shift) & 3;
                    if ((hm[wI % 32] & (1 << (wI / 32))) == 0) qv -= 4;
                    dsub += qv * x[wI];
                }
                sum += a * dsub;
            }
        }
        return sum;
    }

    // ------------------------------- Q4_K: y = d*sc*q - dmin*m (nibble planes) -------------------------------
    // per 64 weights: 32 bytes; low nibbles -> weights 0..31 (scale pair is+0),
    // high nibbles -> weights 32..63 (scale pair is+1).

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotQ4_K(byte* w, float* x, int cols)
    {
        int nb = cols / 256;
        Span<float> lo = Buf.Value; Span<float> hi = Buf.Value.AsSpan(16);
        bool vec = Vector256.IsHardwareAccelerated;
        float sum = 0f;
        for (int b = 0; b < nb; b++, w += 144, x += 256)
        {
            float d = Dtypes.F16ToFloat((ushort)(w[0] | (w[1] << 8)));
            float dmin = Dtypes.F16ToFloat((ushort)(w[2] | (w[3] << 8)));
            byte* packed = w + 4, q = w + 16;
            for (int j = 0; j < 4; j++, q += 32)
            {
                // even sub-block: low nibbles
                BlockDequantizer.GetScaleMinK4(2 * j, packed, out int sc0, out int m0);
                float a0 = d * sc0, mm0 = dmin * m0;
                ReadOnlySpan<byte> qs0 = new(q, 16);
                ReadOnlySpan<byte> qs1 = new(q + 16, 16);
                SimdMath.UnpackLo16(qs0, lo);
                SimdMath.UnpackLo16(qs1, hi);
                sum += a0 * (VecDot16(lo, x + j * 64, vec) + VecDot16(hi, x + j * 64 + 16, vec))
                     - mm0 * SubSum(x + j * 64, 32);
                // odd sub-block: high nibbles
                BlockDequantizer.GetScaleMinK4(2 * j + 1, packed, out int sc1, out int m1);
                float a1 = d * sc1, mm1 = dmin * m1;
                SimdMath.UnpackHi16(qs0, lo);
                SimdMath.UnpackHi16(qs1, hi);
                sum += a1 * (VecDot16(lo, x + j * 64 + 32, vec) + VecDot16(hi, x + j * 64 + 48, vec))
                     - mm1 * SubSum(x + j * 64 + 32, 32);
            }
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float VecDot16(Span<float> q, float* x, bool vec)
    {
        if (vec)
        {
            var vq0 = Vector256.LoadUnsafe(ref q[0]);
            var vq1 = Vector256.LoadUnsafe(ref q[8]);
            var acc = Vector256.Multiply(vq0, Vector256.LoadUnsafe(ref *x, 0));
            acc = Vector256.Add(acc, Vector256.Multiply(vq1, Vector256.LoadUnsafe(ref *x, 8)));
            return SimdMath.HSum(acc);
        }
        float s = 0;
        for (int i = 0; i < 16; i++) s += q[i] * x[i];
        return s;
    }

    // ------------------------------- Q5_K: y = d*sc*q - dmin*m -------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotQ5_K(byte* w, float* x, int cols)
    {
        int nb = cols / 256;
        float sum = 0f;
        for (int b = 0; b < nb; b++, w += 176, x += 256)
        {
            float d = Dtypes.F16ToFloat((ushort)(w[0] | (w[1] << 8)));
            float dmin = Dtypes.F16ToFloat((ushort)(w[2] | (w[3] << 8)));
            byte* packed = w + 4, ql = w + 16, qh = w + 144;
            for (int sub = 0; sub < 8; sub++)
            {
                BlockDequantizer.GetScaleMinK4(sub, packed, out int scI, out int mI);
                float a = d * scI, m = dmin * mI;
                int j = sub / 2;
                byte* q = ql + j * 32;
                int u1 = 1 << (2 * j), u2 = 2 << (2 * j);
                float dsub = 0, xsub = 0;
                int wBase = sub * 32;
                if ((sub & 1) == 0)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        int q5 = (q[i] & 0xF) + ((qh[i] & u1) != 0 ? 16 : 0);
                        dsub += q5 * x[wBase + i];
                        xsub += x[wBase + i];
                    }
                }
                else
                {
                    for (int i = 0; i < 32; i++)
                    {
                        int q5 = (q[i] >> 4) + ((qh[i] & u2) != 0 ? 16 : 0);
                        dsub += q5 * x[wBase + i];
                        xsub += x[wBase + i];
                    }
                }
                sum += a * dsub - m * xsub;
            }
        }
        return sum;
    }

    // ------------------------------- Q6_K: y = d*sc[is]*(q-32) -------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotQ6_K(byte* w, float* x, int cols)
    {
        int nb = cols / 256;
        float sum = 0f;
        for (int b = 0; b < nb; b++, w += 210, x += 256)
        {
            float d = Dtypes.F16ToFloat((ushort)(w[208] | (w[209] << 8)));
            byte* ql = w, qh = w + 128, sc = w + 192;
            float dot = 0;
            for (int n = 0; n < 256; n += 128)
            {
                int scBase = 8 * (n / 128);
                for (int l = 0; l < 32; l++)
                {
                    int isx = l / 16;
                    int q1 = (ql[n / 2 + l] & 0xF) | (((qh[n / 4 + l] >> 0) & 3) << 4);
                    int q2 = (ql[n / 2 + l + 32] & 0xF) | (((qh[n / 4 + l] >> 2) & 3) << 4);
                    int q3 = (ql[n / 2 + l] >> 4) | (((qh[n / 4 + l] >> 4) & 3) << 4);
                    int q4 = (ql[n / 2 + l + 32] >> 4) | (((qh[n / 4 + l] >> 6) & 3) << 4);
                    dot += (sbyte)sc[scBase + isx + 0] * (q1 - 32) * x[n + l];
                    dot += (sbyte)sc[scBase + isx + 2] * (q2 - 32) * x[n + l + 32];
                    dot += (sbyte)sc[scBase + isx + 4] * (q3 - 32) * x[n + l + 64];
                    dot += (sbyte)sc[scBase + isx + 6] * (q4 - 32) * x[n + l + 96];
                }
            }
            sum += d * dot;
        }
        return sum;
    }

    // ------------------------------- TRI: y = scale*code -------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotTri(byte* w, float* x, int cols)
    {
        int nb = cols / 64;
        float sum = 0f;
        Span<float> qf = Buf.Value;
        for (int b = 0; b < nb; b++, w += 18, x += 64)
        {
            float scale = Dtypes.F16ToFloat((ushort)(w[0] | (w[1] << 8)));
            ReadOnlySpan<byte> qs = new(w + 2, 16);
            SimdMath.UnpackTri(qs, 16, scale, qf);
            float dot = 0;
            for (int i = 0; i < 64; i++) dot += qf[i] * x[i];
            sum += dot;
        }
        return sum;
    }

    // ------------------------------- I8_BLK -------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotI8Blk(byte* w, float* x, int cols)
    {
        int nb = cols / 256;
        float sum = 0f;
        Span<float> qf = Buf.Value;
        for (int b = 0; b < nb; b++, w += 258, x += 256)
        {
            float d = Dtypes.F16ToFloat((ushort)(w[0] | (w[1] << 8)));
            float dot = 0;
            int i = 0;
            if (Vector256.IsHardwareAccelerated)
            {
                for (; i + 16 <= 256; i += 16)
                    dot += DotI8Bytes(w + 2 + i, x + i, 16, qf);
            }
            for (; i < 256; i++) dot += (sbyte)w[2 + i] * x[i];
            sum += d * dot;
        }
        return sum;
    }

    // ------------------------------- extreme low-bit: B1 / B1X / T1E(_LO) -------------------------------
    // B1/B1X use the ±1 identity: sum(s_i * x_i) with s = ±1 equals a plain dot of the
    // sign-expanded mask with x — expand to ±1 floats once, then reuse the SIMD f32 dot.
    // B1X adds k exact-value corrections on top of the binary base.
    // T1E decodes the rANS symbol stream per block (0/±alpha) and dots the result.

    private static readonly float[] SignLut = BuildSignLut();

    private static float[] BuildSignLut()
    {
        var lut = new float[256 * 8];
        for (int v = 0; v < 256; v++)
            for (int bit = 0; bit < 8; bit++)
                lut[v * 8 + bit] = (v & (1 << bit)) != 0 ? 1f : -1f;
        return lut;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotB1(byte* w, float* x, int cols)
    {
        int nb = cols / 256;
        float sum = 0f;
        Span<float> sfb = Buf.Value;
        fixed (float* sf = sfb)
        {
            for (int b = 0; b < nb; b++, w += 34, x += 256)
            {
                float a = Dtypes.F16ToFloat((ushort)(w[0] | (w[1] << 8)));
                ExpandSigns(w + 2, sf);
                sum += a * SimdMath.DotF32(sf, x, 256);
            }
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotB1X(byte* w, float* x, int cols)
    {
        int nb = cols / 256;
        float sum = 0f;
        Span<float> sfb = Buf.Value;
        fixed (float* sf = sfb)
        {
            for (int b = 0; b < nb; b++, w += 47, x += 256)
            {
                float a = Dtypes.F16ToFloat((ushort)(w[0] | (w[1] << 8)));
                ExpandSigns(w + 2, sf);
                sum += a * SimdMath.DotF32(sf, x, 256);
                int k = w[34];
                for (int e = 0; e < k; e++)
                {
                    int idx = w[35 + 3 * e];
                    float v = Dtypes.F16ToFloat((ushort)(w[35 + 3 * e + 1] | (w[35 + 3 * e + 2] << 8)));
                    float s = (w[2 + (idx >> 3)] & (1 << (idx & 7))) != 0 ? 1f : -1f;
                    sum += (v - a * s) * x[idx];
                }
            }
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ExpandSigns(byte* mask, float* d)
    {
        fixed (float* pl = SignLut)
        {
            for (int by = 0; by < 32; by++, d += 8)
            {
                float* row = &pl[mask[by] * 8];
                for (int i = 0; i < 8; i++) d[i] = row[i];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotT1E(byte* w, float* x, int cols, int streamBytes)
    {
        int nb = cols / 256;
        float sum = 0f;
        Span<byte> syms = stackalloc byte[256];
        Span<float> qf = Buf.Value;
        for (int b = 0; b < nb; b++, w += 6 + streamBytes, x += 256)
        {
            uint n0 = (uint)(w[0] | (w[1] << 8));
            uint nP = (uint)(w[2] | (w[3] << 8));
            float a = Dtypes.F16ToFloat((ushort)(w[4] | (w[5] << 8)));
            EntropyCoder.Decode3(new ReadOnlySpan<byte>(w + 6, streamBytes), 256, n0, nP, syms);
            float dot = 0;
            for (int i = 0; i < 256; i++)
            {
                byte t = syms[i];
                float v = t == 1 ? a : t == 2 ? -a : 0f;
                dot += v * x[i];
            }
            sum += dot;
        }
        return sum;
    }

    // ------------------------------- drivers -------------------------------

    /// <summary>y = W x for row-major W [rows, cols] stored as compressed blocks. Sequential.</summary>
    public static void MatVec(QuantFormat fmt, byte* w, long rows, long cols, float* x, float* y)
    {
        var (bs, bb) = Dtypes.BlockSpec(fmt);
        long rowBytes = (long)(cols / bs) * bb;
        for (long r = 0; r < rows; r++)
        {
            y[r] = DotRow(fmt, w + r * rowBytes, x, (int)cols);
        }
    }

    /// <summary>Parallel over rows. Safe: each row writes distinct y[r].</summary>
    public static void MatVecParallel(QuantFormat fmt, byte* w, long rows, long cols, float* x, float* y, int minRowsPerThread = 8)
    {
        var (bs, bb) = Dtypes.BlockSpec(fmt);
        long rowBytes = (long)(cols / bs) * bb;
        int threads = Math.Max(1, Environment.ProcessorCount - 1);
        if (rows < minRowsPerThread * 2L || rows * rowBytes < 1 << 20)
        {
            MatVec(fmt, w, rows, cols, x, y);
            return;
        }
        long chunk = (rows + threads - 1) / threads;
        long colsL = cols;
        System.Threading.Tasks.Parallel.For(0, threads, t =>
        {
            long r0 = t * chunk;
            long r1 = Math.Min(rows, r0 + chunk);
            for (long r = r0; r < r1; r++)
                y[r] = DotRow(fmt, w + r * rowBytes, x, (int)colsL);
        });
    }

    /// <summary>
    /// Chunked prefill: Y[T, rows] = X[T, cols] * W^T. Parallel over tokens (each thread
    /// reuses the same streamed weights — optimal cache behavior).
    /// </summary>
    public static void GEMM(QuantFormat fmt, byte* w, int tokens, long rows, long cols, float* x, float* y, bool parallel = true)
    {
        var (bs, bb) = Dtypes.BlockSpec(fmt);
        long rowBytes = (long)(cols / bs) * bb;
        int colsI = (int)cols;
        if (parallel && tokens > 1 && rows >= 32)
        {
            System.Threading.Tasks.Parallel.For(0, tokens, t =>
            {
                float* xt = x + (long)t * cols;
                float* yt = y + (long)t * rows;
                for (long r = 0; r < rows; r++)
                    yt[r] = DotRow(fmt, w + r * rowBytes, xt, colsI);
            });
        }
        else
        {
            for (int t = 0; t < tokens; t++)
            {
                float* xt = x + (long)t * cols;
                float* yt = y + (long)t * rows;
                for (long r = 0; r < rows; r++)
                    yt[r] = DotRow(fmt, w + r * rowBytes, xt, colsI);
            }
        }
    }
}
