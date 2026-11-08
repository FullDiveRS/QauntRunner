using System.Runtime.CompilerServices;

namespace QuantRunner.Core.Quantization;

/// <summary>
/// Quantizes float32 tensors into GGML-compatible block formats, plus QuantRunner-native
/// formats (TRI ternary 1.58-bit, I8_BLK). Block byte layouts are byte-identical to llama.cpp
/// (verified against ggml-quants.c), so GGUF k-quants can pass through during conversion.
///
/// Dequant formulas (decoder contract, matches ggml-quants.c):
///   Q4_0 : y = d * (q - 8)                 q:4-bit  32/block   20 B   d@0
///   Q8_0 : y = d * q                       q:8-bit  32/block   34 B   d@0
///   Q2_K : y = d*sc*q - dmin*m             q:2-bit  256/block  84 B   scales@0 qs@16 d@80 dmin@82
///   Q3_K : y = d*(sc-32)*q3                q:3-bit  256/block 110 B   hmask@0 qs@32 scales@96 d@108
///   Q4_K : y = d*sc*q - dmin*m             q:4-bit  256/block 144 B   d@0 dmin@2 scales@4 qs@16
///   Q5_K : y = d*sc*q - dmin*m             q:5-bit  256/block 176 B   d@0 dmin@2 scales@4 qs@16 qh@144
///   Q6_K : y = d*sc*(q-32)                 q:6-bit  256/block 210 B   ql@0 qh@128 scales@192 d@208
///   TRI  : y = scale*code                  2-bit code 00=0 01=+1 10=-1  64/block 18 B
///   I8_BLK: y = d*q                        q:8-bit  256/block 258 B
/// </summary>
public static unsafe class BlockQuantizer
{
    public static void Quantize(QuantFormat fmt, ReadOnlySpan<float> src, Span<byte> dst)
    {
        switch (fmt)
        {
            case QuantFormat.F32: src.CopyTo(System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(dst)); break;
            case QuantFormat.F16: QuantizeF16(src, dst); break;
            case QuantFormat.BF16: QuantizeBf16(src, dst); break;
            case QuantFormat.Q4_0: QuantizeQ4_0(src, dst); break;
            case QuantFormat.Q8_0: QuantizeQ8_0(src, dst); break;
            case QuantFormat.Q2_K: QuantizeQ2_K(src, dst); break;
            case QuantFormat.Q3_K: QuantizeQ3_K(src, dst); break;
            case QuantFormat.Q4_K: QuantizeQ4_K(src, dst); break;
            case QuantFormat.Q5_K: QuantizeQ5_K(src, dst); break;
            case QuantFormat.Q6_K: QuantizeQ6_K(src, dst); break;
            case QuantFormat.TRI: QuantizeTri(src, dst); break;
            case QuantFormat.I8_BLK: QuantizeI8Blk(src, dst); break;
            case QuantFormat.B1: QuantizeB1(src, dst); break;
            case QuantFormat.B1X: QuantizeB1X(src, dst); break;
            case QuantFormat.T1E: QuantizeT1E(src, dst, 34); break;
            case QuantFormat.T1E_LO: QuantizeT1E(src, dst, 26); break;
            default: throw new NotSupportedException(fmt.ToString());
        }
    }

    // ------------------------------ fp formats ------------------------------

    public static void QuantizeF16(ReadOnlySpan<float> src, Span<byte> dst)
    {
        fixed (byte* pd = dst)
        {
            ushort* o = (ushort*)pd;
            for (int i = 0; i < src.Length; i++) o[i] = Dtypes.FloatToF16(src[i]);
        }
    }

    public static void QuantizeBf16(ReadOnlySpan<float> src, Span<byte> dst)
    {
        fixed (byte* pd = dst)
        {
            ushort* o = (ushort*)pd;
            for (int i = 0; i < src.Length; i++) o[i] = Dtypes.FloatToBf16(src[i]);
        }
    }

    // ------------------------------ Q4_0 (32/block, 20B) ------------------------------

    public static void QuantizeQ4_0(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int nb = src.Length / 32;
        fixed (float* px = src)
        fixed (byte* pd = dst)
        {
            float* x = px; byte* y = pd;
            for (int b = 0; b < nb; b++, x += 32, y += 20)
            {
                float amax = 0f;
                for (int i = 0; i < 32; i++) { float a = MathF.Abs(x[i]); if (a > amax) amax = a; }
                float d = amax / 8f;
                ushort dh = Dtypes.FloatToF16(d);
                y[0] = (byte)(dh & 0xFF); y[1] = (byte)(dh >> 8);
                if (d == 0f) { for (int i = 0; i < 16; i++) y[2 + i] = 8; continue; } // q=8 -> 0
                float id = 1f / d;
                for (int i = 0; i < 16; i++)
                {
                    int q0 = ClampQ((int)MathF.Round(x[i] * id) + 8, 0, 15);
                    int q1 = ClampQ((int)MathF.Round(x[i + 16] * id) + 8, 0, 15);
                    y[2 + i] = (byte)(q0 | (q1 << 4));
                }
            }
        }
    }

    // ------------------------------ Q8_0 (32/block, 34B) ------------------------------

    public static void QuantizeQ8_0(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int nb = src.Length / 32;
        fixed (float* px = src)
        fixed (byte* pd = dst)
        {
            float* x = px; byte* y = pd;
            for (int b = 0; b < nb; b++, x += 32, y += 34)
            {
                float amax = 0f;
                for (int i = 0; i < 32; i++) { float a = MathF.Abs(x[i]); if (a > amax) amax = a; }
                float d = amax / 127f;
                ushort dh = Dtypes.FloatToF16(d);
                y[0] = (byte)(dh & 0xFF); y[1] = (byte)(dh >> 8);
                if (d == 0f) { for (int i = 0; i < 32; i++) y[2 + i] = 0; continue; }
                float id = 1f / d;
                for (int i = 0; i < 32; i++) y[2 + i] = (byte)ClampQ((int)MathF.Round(x[i] * id), -128, 127);
            }
        }
    }

    // ------------------------------ Q2_K (256/block, 84B) ------------------------------
    // Layout: scales[16] @0, qs[64] @16, d @80, dmin @82.
    // scales[j] = scale(4b) | min<<4 ; d = max_scale/15, dmin = max_min/15.
    // qs bit-planes: byte 32*(w/128) + (w%128)%32, shift 2*((w%128)/32).

    public static void QuantizeQ2_K(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int nb = src.Length / 256;
        fixed (float* px = src)
        fixed (byte* pd = dst)
        {
            float* x = px; byte* y = pd;
            Span<float> mins = stackalloc float[16], maxs = stackalloc float[16];
            Span<int> L = stackalloc int[256];
            for (int b = 0; b < nb; b++, x += 256, y += 84)
            {
                float amax = 0f, amink = 0f;
                for (int s = 0; s < 16; s++)
                {
                    float mn = float.MaxValue, mx = float.MinValue;
                    for (int i = 0; i < 16; i++)
                    {
                        float v = x[s * 16 + i];
                        if (v < mn) mn = v; if (v > mx) mx = v;
                    }
                    if (mn > 0) mn = 0;
                    mins[s] = mn; maxs[s] = mx;
                    float range = mx - mn;
                    if (range > amax) amax = range;
                    if (-mn > amink) amink = -mn;
                }
                for (int i = 0; i < 84; i++) y[i] = 0;
                if (amax <= 0f) continue;
                float d = amax / 45f, dmin = amink > 0 ? amink / 15f : 0f;   // 3 levels x 15 max scale
                ushort dh = Dtypes.FloatToF16(d), dmh = Dtypes.FloatToF16(dmin);
                y[80] = (byte)dh; y[81] = (byte)(dh >> 8);
                y[82] = (byte)dmh; y[83] = (byte)(dmh >> 8);
                float iscale = 15f / amax, ismin = dmin > 0 ? 15f / amink : 0f;
                for (int s = 0; s < 16; s++)
                {
                    int sc = ClampQ((int)MathF.Round(iscale * (maxs[s] - mins[s])), 0, 15);
                    int m = dmin > 0 ? ClampQ((int)MathF.Round(ismin * (-mins[s])), 0, 15) : 0;
                    y[s] = (byte)(sc | (m << 4));
                    float a = d * sc, bias = dmin * m;
                    if (a <= 0f) continue;
                    for (int i = 0; i < 16; i++)
                        L[s * 16 + i] = ClampQ((int)MathF.Round((x[s * 16 + i] + bias) / a), 0, 3);
                }
                for (int j = 0; j < 256; j += 128)
                    for (int l = 0; l < 32; l++)
                        y[16 + j / 4 + l] = (byte)(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6));
            }
        }
    }

    // ------------------------------ Q3_K (256/block, 110B) ------------------------------
    // Layout: hmask[32] @0, qs[64] @32, scales[12] @96, d @108.
    // Reference scheme incl. the sign-flip trick (iscale = -32/max_scale):
    //   effective subblock scale = d * (sc6 - 32);  L = clamp(round(x/that), -4, 3) + 4 (0..7)
    //   hmask byte = w%32, bit = w/32 (bit set -> value in 0..3, else value-4)
    //   qs byte 32*(w/128) + (w%128)%32, shift 2*((w%128)/32)

    public static void QuantizeQ3_K(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int nb = src.Length / 256;
        fixed (float* px = src)
        fixed (byte* pd = dst)
        {
            float* x = px; byte* y = pd;
            Span<float> subScale = stackalloc float[16];
            Span<int> L = stackalloc int[256];
            Span<int> sc6 = stackalloc int[16];
            for (int b = 0; b < nb; b++, x += 256, y += 110)
            {
                float maxScale = 0f, maxAbs = 0f;
                float signedMax = 0f;
                for (int s = 0; s < 16; s++)
                {
                    float mx = 0f;
                    for (int i = 0; i < 16; i++) { float a = MathF.Abs(x[s * 16 + i]); if (a > mx) mx = a; }
                    subScale[s] = mx / 4f;
                    if (subScale[s] > maxAbs) { maxAbs = subScale[s]; signedMax = subScale[s]; }
                }
                maxScale = signedMax;
                for (int i = 0; i < 110; i++) y[i] = 0;
                if (maxScale <= 0f) continue;
                float iscale = -32f / maxScale;
                float d = 1f / iscale;                       // negative d (sign-flip trick)
                ushort dh = Dtypes.FloatToF16(d);
                y[108] = (byte)dh; y[109] = (byte)(dh >> 8);

                for (int s = 0; s < 16; s++)
                {
                    int l = ClampQ((int)MathF.Round(iscale * subScale[s]), -32, 31) + 32;  // 0..63
                    sc6[s] = l;
                    if (s < 8) y[96 + s] = (byte)(l & 0xF);
                    else y[96 + s - 8] |= (byte)((l & 0xF) << 4);
                    int hi = l >> 4;                          // 2 bits
                    y[96 + s % 4 + 8] |= (byte)(hi << (2 * (s / 4)));
                }
                // requantize with final effective scales
                for (int s = 0; s < 16; s++)
                {
                    int sc = sc6[s] - 32;
                    float a = d * sc;
                    if (a == 0f) continue;
                    for (int i = 0; i < 16; i++)
                        L[s * 16 + i] = ClampQ((int)MathF.Round(x[s * 16 + i] / a), -4, 3) + 4; // 0..7
                }
                // hmask: byte = w%32, bit = w/32 ; set when L > 3
                for (int w = 0; w < 256; w++)
                {
                    if (L[w] > 3)
                    {
                        y[w % 32] |= (byte)(1 << (w / 32));
                        L[w] -= 4;
                    }
                }
                // qs bit-planes
                for (int j = 0; j < 256; j += 128)
                    for (int l = 0; l < 32; l++)
                        y[32 + j / 4 + l] = (byte)(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6));
            }
        }
    }

    // ------------------------------ Q4_K (256/block, 144B) ------------------------------
    // Layout: d @0, dmin @2, scales[12] @4 (k4 packing), qs[128] @16.
    // y = d*sc*q - dmin*m ; sc/m 6-bit via get_scale_min_k4.

    public static void QuantizeQ4_K(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int nb = src.Length / 256;
        fixed (float* px = src)
        fixed (byte* pd = dst)
        {
            float* x = px; byte* y = pd;
            Span<float> mins = stackalloc float[8], maxs = stackalloc float[8];
            Span<int> L = stackalloc int[256];
            Span<int> sc = stackalloc int[8], m = stackalloc int[8];
            Span<byte> q = stackalloc byte[12];
            for (int b = 0; b < nb; b++, x += 256, y += 144)
            {
                float amax = 0f, amink = 0f;
                for (int s = 0; s < 8; s++)
                {
                    float mn = float.MaxValue, mx = float.MinValue;
                    for (int i = 0; i < 32; i++)
                    {
                        float v = x[s * 32 + i];
                        if (v < mn) mn = v; if (v > mx) mx = v;
                    }
                    if (mn > 0) mn = 0;
                    mins[s] = mn; maxs[s] = mx;
                    float range = mx - mn;
                    if (range > amax) amax = range;
                    if (-mn > amink) amink = -mn;
                }
                for (int i = 0; i < 144; i++) y[i] = 0;
                if (amax <= 0f) continue;
                float d = amax / 945f, dmin = amink > 0 ? amink / 63f : 0f;  // 15 levels x 63 max scale
                ushort dh = Dtypes.FloatToF16(d), dmh = Dtypes.FloatToF16(dmin);
                y[0] = (byte)dh; y[1] = (byte)(dh >> 8);
                y[2] = (byte)dmh; y[3] = (byte)(dmh >> 8);
                float iscale = 63f / amax, ismin = dmin > 0 ? 63f / amink : 0f;
                for (int s = 0; s < 8; s++)
                {
                    sc[s] = ClampQ((int)MathF.Round(iscale * (maxs[s] - mins[s])), 0, 63);
                    m[s] = dmin > 0 ? ClampQ((int)MathF.Round(ismin * (-mins[s])), 0, 63) : 0;
                }
                // k4 packing (exact inverse of get_scale_min_k4)
                for (int j = 0; j < 4; j++)
                {
                    q[j] = (byte)((sc[j] & 63) | ((sc[j + 4] >> 4) << 6));
                    q[j + 4] = (byte)((m[j] & 63) | ((m[j + 4] >> 4) << 6));
                    q[j + 8] = (byte)((sc[j + 4] & 0xF) | ((m[j + 4] & 0xF) << 4));
                }
                for (int i = 0; i < 12; i++) y[4 + i] = q[i];

                for (int s = 0; s < 8; s++)
                {
                    float a = d * sc[s], bias = dmin * m[s];
                    if (a <= 0f) continue;
                    for (int i = 0; i < 32; i++)
                        L[s * 32 + i] = ClampQ((int)MathF.Round((x[s * 32 + i] + bias) / a), 0, 15);
                }
                // nibbles: weights 0..31 low nibbles of bytes 0..31, 32..63 high, per 64-group
                for (int s = 0; s < 8; s++)
                {
                    int baseByte = 16 + (s / 2) * 32;
                    int nibbleHi = s & 1;
                    for (int i = 0; i < 32; i++)
                    {
                        int val = L[s * 32 + i];
                        if (nibbleHi == 0) y[baseByte + i] |= (byte)val;
                        else y[baseByte + i] |= (byte)(val << 4);
                    }
                }
            }
        }
    }

    // ------------------------------ Q5_K (256/block, 176B) ------------------------------
    // Layout: d @0, dmin @2, scales[12] @4, qs[128] @16, qh[32] @144.
    // y = d*sc*q - dmin*m, q 0..31 (4 bits in qs + 5th bit in qh).

    public static void QuantizeQ5_K(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int nb = src.Length / 256;
        fixed (float* px = src)
        fixed (byte* pd = dst)
        {
            float* x = px; byte* y = pd;
            Span<float> mins = stackalloc float[8], maxs = stackalloc float[8];
            Span<int> L = stackalloc int[256];
            Span<int> sc = stackalloc int[8], m = stackalloc int[8];
            Span<byte> q = stackalloc byte[12];
            for (int b = 0; b < nb; b++, x += 256, y += 176)
            {
                float amax = 0f, amink = 0f;
                for (int s = 0; s < 8; s++)
                {
                    float mn = float.MaxValue, mx = float.MinValue;
                    for (int i = 0; i < 32; i++)
                    {
                        float v = x[s * 32 + i];
                        if (v < mn) mn = v; if (v > mx) mx = v;
                    }
                    if (mn > 0) mn = 0;
                    mins[s] = mn; maxs[s] = mx;
                    float range = mx - mn;
                    if (range > amax) amax = range;
                    if (-mn > amink) amink = -mn;
                }
                for (int i = 0; i < 176; i++) y[i] = 0;
                if (amax <= 0f) continue;
                float d = amax / 1953f, dmin = amink > 0 ? amink / 63f : 0f;  // 31 levels x 63 max scale
                ushort dh = Dtypes.FloatToF16(d), dmh = Dtypes.FloatToF16(dmin);
                y[0] = (byte)dh; y[1] = (byte)(dh >> 8);
                y[2] = (byte)dmh; y[3] = (byte)(dmh >> 8);
                float iscale = 63f / amax, ismin = dmin > 0 ? 63f / amink : 0f;
                for (int s = 0; s < 8; s++)
                {
                    sc[s] = ClampQ((int)MathF.Round(iscale * (maxs[s] - mins[s])), 0, 63);
                    m[s] = dmin > 0 ? ClampQ((int)MathF.Round(ismin * (-mins[s])), 0, 63) : 0;
                }
                for (int j = 0; j < 4; j++)
                {
                    q[j] = (byte)((sc[j] & 63) | ((sc[j + 4] >> 4) << 6));
                    q[j + 4] = (byte)((m[j] & 63) | ((m[j + 4] >> 4) << 6));
                    q[j + 8] = (byte)((sc[j + 4] & 0xF) | ((m[j + 4] & 0xF) << 4));
                }
                for (int i = 0; i < 12; i++) y[4 + i] = q[i];

                for (int s = 0; s < 8; s++)
                {
                    float a = d * sc[s], bias = dmin * m[s];
                    if (a <= 0f) continue;
                    for (int i = 0; i < 32; i++)
                        L[s * 32 + i] = ClampQ((int)MathF.Round((x[s * 32 + i] + bias) / a), 0, 31);
                }
                for (int w = 0; w < 256; w++)
                {
                    int s = w / 32;
                    int r = w % 64;
                    int half = r / 32, l = r % 32;
                    int qlByte = 16 + 32 * (w / 64) + l;
                    int lo = L[w] & 0xF, bit5 = (L[w] >> 4) & 1;
                    if (half == 0) y[qlByte] |= (byte)lo;
                    else y[qlByte] |= (byte)(lo << 4);
                    if (bit5 != 0) y[144 + l] |= (byte)(1 << (2 * (w / 64) + half));
                }
            }
        }
    }

    // ------------------------------ Q6_K (256/block, 210B) ------------------------------
    // Layout: ql[128] @0, qh[64] @128, scales[16] @192 (int8), d @208.
    // y = d * sc[is] * (q-32); ql/qh bit layout verified vs dequantize_row_q6_K.

    public static void QuantizeQ6_K(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int nb = src.Length / 256;
        fixed (float* px = src)
        fixed (byte* pd = dst)
        {
            float* x = px; byte* y = pd;
            Span<float> subAmax = stackalloc float[16];
            Span<int> sc = stackalloc int[16];
            for (int b = 0; b < nb; b++, x += 256, y += 210)
            {
                float amax = 0f;
                for (int s = 0; s < 16; s++)
                {
                    float mx = 0f;
                    for (int i = 0; i < 16; i++) { float a = MathF.Abs(x[s * 16 + i]); if (a > mx) mx = a; }
                    subAmax[s] = mx; if (mx > amax) amax = mx;
                }
                for (int i = 0; i < 210; i++) y[i] = 0;
                if (amax <= 0f) continue;
                float d = amax / (31f * 31f);   // 31 levels x 31 max scale
                ushort dh = Dtypes.FloatToF16(d);
                y[208] = (byte)dh; y[209] = (byte)(dh >> 8);
                float id = 1f / d;
                for (int s = 0; s < 16; s++)
                    sc[s] = ClampQ((int)MathF.Round(subAmax[s] / (31f * d)), subAmax[s] > 0 ? 1 : 0, 127);

                for (int w = 0; w < 256; w++)
                {
                    int s = w / 16;
                    float a = d * sc[s];
                    int q = a > 0 ? ClampQ((int)MathF.Round(x[w] / a) + 32, 0, 63) : 32;
                    int lo = q & 0xF, hi = (q >> 4) & 3;
                    int n = (w / 128) * 128;
                    int r = w - n;
                    int slot = r / 32, l = r % 32;
                    switch (slot)
                    {
                        case 0: y[n / 2 + l] |= (byte)lo; break;
                        case 1: y[n / 2 + l + 32] |= (byte)lo; break;
                        case 2: y[n / 2 + l] |= (byte)(lo << 4); break;
                        case 3: y[n / 2 + l + 32] |= (byte)(lo << 4); break;
                    }
                    y[128 + n / 4 + l] |= (byte)(hi << (2 * slot));
                }
                for (int s = 0; s < 16; s++) y[192 + s] = (byte)(sbyte)sc[s];
            }
        }
    }

    // ------------------------------ TRI ternary (64/block, 18B) ------------------------------
    // Optimal-threshold ternarization: instead of the legacy fixed heuristic (scale = mean|w|,
    // keep |w| >= scale/2), search a grid of magnitude thresholds (quantiles of |w| plus the
    // legacy boundary) and for each assignment use the closed-form L2-optimal scale
    //   alpha* = sum(|w| over kept) / count(kept)
    // (d/dα ||w − α·T||² = 0 for fixed ternary pattern T). Layout unchanged — a pure quality
    // upgrade at identical bits; the legacy quantizer is kept for A/B selftests.

    public static void QuantizeTri(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int nb = src.Length / 64;
        fixed (float* px = src)
        fixed (byte* pd = dst)
        {
            float* x = px; byte* y = pd;
            Span<float> sorted = stackalloc float[64];
            for (int b = 0; b < nb; b++, x += 64, y += 18)
            {
                for (int i = 0; i < 18; i++) y[i] = 0;
                double energy = 0;
                float amax = 0f;
                for (int i = 0; i < 64; i++)
                {
                    float a = MathF.Abs(x[i]);
                    sorted[i] = a;
                    energy += a * (double)a;
                    if (a > amax) amax = a;
                }
                if (amax == 0f) continue;
                sorted.Sort();

                double bestErr = double.MaxValue;
                float bestT = 0f, bestAlpha = amax;
                // quantile grid: t = sorted[j] -> keep weights with |w| > t (kept <= 64 - j)
                for (int j = 0; j <= 60; j += 4)
                {
                    float t = sorted[j];
                    double sum = 0; int kept = 0;
                    for (int i = 0; i < 64; i++)
                        if (MathF.Abs(x[i]) > t) { sum += MathF.Abs(x[i]); kept++; }
                    if (kept == 0) continue;
                    float alpha = (float)(sum / kept);
                    // err = energy - 2*alpha*sum + alpha^2 * kept
                    double err = energy - 2.0 * alpha * sum + (double)alpha * alpha * kept;
                    if (err < bestErr) { bestErr = err; bestT = t; bestAlpha = alpha; }
                }
                // legacy boundary as an extra candidate: keep |w| >= mean|w|/2 (the round-to-
                // nearest set of the old heuristic), scale refit -> guarantees new <= legacy error
                {
                    double sumAll = 0;
                    for (int i = 0; i < 64; i++) sumAll += MathF.Abs(x[i]);
                    float tLegacy = (float)(sumAll / 64) * 0.5f;
                    double sum = 0; int kept = 0;
                    for (int i = 0; i < 64; i++)
                        if (MathF.Abs(x[i]) >= tLegacy) { sum += MathF.Abs(x[i]); kept++; }
                    if (kept > 0)
                    {
                        float alpha = (float)(sum / kept);
                        double err = energy - 2.0 * alpha * sum + (double)alpha * alpha * kept;
                        if (err < bestErr) { bestErr = err; bestT = tLegacy - 1e-9f; bestAlpha = alpha; }
                    }
                }

                ushort sh = Dtypes.FloatToF16(bestAlpha);
                y[0] = (byte)sh; y[1] = (byte)(sh >> 8);
                for (int i = 0; i < 64; i++)
                {
                    float a = MathF.Abs(x[i]);
                    int code = a > bestT ? (x[i] > 0 ? 1 : 2) : 0;
                    y[2 + i / 4] |= (byte)(code << (2 * (i & 3)));
                }
            }
        }
    }

    /// <summary>The pre-upgrade ternary heuristic (scale = mean|w|, round-to-nearest codes).
    /// Kept only for A/B quality selftests; layout is identical to QuantizeTri.</summary>
    public static void QuantizeTriLegacy(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int nb = src.Length / 64;
        fixed (float* px = src)
        fixed (byte* pd = dst)
        {
            float* x = px; byte* y = pd;
            for (int b = 0; b < nb; b++, x += 64, y += 18)
            {
                for (int i = 0; i < 18; i++) y[i] = 0;
                double sum = 0;
                for (int i = 0; i < 64; i++) sum += MathF.Abs(x[i]);
                float scale = (float)(sum / 64);
                if (scale == 0f) continue;
                ushort sh = Dtypes.FloatToF16(scale);
                y[0] = (byte)sh; y[1] = (byte)(sh >> 8);
                float iscale = 1f / scale;
                for (int i = 0; i < 64; i++)
                {
                    int v = (int)MathF.Round(x[i] * iscale);
                    int code = v > 0 ? 1 : v < 0 ? 2 : 0;
                    y[2 + i / 4] |= (byte)(code << (2 * (i & 3)));
                }
            }
        }
    }

    // ------------------------------ extreme low-bit formats ------------------------------
    // All three encode 256-weight blocks and were designed around the same observation as
    // BiLLM / OneBit (ICML / NeurIPS research on ~1-bit post-training quantization): the
    // error of a 1-bit fit is dominated by two tails — the few huge outlier weights the
    // sign pattern cannot reach, and the mass of near-zero weights that binary distorts
    // into ±alpha. Different formats attack different tails:
    //
    //   B1     — pure binary: y = ±alpha, alpha* = mean(|w|) (closed-form L2 optimum).
    //            The floor everyone falls back to: 1.0625 bpw stored.
    //   B1X    — binary + up to 4 exact fp16 exceptions chosen by greedy residual matching
    //            (each round pulls the single worst weight out and refits alpha on the rest,
    //            a block-matched-pursuit step). Kills the outlier tail: 1.469 bpw stored.
    //   T1E(_LO) — sparsity-controlled ternary + rANS entropy coding. Ternarization with a
    //            controllable zero fraction codes to its Shannon entropy instead of flat
    //            2-bit packing, which is what breaks the 1.58-bit floor: 1.25 / 1.00 bpw.
    //
    // Which wins depends on the weight distribution (heavy-tailed real weights favor B1X /
    // T1E over B1; near-Gaussian blocks favor B1) — the `analyze` command measures it
    // per tensor so the choice can be made from data.

    public static void QuantizeB1(ReadOnlySpan<float> src, Span<byte> dst)
    {
        QuantizeBinary(src, dst, maxExceptions: 0);
    }

    public static void QuantizeB1X(ReadOnlySpan<float> src, Span<byte> dst)
    {
        QuantizeBinary(src, dst, maxExceptions: 4);
    }

    // layout (both): [0..1] f16 alpha, [2..33] 256-bit sign mask (bit set -> +alpha),
    // B1X only: [34] exception count k, [35..34+3k) k x (u8 index, f16 exact value).
    private static void QuantizeBinary(ReadOnlySpan<float> src, Span<byte> dst, int maxExceptions)
    {
        const int bb = 34, bbx = 47;
        int blockBytes = maxExceptions > 0 ? bbx : bb;
        int nb = src.Length / 256;
        fixed (float* px = src)
        fixed (byte* pd = dst)
        {
            float* x = px; byte* y = pd;
            Span<bool> exempt = stackalloc bool[256];
            for (int b = 0; b < nb; b++, x += 256, y += blockBytes)
            {
                for (int i = 0; i < blockBytes; i++) y[i] = 0;
                double sum = 0;
                for (int i = 0; i < 256; i++) sum += MathF.Abs(x[i]);
                float alpha = (float)(sum / 256);
                if (alpha > 0f || maxExceptions > 0)
                {
                    for (int i = 0; i < 256; i++)
                        if (x[i] >= 0f)
                            y[2 + (i >> 3)] |= (byte)(1 << (i & 7));
                }
                if (alpha == 0f && maxExceptions == 0) continue;

                // greedy residual matching: pull the worst |w| - alpha deviation out of the
                // binary fit and store it exactly; refit alpha over the remaining weights.
                for (int i = 0; i < 256; i++) exempt[i] = false;
                int k = 0, live = 256;
                double liveSum = sum;
                for (int e = 0; e < maxExceptions; e++)
                {
                    if (live <= 0) break;
                    alpha = (float)(liveSum / live);
                    int worst = -1; float worstR = 0f;
                    for (int i = 0; i < 256; i++)
                    {
                        if (exempt[i]) continue;
                        float r = MathF.Abs(MathF.Abs(x[i]) - alpha);
                        if (r > worstR) { worstR = r; worst = i; }
                    }
                    if (worst < 0) break;
                    exempt[worst] = true;
                    live--; liveSum -= MathF.Abs(x[worst]);
                    int off = 35 + 3 * e;
                    y[off] = (byte)worst;
                    ushort vh = Dtypes.FloatToF16(x[worst]);
                    y[off + 1] = (byte)vh; y[off + 2] = (byte)(vh >> 8);
                    k = e + 1;
                }
                if (maxExceptions > 0) y[34] = (byte)k;
                // final alpha over the weights that stay binary
                alpha = live > 0 ? (float)(liveSum / live) : 0f;
                ushort ah = Dtypes.FloatToF16(alpha);
                y[0] = (byte)ah; y[1] = (byte)(ah >> 8);
            }
        }
    }

    // layout: [0..1] u16 zero count n0, [2..3] u16 plus count nP, [4..5] f16 alpha,
    //         [6 .. 6+streamBytes) rANS stream (zero padded). minus count = 256-n0-nP.
    // Rate control: try the error-optimal zero fraction first; when the entropy-coded
    // stream does not fit the block budget, escalate the zero fraction (quantile grid)
    // until it does. Kept weights use the closed-form optimal scale for their set.
    private static void QuantizeT1E(ReadOnlySpan<float> src, Span<byte> dst, int streamBytes)
    {
        const int bb = 40, bbLo = 32;
        int blockBytes = 6 + streamBytes;
        int nb = src.Length / 256;
        fixed (float* px = src)
        fixed (byte* pd = dst)
        {
            float* x = px; byte* y = pd;
            Span<float> sorted = stackalloc float[256];
            Span<byte> sym = stackalloc byte[256];
            Span<byte> stream = stackalloc byte[64];
            // quantile grid: thresholds at these rank positions of sorted |w| (asc).
            Span<int> ranks = stackalloc int[] { 0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 248, 252, 255 };
            for (int b = 0; b < nb; b++, x += 256, y += blockBytes)
            {
                for (int i = 0; i < blockBytes; i++) y[i] = 0;
                double energy = 0;
                float amax = 0f;
                for (int i = 0; i < 256; i++)
                {
                    float a = MathF.Abs(x[i]);
                    sorted[i] = a;
                    energy += a * (double)a;
                    if (a > amax) amax = a;
                }
                if (amax == 0f) continue;
                sorted.Sort();

                // error-optimal starting threshold (same objective as TRI, on 256 weights)
                double bestErr = double.MaxValue;
                int bestRank = 0;
                foreach (int r in ranks)
                {
                    if (r >= 255) continue;
                    float t = sorted[r];
                    double sum = 0; int kept = 0;
                    for (int i = 0; i < 256; i++)
                        if (MathF.Abs(x[i]) > t) { sum += MathF.Abs(x[i]); kept++; }
                    if (kept == 0) continue;
                    float alpha = (float)(sum / kept);
                    double err = energy - 2.0 * alpha * sum + (double)alpha * alpha * kept;
                    if (err < bestErr) { bestErr = err; bestRank = r; }
                }

                // rate control: walk the grid upward from the error optimum (increasing
                // zero fraction) until the entropy-coded stream fits the block budget.
                bool start = true;
                for (int gi = IndexOf(ranks, bestRank); gi < ranks.Length; gi++)
                {
                    int r = ranks[gi];
                    if (!start && r < bestRank) continue;
                    start = false;
                    float t = r == 0 ? -1f : sorted[r];
                    int n0 = 0, nP = 0;
                    double sum = 0;
                    for (int i = 0; i < 256; i++)
                    {
                        if (MathF.Abs(x[i]) > t)
                        {
                            sum += MathF.Abs(x[i]);
                            if (x[i] > 0) { sym[i] = 1; nP++; } else { sym[i] = 2; }
                        }
                        else { sym[i] = 0; n0++; }
                    }
                    int kept = 256 - n0;
                    float alpha = kept > 0 ? (float)(sum / kept) : 0f;
                    int used = EntropyCoder.Encode3(sym, (uint)n0, (uint)nP, stream);
                    if (used >= 0 && used <= streamBytes)
                    {
                        y[0] = (byte)n0; y[1] = (byte)(n0 >> 8);
                        y[2] = (byte)nP; y[3] = (byte)(nP >> 8);
                        ushort ah = Dtypes.FloatToF16(alpha);
                        y[4] = (byte)ah; y[5] = (byte)(ah >> 8);
                        for (int i = 0; i < used; i++) y[6 + i] = stream[i];
                        break;
                    }
                    if (r == 255) throw new InvalidDataException("T1E: even the all-zero encoding overflowed (bug)");
                }
            }
        }
    }

    private static int IndexOf(Span<int> s, int v)
    {
        for (int i = 0; i < s.Length; i++) if (s[i] == v) return i;
        return 0;
    }

    // ------------------------------ I8_BLK (256/block, 258B) ------------------------------

    public static void QuantizeI8Blk(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int nb = src.Length / 256;
        fixed (float* px = src)
        fixed (byte* pd = dst)
        {
            float* x = px; byte* y = pd;
            for (int b = 0; b < nb; b++, x += 256, y += 258)
            {
                float amax = 0f;
                for (int i = 0; i < 256; i++) { float a = MathF.Abs(x[i]); if (a > amax) amax = a; }
                float d = amax / 127f;
                ushort dh = Dtypes.FloatToF16(d);
                y[0] = (byte)dh; y[1] = (byte)(dh >> 8);
                if (d == 0f) { for (int i = 2; i < 258; i++) y[i] = 0; continue; }
                float id = 1f / d;
                for (int i = 0; i < 256; i++) y[2 + i] = (byte)(sbyte)ClampQ((int)MathF.Round(x[i] * id), -128, 127);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ClampQ(int v, int lo, int hi) => v < lo ? lo : v > hi ? hi : v;
}
