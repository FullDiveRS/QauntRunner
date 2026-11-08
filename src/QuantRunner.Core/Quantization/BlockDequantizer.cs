using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace QuantRunner.Core.Quantization;

/// <summary>Decodes all supported block formats into float32. Mirrors ggml-quants.c layouts exactly.</summary>
public static unsafe class BlockDequantizer
{
    /// <summary>Dequantize a whole tensor (n elements = src block-aligned) into dst.</summary>
    public static void Dequantize(QuantFormat fmt, ReadOnlySpan<byte> src, Span<float> dst)
    {
        switch (fmt)
        {
            case QuantFormat.F32: MemoryMarshal.Cast<byte, float>(src).CopyTo(dst); break;
            case QuantFormat.F16: DequantF16(src, dst); break;
            case QuantFormat.BF16: DequantBf16(src, dst); break;
            case QuantFormat.Q4_0: DequantQ4_0(src, dst); break;
            case QuantFormat.Q8_0: DequantQ8_0(src, dst); break;
            case QuantFormat.Q2_K: DequantQ2_K(src, dst); break;
            case QuantFormat.Q3_K: DequantQ3_K(src, dst); break;
            case QuantFormat.Q4_K: DequantQ4_K(src, dst); break;
            case QuantFormat.Q5_K: DequantQ5_K(src, dst); break;
            case QuantFormat.Q6_K: DequantQ6_K(src, dst); break;
            case QuantFormat.TRI: DequantTri(src, dst); break;
            case QuantFormat.I8_BLK: DequantI8Blk(src, dst); break;
            case QuantFormat.B1: DequantB1(src, dst); break;
            case QuantFormat.B1X: DequantB1X(src, dst); break;
            case QuantFormat.T1E: DequantT1E(src, dst, 34); break;
            case QuantFormat.T1E_LO: DequantT1E(src, dst, 26); break;
            default: throw new NotSupportedException(fmt.ToString());
        }
    }

    public static void DequantF16(ReadOnlySpan<byte> src, Span<float> dst)
    {
        fixed (byte* ps = src)
        {
            ushort* s = (ushort*)ps;
            for (int i = 0; i < dst.Length; i++) dst[i] = Dtypes.F16ToFloat(s[i]);
        }
    }

    public static void DequantBf16(ReadOnlySpan<byte> src, Span<float> dst)
    {
        fixed (byte* ps = src)
        {
            ushort* s = (ushort*)ps;
            for (int i = 0; i < dst.Length; i++) dst[i] = Dtypes.Bf16ToFloat(s[i]);
        }
    }

    public static void DequantQ4_0(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int nb = dst.Length / 32;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            for (int b = 0; b < nb; b++, s += 20, y += 32)
            {
                float d = Dtypes.F16ToFloat((ushort)(s[0] | (s[1] << 8)));
                for (int i = 0; i < 16; i++)
                {
                    y[i] = d * ((s[2 + i] & 0xF) - 8);
                    y[i + 16] = d * ((s[2 + i] >> 4) - 8);
                }
            }
        }
    }

    public static void DequantQ8_0(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int nb = dst.Length / 32;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            for (int b = 0; b < nb; b++, s += 34, y += 32)
            {
                float d = Dtypes.F16ToFloat((ushort)(s[0] | (s[1] << 8)));
                for (int i = 0; i < 32; i++) y[i] = d * (sbyte)s[2 + i];
            }
        }
    }

    // ggml layout: scales[16] @0, qs[64] @16, d @80, dmin @82. Bit-plane quants.
    public static void DequantQ2_K(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int nb = dst.Length / 256;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            for (int b = 0; b < nb; b++, s += 84, y += 256)
            {
                float d = Dtypes.F16ToFloat((ushort)(s[80] | (s[81] << 8)));
                float dmin = Dtypes.F16ToFloat((ushort)(s[82] | (s[83] << 8)));
                byte* sc = s, q = s + 16;
                for (int sub = 0; sub < 16; sub++)
                {
                    float a = d * (sc[sub] & 0xF);
                    float m = dmin * (sc[sub] >> 4);
                    int r = sub * 16;
                    for (int i = 0; i < 16; i++)
                    {
                        int w = r + i;
                        int shift = 2 * ((w % 128) / 32);
                        int qv = (q[32 * (w / 128) + (w % 128) % 32] >> shift) & 3;
                        y[w] = a * qv - m;
                    }
                }
            }
        }
    }

    // ggml layout: hmask[32] @0, qs[64] @32, scales[12] @96, d @108.
    public static void DequantQ3_K(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int nb = dst.Length / 256;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            Span<int> sc = stackalloc int[16];
            Span<uint> aux = stackalloc uint[4];
            for (int b = 0; b < nb; b++, s += 110, y += 256)
            {
                float d = Dtypes.F16ToFloat((ushort)(s[108] | (s[109] << 8)));
                byte* hm = s, q = s + 32, packed = s + 96;
                // kmask shuffle (uint32 little-endian over 12 bytes)
                uint a0 = (uint)(packed[0] | (packed[1] << 8) | (packed[2] << 16) | (packed[3] << 24));
                uint a1 = (uint)(packed[4] | (packed[5] << 8) | (packed[6] << 16) | (packed[7] << 24));
                uint a2 = (uint)(packed[8] | (packed[9] << 8) | (packed[10] << 16) | (packed[11] << 24));
                uint tmp = a2;
                uint r2 = ((a0 >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
                uint r3 = ((a1 >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
                uint r0 = (a0 & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);
                uint r1 = (a1 & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);
                aux[0] = r0; aux[1] = r1; aux[2] = r2; aux[3] = r3;
                for (int i = 0; i < 16; i++)
                    sc[i] = (sbyte)((byte)(aux[i / 4] >> (8 * (i % 4)))) - 32;
                for (int w = 0; w < 256; w++)
                {
                    int shift = 2 * ((w % 128) / 32);
                    int qv = (q[32 * (w / 128) + (w % 128) % 32] >> shift) & 3;
                    if ((hm[w % 32] & (1 << (w / 32))) == 0) qv -= 4;
                    y[w] = d * sc[w / 16] * qv;
                }
            }
        }
    }

    /// <summary>ggml get_scale_min_k4: decode 6-bit scale/min pair j from 12-byte packed array.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void GetScaleMinK4(int j, byte* q, out int sc, out int m)
    {
        if (j < 4)
        {
            sc = q[j] & 63; m = q[j + 4] & 63;
        }
        else
        {
            sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
            m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        }
    }

    public static void DequantQ4_K(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int nb = dst.Length / 256;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            for (int b = 0; b < nb; b++, s += 144, y += 256)
            {
                float d = Dtypes.F16ToFloat((ushort)(s[0] | (s[1] << 8)));
                float dmin = Dtypes.F16ToFloat((ushort)(s[2] | (s[3] << 8)));
                byte* packed = s + 4, q = s + 16;
                for (int sub = 0; sub < 8; sub++)
                {
                    // sub-block `sub` = weights [32*sub, 32*sub+31). 64-weight group sub/2 shares
                    // 32 qs bytes: even sub-blocks -> low nibbles, odd -> high nibbles.
                    GetScaleMinK4(sub, packed, out int scI, out int mI);
                    float a = d * scI, m = dmin * mI;
                    byte* qb = q + (sub / 2) * 32;
                    int wBase = sub * 32;
                    if ((sub & 1) == 0)
                        for (int i = 0; i < 32; i++) y[wBase + i] = a * (qb[i] & 0xF) - m;
                    else
                        for (int i = 0; i < 32; i++) y[wBase + i] = a * (qb[i] >> 4) - m;
                }
            }
        }
    }

    // ggml layout: d @0, dmin @2, scales[12] @4, qs[128] @16, qh[32] @144.
    public static void DequantQ5_K(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int nb = dst.Length / 256;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            for (int b = 0; b < nb; b++, s += 176, y += 256)
            {
                float d = Dtypes.F16ToFloat((ushort)(s[0] | (s[1] << 8)));
                float dmin = Dtypes.F16ToFloat((ushort)(s[2] | (s[3] << 8)));
                byte* packed = s + 4, ql = s + 16, qh = s + 144;
                for (int sub = 0; sub < 8; sub++)
                {
                    GetScaleMinK4(sub, packed, out int scI, out int mI);
                    float a = d * scI, m = dmin * mI;
                    int j = sub / 2;
                    byte* q = ql + j * 32;
                    int u1 = 1 << (2 * j), u2 = 2 << (2 * j);
                    if ((sub & 1) == 0)
                        for (int i = 0; i < 32; i++)
                            y[sub * 32 + i] = a * ((q[i] & 0xF) + ((qh[i] & u1) != 0 ? 16 : 0)) - m;
                    else
                        for (int i = 0; i < 32; i++)
                            y[sub * 32 + i] = a * ((q[i] >> 4) + ((qh[i] & u2) != 0 ? 16 : 0)) - m;
                }
            }
        }
    }

    public static void DequantQ6_K(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int nb = dst.Length / 256;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            for (int b = 0; b < nb; b++, s += 210, y += 256)
            {
                float d = Dtypes.F16ToFloat((ushort)(s[208] | (s[209] << 8)));
                byte* ql = s, qh = s + 128, sc = s + 192;
                for (int n = 0; n < 256; n += 128)
                {
                    for (int l = 0; l < 32; l++)
                    {
                        int isx = l / 16;
                        int q1 = (ql[n / 2 + l] & 0xF) | (((qh[n / 4 + l] >> 0) & 3) << 4);
                        int q2 = (ql[n / 2 + l + 32] & 0xF) | (((qh[n / 4 + l] >> 2) & 3) << 4);
                        int q3 = (ql[n / 2 + l] >> 4) | (((qh[n / 4 + l] >> 4) & 3) << 4);
                        int q4 = (ql[n / 2 + l + 32] >> 4) | (((qh[n / 4 + l] >> 6) & 3) << 4);
                        y[n + l] = d * (sbyte)sc[8 * (n / 128) + isx + 0] * (q1 - 32);
                        y[n + l + 32] = d * (sbyte)sc[8 * (n / 128) + isx + 2] * (q2 - 32);
                        y[n + l + 64] = d * (sbyte)sc[8 * (n / 128) + isx + 4] * (q3 - 32);
                        y[n + l + 96] = d * (sbyte)sc[8 * (n / 128) + isx + 6] * (q4 - 32);
                    }
                }
            }
        }
    }

    public static void DequantTri(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int nb = dst.Length / 64;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            for (int b = 0; b < nb; b++, s += 18, y += 64)
            {
                float scale = Dtypes.F16ToFloat((ushort)(s[0] | (s[1] << 8)));
                for (int i = 0; i < 64; i++)
                {
                    int code = (s[2 + i / 4] >> (2 * (i & 3))) & 3;
                    float v = code == 1 ? scale : code == 2 ? -scale : 0f;
                    y[i] = v;
                }
            }
        }
    }

    public static void DequantI8Blk(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int nb = dst.Length / 256;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            for (int b = 0; b < nb; b++, s += 258, y += 256)
            {
                float d = Dtypes.F16ToFloat((ushort)(s[0] | (s[1] << 8)));
                for (int i = 0; i < 256; i++) y[i] = d * (sbyte)s[2 + i];
            }
        }
    }

    // ---------------- extreme low-bit formats ----------------
    // B1  : [0..1] f16 alpha, [2..33] sign mask (bit set -> +alpha)          34 B / 256 w
    // B1X : B1 + [34] exception count k, k x (u8 idx, f16 exact value)       47 B / 256 w
    // T1E : [0..1] u16 n0, [2..3] u16 nP, [4..5] f16 alpha, rANS stream      40/32 B / 256 w
    //       (symbol 0 -> 0, 1 -> +alpha, 2 -> -alpha; stream is 34 / 26 B)

    public static void DequantB1(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int nb = dst.Length / 256;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            for (int b = 0; b < nb; b++, s += 34, y += 256)
            {
                float a = Dtypes.F16ToFloat((ushort)(s[0] | (s[1] << 8)));
                for (int i = 0; i < 256; i++)
                    y[i] = (s[2 + (i >> 3)] & (1 << (i & 7))) != 0 ? a : -a;
            }
        }
    }

    public static void DequantB1X(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int nb = dst.Length / 256;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            for (int b = 0; b < nb; b++, s += 47, y += 256)
            {
                float a = Dtypes.F16ToFloat((ushort)(s[0] | (s[1] << 8)));
                for (int i = 0; i < 256; i++)
                    y[i] = (s[2 + (i >> 3)] & (1 << (i & 7))) != 0 ? a : -a;
                int k = s[34];
                for (int e = 0; e < k; e++)
                {
                    int idx = s[35 + 3 * e];
                    float v = Dtypes.F16ToFloat((ushort)(s[35 + 3 * e + 1] | (s[35 + 3 * e + 2] << 8)));
                    y[idx] = v;
                }
            }
        }
    }

    public static void DequantT1E(ReadOnlySpan<byte> src, Span<float> dst, int streamBytes = 34)
    {
        int nb = dst.Length / 256;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            Span<byte> syms = stackalloc byte[256];
            for (int b = 0; b < nb; b++, s += 6 + streamBytes, y += 256)
            {
                uint n0 = (uint)(s[0] | (s[1] << 8));
                uint nP = (uint)(s[2] | (s[3] << 8));
                float a = Dtypes.F16ToFloat((ushort)(s[4] | (s[5] << 8)));
                EntropyCoder.Decode3(new ReadOnlySpan<byte>(s + 6, streamBytes), 256, n0, nP, syms);
                for (int i = 0; i < 256; i++)
                {
                    byte t = syms[i];
                    y[i] = t == 1 ? a : t == 2 ? -a : 0f;
                }
            }
        }
    }
}
