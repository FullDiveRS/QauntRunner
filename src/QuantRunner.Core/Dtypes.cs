using System.Runtime.CompilerServices;

namespace QuantRunner.Core;

/// <summary>Storage data types / quantization formats supported by QuantRunner.</summary>
public enum QuantFormat : int
{
    F32 = 0,     // 4 B/weight
    F16 = 1,     // 2 B/weight
    BF16 = 2,    // 2 B/weight (brain float)
    Q4_0 = 3,    // GGML: 32w/block, 20 B  (4-bit + f16 scale)
    Q8_0 = 4,    // GGML: 32w/block, 34 B  (8-bit + f16 scale)
    Q2_K = 5,    // GGML: 256w/block, 84 B (super-block 2-bit)
    Q3_K = 10,   // GGML: 256w/block, 110 B (super-block 3-bit)
    Q4_K = 6,    // GGML: 256w/block, 144 B (super-block 4-bit)
    Q5_K = 11,   // GGML: 256w/block, 176 B (super-block 5-bit)
    Q6_K = 7,    // GGML: 256w/block, 210 B (super-block 6-bit)
    TRI = 8,     // QuantRunner ternary 1.58-bit: 64w/block, 18 B (2.25 B... bits/weight = 2.25)
    I8_BLK = 9,  // QuantRunner: 256w/block, 258 B (int8 + f16 scale per super-block)

    // ---- extreme low-bit formats (below TRI) ----
    B1 = 12,     // binary ±1: 256w/block, 34 B (f16 scale + 256-bit sign mask) = 1.0625 bpw
    B1X = 13,    // binary + up to 4 exact exceptions: 256w/block, 47 B (34B B1 + u8 count + 4x(u8 idx + f16 val)) = 1.469 bpw
    T1E = 14,    // entropy-coded sparse ternary: 256w/block, 40 B (u16 zeros + u16 pluses + f16 scale + 34B rANS stream) = 1.25 bpw
    T1E_LO = 15  // entropy-coded sparse ternary, tighter: 256w/block, 32 B (same header + 26B rANS stream) = 1.0 bpw
}

public static class QuantFormatEx
{
    public static bool IsFp(this QuantFormat f) => f is QuantFormat.F32 or QuantFormat.F16 or QuantFormat.BF16;
    public static bool IsQuantized(this QuantFormat f) => !f.IsFp();
}

public static class Dtypes
{
    /// <summary>Bytes per GGML block of the given format. FP formats report 1 element/block.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (int blockSize, int blockBytes) BlockSpec(QuantFormat f) => f switch
    {
        QuantFormat.F32 => (1, 4),
        QuantFormat.F16 => (1, 2),
        QuantFormat.BF16 => (1, 2),
        QuantFormat.Q4_0 => (32, 20),
        QuantFormat.Q8_0 => (32, 34),
        QuantFormat.Q2_K => (256, 84),
        QuantFormat.Q3_K => (256, 110),
        QuantFormat.Q4_K => (256, 144),
        QuantFormat.Q5_K => (256, 176),
        QuantFormat.Q6_K => (256, 210),
        QuantFormat.TRI => (64, 18),
        QuantFormat.I8_BLK => (256, 258),
        QuantFormat.B1 => (256, 34),
        QuantFormat.B1X => (256, 47),
        QuantFormat.T1E => (256, 40),
        QuantFormat.T1E_LO => (256, 32),
        _ => throw new NotSupportedException($"format {f}")
    };

    /// <summary>Total byte size of a tensor with n elements in the given format (n must be divisible by block).</summary>
    public static long TensorBytes(long n, QuantFormat f)
    {
        var (bs, bb) = BlockSpec(f);
        if (n % bs != 0) throw new ArgumentException($"element count {n} not a multiple of block size {bs} for {f}");
        return n / bs * bb;
    }

    public static long BlockCount(long n, QuantFormat f)
    {
        var (bs, _) = BlockSpec(f);
        return n / bs;
    }

    // ---------------- float16 <-> float32 (manual, no Half struct overhead) ----------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float F16ToFloat(ushort h)
    {
        uint sign = (uint)(h & 0x8000) << 16;
        uint exp = (uint)(h >> 10) & 0x1F;
        uint man = (uint)(h & 0x3FF);
        uint bits;
        if (exp == 0)
        {
            if (man == 0) { bits = sign; }
            else
            {
                // subnormal half -> normalize
                int e = -1;
                uint m = man;
                do { m <<= 1; e++; } while ((m & 0x400) == 0);
                bits = sign | ((uint)(127 - 15 - e) << 23) | ((m & 0x3FF) << 13);
            }
        }
        else if (exp == 31) { bits = sign | 0x7F800000u | (man << 13); }
        else { bits = sign | ((exp + 112) << 23) | (man << 13); }
        return BitConverter.UInt32BitsToSingle(bits);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort FloatToF16(float f)
    {
        uint x = BitConverter.SingleToUInt32Bits(f);
        uint sign = (x >> 16) & 0x8000u;
        int exp = (int)((x >> 23) & 0xFF) - 127;
        uint man = x & 0x7FFFFF;
        if (((x >> 23) & 0xFF) == 0xFF) // inf/nan
            return (ushort)(sign | 0x7C00u | (man != 0 ? 0x200u : 0u));
        if (exp > 15) return (ushort)(sign | 0x7C00u);          // overflow -> inf
        if (exp >= -14)
        {
            uint rounded = man + 0x1000u;
            if ((rounded & 0x800000u) != 0) { rounded = 0; exp++; } // carry
            return (ushort)(sign | ((uint)(exp + 15) << 10) | (rounded >> 13));
        }
        // subnormal
        if (exp < -25) return (ushort)sign;
        uint m2 = (0x800000u | man) >> (1 - exp - 15 + 13 - 10); // shift into 10 bits
        // simpler: value = (1<<23)|man scaled
        uint bits23 = 0x800000u | man;
        int shift = (-exp - 14) + 13;
        uint half = bits23 >> shift;
        uint rem = bits23 & ((1u << shift) - 1);
        uint mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1) != 0)) half++;
        return (ushort)(sign | half);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Bf16ToFloat(ushort b) => BitConverter.UInt32BitsToSingle((uint)b << 16);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort FloatToBf16(float f)
    {
        uint x = BitConverter.SingleToUInt32Bits(f);
        // round to nearest even
        uint lsb = (x >> 16) & 1;
        x += 0x7FFFu + lsb;
        return (ushort)(x >> 16);
    }

    // ---------------- float8_e4m3 (OCP "fn" variant, torch.float8_e4m3fn) ----------------
    // 1 sign | 4 exponent (bias 7) | 3 mantissa. No infinities; 0x7F/0xFF are NaN. Max 448.

    private static readonly float[] F8E4M3Table = BuildF8E4M3Table();

    private static float[] BuildF8E4M3Table()
    {
        var t = new float[256];
        for (int b = 0; b < 256; b++)
        {
            int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
            float v;
            if (e == 0) v = m * (1f / 512f);                       // subnormal: m/8 * 2^-6
            else if (e == 15 && m == 7) v = float.NaN;             // fn variant: only NaN, no inf
            else v = MathF.Pow(2, e - 7) * (1f + m * (1f / 8f));
            t[b] = s != 0 ? -v : v;
        }
        return t;
    }

    /// <summary>Decode one float8_e4m3fn byte (table lookup).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float F8E4M3ToFloat(byte b) => F8E4M3Table[b];

    /// <summary>Round-to-nearest-even encode of a float into float8_e4m3fn (used by tests/synthetic data).</summary>
    public static byte FloatToF8E4M3(float f)
    {
        if (float.IsNaN(f)) return 0x7F;
        uint x = BitConverter.SingleToUInt32Bits(f);
        uint sign = (x >> 31) != 0 ? 0x80u : 0u;
        x &= 0x7FFFFFFF;
        if (x >= 0x7F800000) return (byte)(sign | 0x7F);           // inf -> NaN encoding (fn has no inf)
        int e = (int)((x >> 23) & 0xFF) - 127 + 7;                 // exponent in bias-7 terms
        uint man = x & 0x7FFFFF;
        if (e <= 0)
        {
            // value = (0x800000|man) * 2^(e-30); subnormal grid: m * 2^-9  =>  m = value >> (21 - e)
            int shift = 21 - e;
            if (shift >= 31) return (byte)sign;                    // underflow -> zero
            uint full = 0x800000u | man;
            uint m = full >> shift;
            uint rem = full & ((1u << shift) - 1), mid = 1u << (shift - 1);
            if (rem > mid || (rem == mid && (m & 1) != 0)) m++;
            if (m > 7) return (byte)(sign | 0x08);                 // rounds up to smallest normal
            return (byte)(sign | m);
        }
        // normal: round mantissa 23 -> 3 bits (nearest even); carry may bump the exponent
        uint m3 = man >> 20, rem3 = man & 0xFFFFF, mid3 = 0x80000;
        if (rem3 > mid3 || (rem3 == mid3 && (m3 & 1) != 0)) m3++;
        if (m3 == 8) { m3 = 0; e++; }
        if (e >= 16) return (byte)(sign | 0x7E);                   // saturate to max normal 448
        return (byte)(sign | (uint)(e << 3) | m3);
    }
}
