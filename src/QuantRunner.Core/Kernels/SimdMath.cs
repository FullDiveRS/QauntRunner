using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace QuantRunner.Core.Kernels;

/// <summary>
/// Cross-platform SIMD math (Vector256 when available, Vector128/scalar fallback).
/// All hot paths accumulate across blocks and horizontal-sum once per row.
/// </summary>
public static unsafe class SimdMath
{
    // ------------------------- horizontal sum (once per row, cheap) -------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float HSum(Vector256<float> v)
    {
        float s = 0;
        for (int i = 0; i < 8; i++) s += v.GetElement(i);
        return s;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float HSum(Vector128<float> v)
    {
        float s = 0;
        for (int i = 0; i < 4; i++) s += v.GetElement(i);
        return s;
    }

    // ------------------------- fp32 dot / axpy -------------------------

    public static float DotF32(float* a, float* b, int n)
    {
        if (Vector256.IsHardwareAccelerated && n >= 8)
        {
            var acc = Vector256<float>.Zero;
            int i = 0;
            for (; i + 8 <= n; i += 8)
            {
                var va = Vector256.Load(a + i);
                var vb = Vector256.Load(b + i);
                acc = Vector256.Add(acc, Vector256.Multiply(va, vb));
            }
            float s = HSum(acc);
            for (; i < n; i++) s += a[i] * b[i];
            return s;
        }
        float ss = 0;
        for (int i = 0; i < n; i++) ss += a[i] * b[i];
        return ss;
    }

    public static void AxpyF32(float* x, float* y, float alpha, int n) // y += alpha * x
    {
        if (Vector256.IsHardwareAccelerated && n >= 8)
        {
            var va = Vector256.Create(alpha);
            int i = 0;
            for (; i + 8 <= n; i += 8)
            {
                var vy = Vector256.Load(y + i);
                var vx = Vector256.Load(x + i);
                Vector256.Store(Vector256.Add(vy, Vector256.Multiply(va, vx)), y + i);
            }
            for (; i < n; i++) y[i] += alpha * x[i];
            return;
        }
        for (int i = 0; i < n; i++) y[i] += alpha * x[i];
    }

    public static void CopyF32(float* src, float* dst, int n)
    {
        if (Vector256.IsHardwareAccelerated && n >= 8)
        {
            int i = 0;
            for (; i + 8 <= n; i += 8)
                Vector256.Store(Vector256.Load(src + i), dst + i);
            for (; i < n; i++) dst[i] = src[i];
            return;
        }
        for (int i = 0; i < n; i++) dst[i] = src[i];
    }

    // ------------------------- nibble unpack: 16 bytes -> 32 floats -------------------------
    // loOut[i] = low nibble of qs[i] (weights 0..15), hiOut[i] = high nibble (weights 16..31)

    private static readonly float[] NibLut = BuildNibLut();
    private static float[] BuildNibLut()
    {
        var lut = new float[512];
        for (int b = 0; b < 256; b++)
        {
            lut[2 * b] = b & 0xF;
            lut[2 * b + 1] = b >> 4;
        }
        return lut;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void UnpackNibbles32(ReadOnlySpan<byte> qs, Span<float> loOut, Span<float> hiOut)
    {
        if (Vector256.IsHardwareAccelerated && BitConverter.IsLittleEndian)
        {
            ref byte r = ref MemoryMarshal.GetReference(qs);
            Vector128<byte> b = Vector128.LoadUnsafe(ref r);
            Vector128<byte> mask = Vector128.Create((byte)0x0F);
            Vector128<byte> loB = b & mask;
            Vector128<byte> hiB = Vector128.ShiftRightLogical(b, 4) & mask;

            NibblesToFloats(loB, loOut);
            NibblesToFloats(hiB, hiOut);
            return;
        }
        var lut = NibLut;
        for (int i = 0; i < 16; i++)
        {
            byte v = qs[i];
            loOut[i] = lut[2 * v];
            hiOut[i] = lut[2 * v + 1];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NibblesToFloats(Vector128<byte> v, Span<float> dst)
    {
        // byte -> ushort -> uint -> float (values 0..15, safe reinterprets)
        var (w0, w1) = Vector128.Widen(v);           // ushort x8 + x8
        var (d0, d1) = Vector128.Widen(w0);          // uint x4 + x4
        var (d2, d3) = Vector128.Widen(w1);
        Vector128.ConvertToSingle(d0.AsInt32()).CopyTo(dst);
        Vector128.ConvertToSingle(d1.AsInt32()).CopyTo(dst.Slice(4));
        Vector128.ConvertToSingle(d2.AsInt32()).CopyTo(dst.Slice(8));
        Vector128.ConvertToSingle(d3.AsInt32()).CopyTo(dst.Slice(12));
    }

    /// <summary>Low nibbles of 16 bytes -> 16 floats (Q4_K even sub-blocks).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void UnpackLo16(ReadOnlySpan<byte> qs, Span<float> dst)
    {
        if (Vector256.IsHardwareAccelerated && BitConverter.IsLittleEndian)
        {
            ref byte r = ref MemoryMarshal.GetReference(qs);
            Vector128<byte> b = Vector128.LoadUnsafe(ref r) & Vector128.Create((byte)0x0F);
            NibblesToFloats(b, dst);
            return;
        }
        var lut = NibLut;
        for (int i = 0; i < 16; i++) dst[i] = lut[2 * qs[i]];
    }

    /// <summary>High nibbles of 16 bytes -> 16 floats (Q4_K odd sub-blocks).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void UnpackHi16(ReadOnlySpan<byte> qs, Span<float> dst)
    {
        if (Vector256.IsHardwareAccelerated && BitConverter.IsLittleEndian)
        {
            ref byte r = ref MemoryMarshal.GetReference(qs);
            Vector128<byte> b = (Vector128.ShiftRightLogical(Vector128.LoadUnsafe(ref r), 4) & Vector128.Create((byte)0x0F));
            NibblesToFloats(b, dst);
            return;
        }
        var lut = NibLut;
        for (int i = 0; i < 16; i++) dst[i] = lut[2 * qs[i] + 1];
    }

    // ------------------------- 2-bit unpack: n bytes -> 4n floats (values 0..3) -------------------------

    private static readonly float[] TwoBitLut = { 0f, 1f, 2f, 3f };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Unpack2Bit(ReadOnlySpan<byte> qs, int nBytes, Span<float> dst)
    {
        for (int i = 0; i < nBytes; i++)
        {
            byte v = qs[i];
            dst[4 * i] = TwoBitLut[v & 3];
            dst[4 * i + 1] = TwoBitLut[(v >> 2) & 3];
            dst[4 * i + 2] = TwoBitLut[(v >> 4) & 3];
            dst[4 * i + 3] = TwoBitLut[(v >> 6) & 3];
        }
    }

    /// <summary>TRI code -> float: 00=0, 01=+scale, 10=-scale.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void UnpackTri(ReadOnlySpan<byte> qs, int nBytes, float scale, Span<float> dst)
    {
        for (int i = 0; i < nBytes; i++)
        {
            byte v = qs[i];
            int c0 = v & 3, c1 = (v >> 2) & 3, c2 = (v >> 4) & 3, c3 = (v >> 6) & 3;
            dst[4 * i] = c0 == 1 ? scale : c0 == 2 ? -scale : 0f;
            dst[4 * i + 1] = c1 == 1 ? scale : c1 == 2 ? -scale : 0f;
            dst[4 * i + 2] = c2 == 1 ? scale : c2 == 2 ? -scale : 0f;
            dst[4 * i + 3] = c3 == 1 ? scale : c3 == 2 ? -scale : 0f;
        }
    }
}
