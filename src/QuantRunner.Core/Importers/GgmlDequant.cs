using System.Runtime.InteropServices;
using QuantRunner.Core.Quantization;

namespace QuantRunner.Core.Importers;

/// <summary>
/// Dequantization for GGML block formats, bit-exact with the llama.cpp reference
/// (ggml-quants.c / ggml-common.h).
///
/// Two families:
///  • layout-identical to QuantRunner's own codecs — F32/F16/BF16, Q8_0 and the K-quants
///    Q2_K/Q3_K/Q4_K/Q5_K/Q6_K (same block sizes, same field order, same dequant formula,
///    verified against the ggml reference) — delegated to BlockDequantizer;
///  • legacy formats that differ — Q4_0 (18 B, signed delta d = max/−8), Q4_1 (20 B,
///    delta+min), Q5_0 (22 B), Q5_1 (24 B) — implemented here from the reference.
///
/// Note QuantRunner's own Q4_0 is a different 20-byte format (positive delta); a GGUF Q4_0
/// tensor must therefore never pass through byte-exact — it is always dequantized here and
/// re-encoded to the target format.
/// </summary>
public static class GgmlDequant
{
    public const int GgmlF32 = 0, GgmlF16 = 1, GgmlQ4_0 = 2, GgmlQ4_1 = 3, GgmlQ5_0 = 6,
        GgmlQ5_1 = 7, GgmlQ8_0 = 8, GgmlQ2_K = 10, GgmlQ3_K = 11, GgmlQ4_K = 12,
        GgmlQ5_K = 13, GgmlQ6_K = 14, GgmlBF16 = 30;

    /// <summary>GGML types whose byte layout is identical to the same-named QuantRunner format
    /// (byte-exact passthrough allowed in both directions).</summary>
    public static bool LayoutMatchesOurs(int ggmlType) => ggmlType switch
    {
        GgmlF32 or GgmlF16 or GgmlBF16 or GgmlQ8_0 or GgmlQ2_K or GgmlQ3_K or GgmlQ4_K
            or GgmlQ5_K or GgmlQ6_K => true,
        _ => false
    };

    /// <summary>Dequantize dst.Length elements of a GGML tensor; src must hold exactly the
    /// blocks covering them. Supports every type GgufFile can address (F32/F16/BF16,
    /// Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q2_K..Q6_K).</summary>
    public static void Dequant(int ggmlType, ReadOnlySpan<byte> src, Span<float> dst)
    {
        switch (ggmlType)
        {
            case GgmlQ4_0: DequantQ4_0(src, dst); break;
            case GgmlQ4_1: DequantQ4_1(src, dst); break;
            case GgmlQ5_0: DequantQ5_0(src, dst); break;
            case GgmlQ5_1: DequantQ5_1(src, dst); break;
            case GgmlF32: BlockDequantizer.Dequantize(QuantFormat.F32, src, dst); break;
            case GgmlF16: BlockDequantizer.Dequantize(QuantFormat.F16, src, dst); break;
            case GgmlBF16: BlockDequantizer.Dequantize(QuantFormat.BF16, src, dst); break;
            case GgmlQ8_0: BlockDequantizer.Dequantize(QuantFormat.Q8_0, src, dst); break;
            case GgmlQ2_K: BlockDequantizer.Dequantize(QuantFormat.Q2_K, src, dst); break;
            case GgmlQ3_K: BlockDequantizer.Dequantize(QuantFormat.Q3_K, src, dst); break;
            case GgmlQ4_K: BlockDequantizer.Dequantize(QuantFormat.Q4_K, src, dst); break;
            case GgmlQ5_K: BlockDequantizer.Dequantize(QuantFormat.Q5_K, src, dst); break;
            case GgmlQ6_K: BlockDequantizer.Dequantize(QuantFormat.Q6_K, src, dst); break;
            default:
                throw new NotSupportedException(
                    $"GGML type {ggmlType} ({GgufFile.GgmlTypeName(ggmlType)}) is not supported for dequantization — " +
                    "supported: F32/F16/BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K (IQ*/TQ* are not)");
        }
    }
    // ------------------------------------------------------------------ legacy dequants
    // Reference: ggml-quants.c dequantize_row_q4_0/q4_1/q5_0/q5_1.
    // Nibble order (all four): byte j holds element j in the LOW nibble and element j+16
    // in the HIGH nibble (de-interleaved halves, not sequential pairs).

    // block_q4_0 { f16 d; u8 qs[16] } = 18 B — y = (q − 8) · d, d = max/−8 (typically negative)
    public static unsafe void DequantQ4_0(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int nb = dst.Length / 32;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            for (int b = 0; b < nb; b++, s += 18, y += 32)
            {
                float d = Dtypes.F16ToFloat((ushort)(s[0] | (s[1] << 8)));
                for (int j = 0; j < 16; j++)
                {
                    y[j] = d * ((s[2 + j] & 0xF) - 8);
                    y[j + 16] = d * ((s[2 + j] >> 4) - 8);
                }
            }
        }
    }

    // block_q4_1 { f16 d; f16 m; u8 qs[16] } = 20 B — y = q · d + m
    public static unsafe void DequantQ4_1(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int nb = dst.Length / 32;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            for (int b = 0; b < nb; b++, s += 20, y += 32)
            {
                float d = Dtypes.F16ToFloat((ushort)(s[0] | (s[1] << 8)));
                float m = Dtypes.F16ToFloat((ushort)(s[2] | (s[3] << 8)));
                for (int j = 0; j < 16; j++)
                {
                    y[j] = d * (s[4 + j] & 0xF) + m;
                    y[j + 16] = d * (s[4 + j] >> 4) + m;
                }
            }
        }
    }

    // block_q5_0 { f16 d; u8 qh[4]; u8 qs[16] } = 22 B — y = ((q | highbit) − 16) · d
    // qh is a little-endian uint32: bit j = 5th bit of element j, bit j+16 = 5th bit of element j+16.
    public static unsafe void DequantQ5_0(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int nb = dst.Length / 32;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            for (int b = 0; b < nb; b++, s += 22, y += 32)
            {
                float d = Dtypes.F16ToFloat((ushort)(s[0] | (s[1] << 8)));
                uint qh = (uint)(s[2] | (s[3] << 8) | (s[4] << 16) | (s[5] << 24));
                for (int j = 0; j < 16; j++)
                {
                    int x0 = (s[6 + j] & 0xF) | (int)(((qh >> j) & 1u) << 4);
                    int x1 = (s[6 + j] >> 4) | (int)(((qh >> (j + 16)) & 1u) << 4);
                    y[j] = d * (x0 - 16);
                    y[j + 16] = d * (x1 - 16);
                }
            }
        }
    }

    // block_q5_1 { f16 d; f16 m; u8 qh[4]; u8 qs[16] } = 24 B — y = (q | highbit) · d + m
    public static unsafe void DequantQ5_1(ReadOnlySpan<byte> src, Span<float> dst)
    {
        int nb = dst.Length / 32;
        fixed (byte* ps = src)
        fixed (float* pd = dst)
        {
            byte* s = ps; float* y = pd;
            for (int b = 0; b < nb; b++, s += 24, y += 32)
            {
                float d = Dtypes.F16ToFloat((ushort)(s[0] | (s[1] << 8)));
                float m = Dtypes.F16ToFloat((ushort)(s[2] | (s[3] << 8)));
                uint qh = (uint)(s[4] | (s[5] << 8) | (s[6] << 16) | (s[7] << 24));
                for (int j = 0; j < 16; j++)
                {
                    int x0 = (s[8 + j] & 0xF) | (int)(((qh >> j) & 1u) << 4);
                    int x1 = (s[8 + j] >> 4) | (int)(((qh >> (j + 16)) & 1u) << 4);
                    y[j] = d * x0 + m;
                    y[j + 16] = d * x1 + m;
                }
            }
        }
    }

    // ------------------------------------------------------------------ writer-side re-quant
    // QuantRunner formats that have no ggml equivalent are exported as F16; our own Q4_0
    // (20 B, positive delta) is re-encoded to a TRUE ggml Q4_0 so a qw -> gguf -> qw round
    // trip keeps 4-bit storage size instead of inflating to F16.

    /// <summary>Encode n elements as reference ggml Q4_0 (18 B blocks): d = max/−8,
    /// xi = min(15, trunc(x·(1/d) + 8.5)), nibble j = elem j, nibble j&lt;&lt;4 = elem j+16.</summary>
    public static unsafe void QuantizeGgmlQ4_0(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int nb = src.Length / 32;
        fixed (float* px = src)
        fixed (byte* pd = dst)
        {
            float* x = px; byte* y = pd;
            for (int b = 0; b < nb; b++, x += 32, y += 18)
            {
                float amax = 0f, max = 0f;
                for (int j = 0; j < 32; j++)
                {
                    float a = MathF.Abs(x[j]);
                    if (a > amax) { amax = a; max = x[j]; }
                }
                float d = max / -8f;
                ushort dh = Dtypes.FloatToF16(d);
                y[0] = (byte)(dh & 0xFF); y[1] = (byte)(dh >> 8);
                if (d == 0f) { for (int j = 0; j < 16; j++) y[2 + j] = 8; continue; }
                float id = 1f / d;
                for (int j = 0; j < 16; j++)
                {
                    // C float->int cast truncates toward zero, same as (int8_t)(x + 8.5f)
                    int v0 = Math.Min(15, (int)(x[j] * id + 8.5f));
                    int v1 = Math.Min(15, (int)(x[j + 16] * id + 8.5f));
                    y[2 + j] = (byte)(v0 | (v1 << 4));
                }
            }
        }
    }

    /// <summary>Bytes per element payload when exporting <paramref name="fmt"/> to a GGML type
    /// (used only for reporting; the writer computes offsets from the type mapping).</summary>
    public static int ExportTypeBytes(QuantFormat fmt) => fmt switch
    {
        QuantFormat.F32 => 4,
        QuantFormat.F16 or QuantFormat.BF16 => 2,
        QuantFormat.Q4_0 => 18 * 8 / 32,
        QuantFormat.Q8_0 => 34 * 8 / 32,
        QuantFormat.Q2_K => 84 * 8 / 256,
        QuantFormat.Q3_K => 110 * 8 / 256,
        QuantFormat.Q4_K => 144 * 8 / 256,
        QuantFormat.Q5_K => 176 * 8 / 256,
        QuantFormat.Q6_K => 210 * 8 / 256,
        _ => 2
    };
}
