namespace QuantRunner.Core.Quantization;

/// <summary>
/// Tiny static-ratio rANS entropy coder for a 3-symbol alphabet whose probabilities are
/// integer counts summing to 256 (8-bit probability scale).
///
/// Why it exists: TRI packs ternary symbols at a flat 2 bits/symbol, so its stored cost
/// floor is 2.25 bpw (with block scale) no matter how sparse the weights are. When the
/// ternary assignment is zero-heavy — which sparsity-controlled ternarization makes true
/// by construction — the Shannon entropy of the symbol stream is far below 2 bits, and an
/// entropy coder converts that slack into stored bits. That is the mechanism that pushes
/// the T1E formats below the 1.58-bit ternary floor: T1E stores 256 ternary weights in
/// 34 code bytes (T1E_LO: 26), i.e. 1.06 / 0.81 content bits per weight on top of the
/// header, with thecoder wasting less than ~0.1 bits/symbol versus ideal.
///
/// Construction (canonical ryg-style rANS, 32-bit state):
///   - probability scale M = 256, lower renormalization bound L = 1 &lt;&lt; 23.
///     Max state after a step is &lt; 256*L + 256 &lt; 2^32, so everything fits u32 exactly.
///   - encode walks the symbols BACKWARDS and emits renormalization bytes backwards too;
///     the final 4-byte state flush therefore lands at the FRONT of the payload, which is
///     exactly where the forward-reading decoder starts. Bit-exact by construction.
///   - decode: renorm while x &lt; L (x = x&lt;&lt;8 | byte), then slot = x &amp; 255 selects the
///     symbol via cumulative frequencies, x = f * (x &gt;&gt; 8) + slot - cum.
///   - correctness needs cum[s] + freq[s] &lt;= 256 for every symbol — guaranteed because
///     the three counts sum to exactly 256.
/// </summary>
public static class EntropyCoder
{
    public const uint ProbScale = 256;
    public const uint LowerBound = 1u << 23;

    /// <summary>
    /// Encode symbols (values 0/1/2; symbols with zero frequency must not occur) with
    /// frequencies (f0, f1, 256 - f0 - f1). Returns the number of bytes written to
    /// dst[0..ret), or -1 when the stream does not fit into dst (the caller is expected to
    /// re-encode with a sparser distribution — T1E rate control works exactly this way).
    /// </summary>
    public static int Encode3(ReadOnlySpan<byte> symbols, uint f0, uint f1, Span<byte> dst)
    {
        uint f2 = ProbScale - f0 - f1;
        int ptr = dst.Length;                       // write backwards from the end
        uint x = LowerBound;
        for (int i = symbols.Length - 1; i >= 0; i--)
        {
            uint s = symbols[i];
            uint f, c;
            if (s == 0) { f = f0; c = 0u; }
            else if (s == 1) { f = f1; c = f0; }
            else { f = f2; c = f0 + f1; }
            if (f == 0) throw new ArgumentException($"symbol {s} has zero frequency");
            // renormalize: emit low bytes while the post-step state could overflow u32.
            // Post-step x < 256*L + 256, so at most 2 bytes per symbol.
            while (x >= LowerBound * f)
            {
                if (ptr == 0) return -1;
                dst[--ptr] = (byte)(x & 0xFF);
                x >>= 8;
            }
            x = ((x / f) << 8) + (x % f) + c;
        }
        // flush the full state (4 bytes, big-endian at the payload front)
        for (int b = 0; b < 4; b++)
        {
            if (ptr == 0) return -1;
            dst[--ptr] = (byte)(x & 0xFF);
            x >>= 8;
        }
        int used = dst.Length - ptr;
        if (ptr > 0)
        {
            // slide the used tail to the front; regions overlap, but the write index always
            // stays behind the read index, so a forward in-place copy is safe for any size
            for (int i = 0; i < used; i++) dst[i] = dst[ptr + i];
        }
        return used;
    }

    /// <summary>
    /// Decode nSymbols from src. src must be the payload returned by Encode3 (padded tail is
    /// ignored — decoding reads exactly the bytes the encoder emitted). Symbols are written
    /// to outSym (length &gt;= nSymbols).
    /// </summary>
    public static void Decode3(ReadOnlySpan<byte> src, int nSymbols, uint f0, uint f1, Span<byte> outSym)
    {
        uint f2 = ProbScale - f0 - f1;
        int p = 0;
        uint x = 0;
        for (int b = 0; b < 4; b++)
        {
            if (p >= src.Length) throw new InvalidDataException("rANS payload truncated (state)");
            x = (x << 8) | src[p++];
        }
        for (int i = 0; i < nSymbols; i++)
        {
            while (x < LowerBound)
            {
                if (p >= src.Length) throw new InvalidDataException("rANS payload truncated (renorm)");
                x = (x << 8) | src[p++];
            }
            uint slot = x & (ProbScale - 1);
            uint s, f, c;
            if (slot < f0) { s = 0; f = f0; c = 0; }
            else if (slot < f0 + f1) { s = 1; f = f1; c = f0; }
            else { s = 2; f = f2; c = f0 + f1; }
            if (f == 0) throw new InvalidDataException("rANS decoded a zero-frequency symbol (corrupt stream)");
            x = f * (x >> 8) + slot - c;
            outSym[i] = (byte)s;
        }
    }
}
