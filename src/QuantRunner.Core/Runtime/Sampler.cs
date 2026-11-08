namespace QuantRunner.Core.Runtime;

/// <summary>
/// Sampling with temperature, top-k, top-p (nucleus) and a repeat-penalty window.
/// temperature &lt;= 0 means greedy argmax (deterministic). Sorts with Array.Sort(keys, items)
/// so no lambdas capture the logits buffer (span-capture is illegal and slower anyway).
/// </summary>
public sealed class Sampler
{
    public float Temperature { get; set; } = 0.8f;
    public int TopK { get; set; } = 40;            // 0 = off
    public float TopP { get; set; } = 0.95f;       // >= 1 = off
    public float MinP { get; set; } = 0f;          // 0 = off; keep tokens with p >= MinP * p_max
    public float RepeatPenalty { get; set; } = 1.1f;
    public int RepeatWindow { get; set; } = 64;
    public int Seed { get; set; } = -1;

    private Random? _rng;
    private int[] _order = Array.Empty<int>();     // reused scratch (no per-token allocs)

    public Sampler() { }

    /// <summary>Re-initialize the RNG from Seed (lazily applied on first Sample otherwise).</summary>
    public void ResetRng() => _rng = null;

    private void EnsureRng() => _rng ??= Seed >= 0 ? new Random(Seed) : new Random();

    /// <summary>Sample from logits (buffer may be modified in place). Returns vocab id.</summary>
    public unsafe int Sample(Span<float> logits, ReadOnlySpan<int> recentTokens)
    {
        EnsureRng();
        int vocab = logits.Length;
        if (_order.Length != vocab) _order = new int[vocab];

        // ---- repeat penalty ----
        if (RepeatPenalty > 1f && recentTokens.Length > 0)
        {
            int start = Math.Max(0, recentTokens.Length - RepeatWindow);
            for (int i = start; i < recentTokens.Length; i++)
            {
                int t = recentTokens[i];
                if ((uint)t < (uint)vocab)
                    logits[t] = logits[t] > 0 ? logits[t] / RepeatPenalty : logits[t] * RepeatPenalty;
            }
        }

        // ---- greedy ----
        if (Temperature <= 0f)
        {
            int best = 0;
            float bv = logits[0];
            for (int i = 1; i < vocab; i++)
                if (logits[i] > bv) { bv = logits[i]; best = i; }
            return best;
        }

        // ---- temperature ----
        float invT = 1f / Temperature;
        for (int i = 0; i < vocab; i++) logits[i] *= invT;

        // sort descending: order[0] = id of largest logit
        for (int i = 0; i < vocab; i++) _order[i] = i;
        float[] keys = logits.ToArray();
        Array.Sort(keys, _order);                       // ascending by key
        Array.Reverse(_order);                          // descending

        // ---- top-k ----
        int keep = vocab;
        if (TopK > 0 && TopK < vocab) keep = TopK;
        for (int i = keep; i < vocab; i++) logits[_order[i]] = float.NegativeInfinity;

        // ---- min-p: p_i >= MinP * p_max  <=>  l_i >= l_max + ln(MinP) (order[0] is the max) ----
        if (MinP > 0f && MinP < 1f)
        {
            float cut = logits[_order[0]] + MathF.Log(MinP);
            for (int i = 0; i < vocab; i++)
                if (logits[i] != float.NegativeInfinity && logits[i] < cut)
                    logits[i] = float.NegativeInfinity;
        }

        // ---- top-p ----
        if (TopP < 1f)
        {
            float totalExp = 0;
            for (int i = 0; i < vocab; i++)
            {
                float l = logits[_order[i]];
                if (l == float.NegativeInfinity) break;
                totalExp += MathF.Exp(l);
            }
            float cum = 0;
            for (int i = 0; i < vocab; i++)
            {
                float l = logits[_order[i]];
                if (l == float.NegativeInfinity) break;
                cum += MathF.Exp(l);
                if (cum / Math.Max(totalExp, 1e-9f) >= TopP) { keep = i + 1; break; }
            }
            for (int i = keep; i < vocab; i++) logits[_order[i]] = float.NegativeInfinity;
        }

        // ---- softmax over survivors + draw ----
        float max = float.NegativeInfinity;
        for (int i = 0; i < vocab; i++) if (logits[i] > max) max = logits[i];
        float sum = 0;
        for (int i = 0; i < vocab; i++) { logits[i] = MathF.Exp(logits[i] - max); sum += logits[i]; }
        float r = (float)_rng.NextDouble() * sum;
        float acc = 0;
        for (int i = 0; i < vocab; i++)
        {
            acc += logits[i];
            if (acc >= r) return i;
        }
        return vocab - 1;
    }
}
