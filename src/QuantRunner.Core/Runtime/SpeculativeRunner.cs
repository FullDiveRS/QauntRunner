using System.Diagnostics;

namespace QuantRunner.Core.Runtime;

/// <summary>
/// Greedy speculative decoding: a small draft model proposes up to <see cref="DraftTokens"/>
/// tokens per round (greedy argmax), the target verifies them with its own forwards. A draft
/// token is accepted iff the target's greedy token at that position matches — so the output is
/// EXACTLY the target's greedy trajectory (bit-for-bit, repeat penalty included when the target
/// sampler sets one), but each accepted token costs one cheap draft forward instead of an
/// expensive target forward. On the first mismatch the target's own prediction corrects the
/// draft, both engines rewind (O(1) cache truncate + KDA state restore) and re-sync on the
/// correction token. A fully-accepted round also yields one bonus token from the last verify
/// forward for free. EOS is the engine-wide convention (token id 0 stops).
/// </summary>
public sealed class SpeculativeRunner
{
    public StreamingEngine Target { get; }
    public StreamingEngine Draft { get; }
    public int DraftTokens { get; set; } = 4;

    /// <summary>(drafted, accepted, rounds) over the last Generate — acceptance ratio is the
    /// speed lever: ~accepted+1 target tokens per round vs 1 without speculation.</summary>
    public (int Drafted, int Accepted, int Rounds) LastStats { get; private set; }

    public SpeculativeRunner(StreamingEngine target, StreamingEngine draft)
    {
        Target = target;
        Draft = draft;
        if (Target.Config.VocabSize != Draft.Config.VocabSize)
            throw new InvalidOperationException(
                $"draft vocab {Draft.Config.VocabSize} != target vocab {Target.Config.VocabSize} — token ids must be comparable");
    }

    /// <summary>Greedy speculative generation. The target's choice at every position goes
    /// through <paramref name="targetSampler"/> when given (must be greedy: temperature 0 — its
    /// repeat penalty/window are honored, which is what makes the output identical to plain
    /// greedy decoding). onToken(id, indexSoFar) streams as tokens are committed.</summary>
    public int[] Generate(int[] prompt, int maxNew, Sampler? targetSampler = null, Action<int, int>? onToken = null)
    {
        if (targetSampler != null && targetSampler.Temperature > 0f)
            throw new InvalidOperationException(
                "speculative decoding verifies GREEDY — the target sampler must use temperature 0 (sampling-with-distribution acceptance is not implemented)");
        var sw = Stopwatch.StartNew();
        var generated = new List<int>(maxNew);
        var recent = new List<int>(prompt.Length + maxNew);
        recent.AddRange(prompt);

        // prime both engines over the prompt (identical positions)
        float[] lt = Target.Forward(prompt[0], 0);
        float[] ld = Draft.Forward(prompt[0], 0);
        for (int pos = 1; pos < prompt.Length; pos++)
        {
            lt = Target.Forward(prompt[pos], pos);
            ld = Draft.Forward(prompt[pos], pos);
        }
        double promptSecs = sw.Elapsed.TotalSeconds;

        int pos1 = prompt.Length;                 // both caches now hold 0..pos1-1; lt/ld predict pos1
        int drafted = 0, accepted = 0, rounds = 0;
        var proposed = new int[DraftTokens];

        while (generated.Count < maxNew && pos1 < Target.MaxCtx)
        {
            int tp = Choose(lt, recent, targetSampler);       // target's greedy choice at pos1
            if (tp == 0) break;                   // EOS — nothing to verify against

            rounds++;
            int k = Math.Min(DraftTokens, maxNew - generated.Count);

            // ---- draft phase: propose up to k tokens (draft advances its own caches) ----
            long snap = Draft.SnapshotRecurrentState();
            int k2 = 0;
            while (k2 < k)
            {
                int t = Argmax(ld);
                if (t == 0) break;                // draft hit EOS: propose only what came before
                proposed[k2++] = t;
                if (k2 < k) ld = Draft.Forward(t, pos1 + k2 - 1);
            }
            drafted += k2;

            // ---- verify: accept while the target's greedy token agrees ----
            int acc = 0;
            while (acc < k2 && generated.Count < maxNew && tp == proposed[acc])
            {
                generated.Add(proposed[acc]);
                recent.Add(proposed[acc]);
                onToken?.Invoke(proposed[acc], generated.Count);
                acc++;
                lt = Target.Forward(proposed[acc - 1], pos1 + acc - 1);   // commit on the target
                tp = Choose(lt, recent, targetSampler);
            }
            accepted += acc;

            if (generated.Count >= maxNew) break;

            if (acc == k2)
            {
                // full acceptance: lt now predicts pos1+k2 — one bonus token for free
                int bonus = tp;
                pos1 += k2;
                if (bonus == 0) break;            // target says EOS after the accepted run
                generated.Add(bonus);
                recent.Add(bonus);
                onToken?.Invoke(bonus, generated.Count);
                lt = Target.Forward(bonus, pos1);       // target cache -> 0..pos1
                ld = Draft.Forward(bonus, pos1);        // draft re-syncs on the bonus too
                pos1++;
            }
            else
            {
                // mismatch at acc: tp is the target's correction at pos1+acc
                if (tp == 0) break;                   // EOS correction: stop without emitting
                generated.Add(tp);
                recent.Add(tp);
                onToken?.Invoke(tp, generated.Count);
                int corrPos = pos1 + acc;
                // draft rewinds to the pre-round state and replays the accepted run + correction
                Draft.RestoreRecurrentState();
                Draft.TruncateTo(pos1);
                for (int t = 0; t < acc; t++) ld = Draft.Forward(proposed[t], pos1 + t);
                ld = Draft.Forward(tp, corrPos);
                lt = Target.Forward(tp, corrPos);       // target never consumed the bad token
                pos1 = corrPos + 1;
            }
        }

        sw.Stop();
        LastStats = (drafted, accepted, rounds);
        double genSecs = Math.Max(sw.Elapsed.TotalSeconds - promptSecs, 1e-9);
        Target.PublishStats(new GenerationStats(prompt.Length, generated.Count, promptSecs,
            genSecs, generated.Count / genSecs,
            Target.Prefetch?.BytesStreamed ?? 0,
            Target.Prefetch != null ? Target.Prefetch.WaitTicks * 1000.0 / Stopwatch.Frequency : 0,
            Target.AttentionStateBytes));
        return generated.ToArray();
    }

    /// <summary>The target's choice at the current position: through the sampler (greedy +
    /// repeat penalty — identical to what engine.Generate would pick) or raw argmax.</summary>
    private static int Choose(float[] logits, List<int> recent, Sampler? sampler)
        => sampler == null
            ? Argmax(logits)
            : sampler.Sample(logits, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(recent));

    private static int Argmax(ReadOnlySpan<float> v)
    {
        int best = 0;
        float bv = v[0];
        for (int i = 1; i < v.Length; i++)
            if (v[i] > bv) { bv = v[i]; best = i; }
        return best;
    }
}
