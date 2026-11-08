using System.Runtime.InteropServices;
using System.Threading;
using QuantRunner.Core.Memory;

namespace QuantRunner.Core.Runtime;

/// <summary>
/// Async layer prefetch pipeline — QuantRunner's headline feature over AirLLM.
///
/// AirLLM synchronously streams each layer's weights from disk, so every layer pays the
/// full disk latency on the critical path. QuantRunner instead kicks off a background
/// positional read of layer N+1 the moment layer N's weights are ready, overlapping
/// 100% of the I/O with compute. Weights land in a pinned ping-pong staging buffer and
/// the compute path reads them through plain pointers — no page faults mid-matmul.
///
/// If a layer exceeds the byte budget (huge MoE expert tensors) or the pipeline is
/// disabled, <see cref="Acquire"/> transparently falls back to the zero-copy mmap view.
/// </summary>
public sealed unsafe class PrefetchPipeline : IDisposable
{
    private sealed class Slot
    {
        public byte[]? Buffer;
        public GCHandle Pin;
        public byte* Ptr;
        public long Bytes;
        public int ForLayer = int.MinValue;
        public bool Staged;
        public Task Task = Task.CompletedTask;
        public Dictionary<string, long> Offsets = new();
    }

    private readonly ModelStore _store;
    private readonly long _budgetBytes;
    private readonly FileStream _fs;
    private readonly Slot[] _slots = { new(), new() };
    private readonly string[] _tensorNames;      // stable iteration order
    private bool _disposed;

    public long BytesStreamed => Interlocked.Read(ref _bytesStreamed);  // total bytes staged from disk
    public long WaitTicks { get; private set; }          // time the compute thread spent waiting on I/O
    public long ReadTicks { get; private set; }          // time spent inside background reads
    public int StagedLayers => _stagedLayers;
    public int SkippedLayers => _skippedLayers;          // over budget -> mmap fallback

    private long _bytesStreamed;
    private int _stagedLayers;
    private int _skippedLayers;

    public bool Enabled => _budgetBytes > 0;

    public PrefetchPipeline(ModelStore store, long budgetBytes)
    {
        _store = store;
        _budgetBytes = budgetBytes;
        _tensorNames = new string[store.Tensors.Count];
        int i = 0;
        foreach (var name in store.Tensors.Keys) _tensorNames[i++] = name;
        _fs = new FileStream(store.Path, FileMode.Open, FileAccess.Read, FileShare.Read,
            1 << 16, FileOptions.Asynchronous | FileOptions.SequentialScan);
    }

    /// <summary>Start the background read of a layer into staging slot (layer &amp; 1).</summary>
    public void Begin(int layer)
    {
        if (!Enabled) return;
        int s = layer & 1;
        var slot = _slots[s];
        // wait for any previous use of this slot to finish before recycling it
        slot.Task.Wait();
        slot.ForLayer = layer;
        slot.Staged = false;
        slot.Offsets.Clear();
        slot.Task = Task.Run(() => ReadLayer(layer, slot));
    }

    /// <summary>
    /// Resolve the byte pointer for a tensor of <paramref name="layer"/>, waiting for the
    /// prefetch if this layer is being staged. Returns the mmap pointer when staging is
    /// disabled or the layer exceeded the budget.
    /// </summary>
    public byte* Acquire(string name, int layer)
    {
        var slot = _slots[layer & 1];
        if (slot.ForLayer != layer) return _store.Get(name).Ptr;   // this layer was never staged

        if (!slot.Task.IsCompleted)
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            slot.Task.Wait();
            WaitTicks += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
        }
        if (!slot.Staged) return _store.Get(name).Ptr;             // over budget / read failed -> mmap
        return slot.Offsets.TryGetValue(name, out long off)
            ? slot.Ptr + off
            : _store.Get(name).Ptr;   // not part of the layer (e.g. norms fetched separately)
    }

    /// <summary>Wait for a pending prefetch of this layer without resolving tensors (for stats).</summary>
    public void WaitLayer(int layer)
    {
        var slot = _slots[layer & 1];
        if (slot.ForLayer == layer && !slot.Task.IsCompleted)
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            slot.Task.Wait();
            WaitTicks += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
        }
    }

    private void ReadLayer(int layer, Slot slot)
    {
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        try
        {
            string prefix = layer < 0 ? "" : $"l{layer}.";
            long total = 0;
            var names = new List<string>();
            var views = new List<TensorView>();
            foreach (var name in _tensorNames)
            {
                if (prefix.Length > 0 && !name.StartsWith(prefix, StringComparison.Ordinal)) continue;
                if (name.EndsWith("norm", StringComparison.Ordinal) || name.EndsWith("router", StringComparison.Ordinal))
                    continue; // tiny tensors: leave them on the mmap
                var v = _store.Get(name);
                total += v.SizeBytes;
                names.Add(name);
                views.Add(v);
            }

            if (total == 0 || total > _budgetBytes)
            {
                Interlocked.Increment(ref _skippedLayers);
                return;
            }

            if (slot.Buffer == null || slot.Bytes < total)
            {
                if (slot.Pin.IsAllocated) slot.Pin.Free();
                slot.Buffer = new byte[total];
                slot.Pin = GCHandle.Alloc(slot.Buffer, GCHandleType.Pinned);
                slot.Ptr = (byte*)slot.Pin.AddrOfPinnedObject();
                slot.Bytes = total;
            }

            long off = 0;
            for (int i = 0; i < names.Count; i++)
            {
                _fs.Seek(_store.OffsetOf(views[i]), SeekOrigin.Begin);
                ReadExact(slot.Buffer, off, views[i].SizeBytes);
                slot.Offsets[names[i]] = off;
                off += views[i].SizeBytes;
            }
            Interlocked.Add(ref _bytesStreamed, total);
            slot.Staged = true;
            Interlocked.Increment(ref _stagedLayers);
        }
        catch (Exception)
        {
            slot.Staged = false;   // graceful: compute falls back to mmap
        }
        finally
        {
            ReadTicks += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
        }
    }

    private void ReadExact(byte[] buf, long offset, long count)
    {
        int done = 0;
        while (done < count)
        {
            int r = _fs.Read(buf, (int)(offset + done), (int)(count - done));
            if (r <= 0) throw new EndOfStreamException("prefetch: unexpected EOF");
            done += r;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var slot in _slots)
        {
            try { slot.Task.Wait(50); } catch { /* ignore */ }
            if (slot.Pin.IsAllocated) slot.Pin.Free();
        }
        _fs.Dispose();
    }
}
