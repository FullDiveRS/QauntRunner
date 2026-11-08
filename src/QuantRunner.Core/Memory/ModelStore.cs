using System.IO.MemoryMappedFiles;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using QuantRunner.Core.Models;

namespace QuantRunner.Core.Memory;

/// <summary>A read-only view over a tensor stored in a memory-mapped .qw file.</summary>
public sealed unsafe class TensorView
{
    public required string Name { get; init; }
    public required long[] Dims { get; init; }
    public required QuantFormat Format { get; init; }
    public required byte* Ptr { get; init; }        // absolute pointer into mmap
    public required long SizeBytes { get; init; }
    public long NumElements
    {
        get
        {
            long n = 1;
            foreach (var d in Dims) n *= d;
            return n;
        }
    }

    public ReadOnlySpan<byte> Bytes => new(Ptr, (int)SizeBytes);

    /// <summary>Pointer to the start of row `r` of a 2-D tensor [rows, cols].</summary>
    public byte* RowPtr(long r)
    {
        long cols = Dims[^1];
        var (bs, bb) = Dtypes.BlockSpec(Format);
        return Ptr + r * (cols / bs) * bb;
    }

    public long RowBytes()
    {
        long cols = Dims[^1];
        var (bs, bb) = Dtypes.BlockSpec(Format);
        return cols / bs * bb;
    }
}

/// <summary>
/// QuantRunner Weight file (.qw): a streaming-first, 64-bit-offset, zero-copy format.
///
///   [0..3]   magic "QWR1"
///   [4..7]   int32 header length
///   [8..]    UTF-8 JSON header: { Config, data_offset?, Tensors[{Name, Dims, Format, Offset(rel to data start), SizeBytes}] }
///   [data..] tensor data, first tensor 64-byte aligned
///
/// Offsets are long (supports multi-TB models). Tensors are read directly from the mmap —
/// no copies, no loading step, so a 2.8T checkpoint streams from disk exactly like a 7B one.
///
/// When the header contains an absolute "data_offset" field, the header region is RESERVED:
/// the JSON can grow (new tensors appended later) without moving the data region, so shards
/// can be assembled into one .qw with ModelStoreAppender — zero bytes of tensor data copied.
/// </summary>
public sealed unsafe class ModelStore : IDisposable
{
    public const string Magic = "QWR1";
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _acc;
    private byte* _basePtr;
    private readonly long _dataStart;
    public string Path { get; }
    public JsonElement ConfigJson { get; }
    public IReadOnlyDictionary<string, TensorView> Tensors { get; }

    private ModelStore(string path, MemoryMappedFile mmf, MemoryMappedViewAccessor acc, byte* basePtr,
        long dataStart, JsonElement configJson, Dictionary<string, TensorView> tensors)
    {
        Path = path; _mmf = mmf; _acc = acc; _basePtr = basePtr; _dataStart = dataStart;
        ConfigJson = configJson; Tensors = tensors;
    }

    public bool Has(string name) => Tensors.ContainsKey(name);
    public TensorView Get(string name) => Tensors.TryGetValue(name, out var t)
        ? t : throw new KeyNotFoundException($"tensor '{name}' not found in {Path}");

    /// <summary>File offset of a tensor's data (for positional reads that bypass the mmap).</summary>
    public long OffsetOf(TensorView t) => (long)(t.Ptr - _basePtr);

    public static unsafe ModelStore Open(string path)
    {
        var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        var acc = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        byte* ptr = null;
        acc.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);

        if (ptr[0] != 'Q' || ptr[1] != 'W' || ptr[2] != 'R' || ptr[3] != '1')
            throw new InvalidDataException($"{path}: not a QuantRunner .qw file");
        int headerLen = BitConverter.ToInt32(new ReadOnlySpan<byte>(ptr + 4, 4));
        var headerBytes = new byte[headerLen];
        new ReadOnlySpan<byte>(ptr + 8, headerLen).CopyTo(headerBytes);

        var doc = JsonDocument.Parse(headerBytes);
        var root = doc.RootElement;
        // explicit data_offset (reserved-header files) wins; otherwise derive from header length
        long dataStart = root.TryGetProperty("data_offset", out var dOff) && dOff.TryGetInt64(out long dOffVal)
            ? dOffVal
            : 8L + headerLen;
        dataStart = (dataStart + 63) & ~63L;   // align data to 64
        if (dataStart < 8L + headerLen)
            throw new InvalidDataException($"{path}: corrupt header (data_offset {dataStart} < header end {8L + headerLen})");

        var tensors = new Dictionary<string, TensorView>();
        foreach (var te in root.GetProperty("tensors").EnumerateArray())
        {
            var dims = new List<long>();
            foreach (var d in te.GetProperty("dims").EnumerateArray()) dims.Add(d.GetInt64());
            long offset = te.GetProperty("offset").GetInt64();
            long size = te.GetProperty("size").GetInt64();
            var fmt = (QuantFormat)te.GetProperty("format").GetInt32();
            tensors[te.GetProperty("name").GetString()] = new TensorView
            {
                Name = te.GetProperty("name").GetString(),
                Dims = dims.ToArray(),
                Format = fmt,
                Ptr = ptr + dataStart + offset,
                SizeBytes = size
            };
        }
        return new ModelStore(path, mmf, acc, ptr, dataStart, root.GetProperty("config").Clone(), tensors);
    }

    public void Dispose()
    {
        _acc.Dispose();
        _mmf.Dispose();
    }
}

/// <summary>Header entry for building a .qw file.</summary>
public sealed record TensorPlan(string Name, long[] Dims, QuantFormat Format)
{
    public long NumElements { get { long n = 1; foreach (var d in Dims) n *= d; return n; } }
    public long SizeBytes => Dtypes.TensorBytes(NumElements, Format);
}

/// <summary>Streams tensors into a .qw file. Sizes are computed up-front so the header is exact.</summary>
public sealed unsafe class ModelStoreBuilder : IDisposable
{
    private readonly FileStream _fs;
    private readonly BinaryWriter _bw;
    private readonly List<TensorPlan> _plans = new();
    private readonly JsonElement _config;
    private readonly long _headerReserveBytes;
    private bool _headerWritten;

    /// <summary>
    /// headerReserveBytes > 0 pads the header region so tensors can be appended later
    /// (an absolute "data_offset" is recorded; see ModelStoreAppender).
    /// </summary>
    public ModelStoreBuilder(string path, JsonElement config, long headerReserveBytes = 0)
    {
        _fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20);
        _bw = new BinaryWriter(_fs);
        _config = config;
        _headerReserveBytes = Math.Max(0, headerReserveBytes);
    }

    public TensorPlan Plan(string name, long[] dims, QuantFormat fmt)
    {
        if (_headerWritten) throw new InvalidOperationException("cannot plan after writing started");
        var p = new TensorPlan(name, dims, fmt);
        _plans.Add(p);
        return p;
    }

    /// <summary>Write header, then stream tensor bodies in plan order.</summary>
    public void Commit(Action<TensorPlan, Stream> writeTensorData)
    {
        if (_headerWritten) throw new InvalidOperationException("already committed");
        _headerWritten = true;

        // header JSON
        long offset = 0;
        var entries = new object[_plans.Count];
        for (int i = 0; i < _plans.Count; i++)
        {
            var p = _plans[i];
            entries[i] = new Dictionary<string, object>
            {
                ["name"] = p.Name,
                ["dims"] = p.Dims,
                ["format"] = (int)p.Format,
                ["offset"] = offset,
                ["size"] = p.SizeBytes
            };
            offset += p.SizeBytes;
        }
        var headerObj = new Dictionary<string, object>
        {
            ["config"] = JsonSerializer.Deserialize<JsonElement>(_config.GetRawText()),
            ["tensors"] = entries
        };
        var headerBytes = JsonSerializer.SerializeToUtf8Bytes(headerObj);

        long dataStart;
        if (_headerReserveBytes > 0)
        {
            dataStart = (8 + _headerReserveBytes + 63) & ~63L;
            if (8 + headerBytes.Length > dataStart)
                throw new IOException($"header needs {headerBytes.Length} bytes, reserved region only {_headerReserveBytes}");
            headerObj["data_offset"] = dataStart;
            headerBytes = JsonSerializer.SerializeToUtf8Bytes(headerObj);   // re-serialize with data_offset
        }
        else
        {
            dataStart = (8 + headerBytes.Length + 63) & ~63L;
        }

        _bw.Write(Encoding.ASCII.GetBytes(ModelStore.Magic));
        _bw.Write(headerBytes.Length);
        _bw.Write(headerBytes);
        while (_fs.Position < dataStart) _bw.Write((byte)0);

        foreach (var p in _plans)
        {
            long before = _fs.Position;
            writeTensorData(p, _fs);
            long written = _fs.Position - before;
            if (written != p.SizeBytes)
                throw new IOException($"tensor '{p.Name}': wrote {written} bytes, planned {p.SizeBytes}");
        }
        _bw.Flush();
    }

    public void Dispose() { _bw?.Dispose(); _fs?.Dispose(); }
}

/// <summary>
/// Adds tensors to an existing .qw that was built with a reserved header region
/// (absolute "data_offset" in the JSON). The header is rewritten in place and new
/// tensor bodies are appended at end-of-file — existing tensor data is NEVER copied,
/// so a sharded multi-TB checkpoint is assembled one shard at a time:
///
///   quantize model-00001-of-00062.safetensors -o model.qw
///   quantize --append model-00002-of-00062.safetensors -o model.qw
///   ...
/// </summary>
public sealed unsafe class ModelStoreAppender : IDisposable
{
    private readonly FileStream _fs;
    private JsonElement _config;                             // mutable via ReplaceConfig
    private readonly long _dataOffset;                       // absolute file offset of data region
    private readonly List<(TensorPlan Plan, long Offset)> _existing = new();
    private readonly List<(TensorPlan Plan, long Offset)> _added = new();
    private bool _committed;

    public string Path { get; }
    public int ExistingCount => _existing.Count;
    public JsonElement ConfigJson => _config;

    /// <summary>Replace the stored header config (used by --append healing so the healed
    /// values are persisted, not just applied in memory).</summary>
    public void ReplaceConfig(ModelConfig cfg)
    {
        _config = JsonDocument.Parse(cfg.ToJson()).RootElement.Clone();
    }

    public ModelStoreAppender(string path)
    {
        Path = path;
        _fs = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None, 1 << 20);

        Span<byte> magic = stackalloc byte[4];
        if (_fs.Read(magic) != 4 || magic[0] != 'Q' || magic[1] != 'W' || magic[2] != 'R' || magic[3] != '1')
            throw new InvalidDataException($"{path}: not a QuantRunner .qw file");
        Span<byte> lenBuf = stackalloc byte[4];
        if (_fs.Read(lenBuf) != 4) throw new InvalidDataException($"{path}: truncated header");
        int headerLen = BitConverter.ToInt32(lenBuf);
        if (headerLen < 2 || 8L + headerLen > _fs.Length)
            throw new InvalidDataException($"{path}: corrupt header length {headerLen}");
        var headerBytes = new byte[headerLen];
        int got = 0;
        while (got < headerLen)
        {
            int r = _fs.Read(headerBytes, got, headerLen - got);
            if (r <= 0) throw new EndOfStreamException($"{path}: header");
            got += r;
        }

        var doc = JsonDocument.Parse(headerBytes);
        _config = doc.RootElement.GetProperty("config").Clone();
        if (!doc.RootElement.TryGetProperty("data_offset", out var dOff) || !dOff.TryGetInt64(out _dataOffset))
            throw new InvalidDataException(
                $"{path}: was created without a reserved header region — tensors cannot be appended. " +
                "Re-create it with: quantize --append <first-shard> -o " + System.IO.Path.GetFileName(path));
        _dataOffset = (_dataOffset + 63) & ~63L;
        if (_dataOffset < 8L + headerLen || _dataOffset > _fs.Length)
            throw new InvalidDataException($"{path}: corrupt data_offset {_dataOffset}");

        long dataLen = _fs.Length - _dataOffset;
        foreach (var te in doc.RootElement.GetProperty("tensors").EnumerateArray())
        {
            var dims = new List<long>();
            foreach (var d in te.GetProperty("dims").EnumerateArray()) dims.Add(d.GetInt64());
            long offset = te.GetProperty("offset").GetInt64();
            long size = te.GetProperty("size").GetInt64();
            if (offset + size > dataLen)
                throw new InvalidDataException($"{path}: tensor '{te.GetProperty("name")}' extends past end of file (truncated?)");
            _existing.Add((new TensorPlan(te.GetProperty("name").GetString()!, dims.ToArray(),
                (QuantFormat)te.GetProperty("format").GetInt32()), offset));
        }
    }

    public bool Has(string name) =>
        _existing.Any(e => e.Plan.Name == name) || _added.Any(e => e.Plan.Name == name);

    /// <summary>Existing tensor plan by canonical name, or null.</summary>
    public TensorPlan? Find(string name) =>
        _existing.Concat(_added).FirstOrDefault(e => e.Plan.Name == name).Plan;

    private long DataEndRelative => _existing.Concat(_added).Select(e => e.Offset + e.Plan.SizeBytes).DefaultIfEmpty(0).Max();

    /// <summary>Queue a new tensor. Throws if the name already exists.</summary>
    public TensorPlan Plan(string name, long[] dims, QuantFormat fmt)
    {
        if (_committed) throw new InvalidOperationException("already committed");
        if (Has(name)) throw new InvalidOperationException($"tensor '{name}' already present in {Path}");
        long rel = (DataEndRelative + 63) & ~63L;   // 64-byte align relative to data start
        var p = new TensorPlan(name, dims, fmt);
        _added.Add((p, rel));
        return p;
    }

    /// <summary>Rewrite the header (old + new tensors) in place, then stream new tensor bodies at EOF.</summary>
    public void Commit(Action<TensorPlan, Stream> writeTensorData)
    {
        if (_committed) throw new InvalidOperationException("already committed");
        _committed = true;

        var entries = new object[_existing.Count + _added.Count];
        int i = 0;
        foreach (var (plan, off) in _existing.Concat(_added))
        {
            entries[i++] = new Dictionary<string, object>
            {
                ["name"] = plan.Name,
                ["dims"] = plan.Dims,
                ["format"] = (int)plan.Format,
                ["offset"] = off,
                ["size"] = plan.SizeBytes
            };
        }
        var headerObj = new Dictionary<string, object>
        {
            ["config"] = JsonSerializer.Deserialize<JsonElement>(_config.GetRawText()),
            ["data_offset"] = _dataOffset,
            ["tensors"] = entries
        };
        var headerBytes = JsonSerializer.SerializeToUtf8Bytes(headerObj);
        if (8 + headerBytes.Length > _dataOffset)
            throw new IOException(
                $"{Path}: header needs {headerBytes.Length + 8} bytes but the reserved region is {_dataOffset} — " +
                "re-create the file with a larger --reserve-mb");

        // header rewrite in place (old data region untouched)
        _fs.Position = 0;
        _fs.Write(Encoding.ASCII.GetBytes(ModelStore.Magic), 0, 4);
        _fs.Write(BitConverter.GetBytes(headerBytes.Length), 0, 4);
        _fs.Write(headerBytes, 0, headerBytes.Length);
        long padEnd = _dataOffset;
        var zeros = new byte[4096];
        while (_fs.Position < padEnd)
        {
            int n = (int)Math.Min(zeros.Length, padEnd - _fs.Position);
            _fs.Write(zeros, 0, n);
        }

        // append new bodies right after the existing data (truncate any stale tail from a crashed run)
        // NOTE: DataEndRelative already includes the added plans — the write START is the first
        // added tensor's relative offset, not the new data end. Planned offsets are 64-byte
        // aligned (Plan()), so honor each tensor's planned position: pad/skip the alignment gap
        // that SetLength already zero-filled — writing back-to-back would place every tensor
        // after a non-64-multiple-sized one at the wrong offset (silent corruption).
        long finalLen = _dataOffset + DataEndRelative;
        _fs.SetLength(finalLen);
        foreach (var (plan, off) in _added)
        {
            long target = _dataOffset + off;
            if (_fs.Position > target)
                throw new IOException($"{Path}: internal error — write position {_fs.Position} passed planned offset {target} for '{plan.Name}'");
            _fs.Position = target;      // gap bytes are zeros (SetLength zero-fills the extension)
            long before = _fs.Position;
            writeTensorData(plan, _fs);
            long written = _fs.Position - before;
            if (written != plan.SizeBytes)
                throw new IOException($"tensor '{plan.Name}': wrote {written} bytes, planned {plan.SizeBytes}");
        }
        _fs.Flush(true);
    }

    public void Dispose() => _fs.Dispose();
}
