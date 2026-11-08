using System.Buffers.Binary;
using System.Text;

namespace QuantRunner.Core.Importers;

/// <summary>Minimal GGUF reader (v2/v3) — streams metadata + tensor directory.</summary>
public sealed class GgufFile : IDisposable
{
    public long Version { get; private set; }
    public Dictionary<string, object> Kvs { get; } = new();
    public List<GgufTensorInfo> Tensors { get; } = new();
    public long DataStart { get; private set; }
    public FileStream Stream { get; private set; }

    // raw metadata (KV) section range — preserved verbatim when importing to .qw so that
    // qw -> gguf / qw -> HF exports can restore the tokenizer and arch metadata losslessly
    public const long KvSectionStart = 24;              // magic(4) + version(4) + 2x u64 counts
    public long KvCount { get; private set; }
    public long KvSectionEnd { get; private set; }

    /// <summary>Copy of the original GGUF KV section bytes (GGUF wire format, counts excluded).</summary>
    public byte[] ReadKvSection()
    {
        long len = KvSectionEnd - KvSectionStart;
        if (len <= 0) return Array.Empty<byte>();
        var buf = new byte[len];
        Stream.Position = KvSectionStart;
        int got = 0;
        while (got < buf.Length)
        {
            int r = Stream.Read(buf, got, buf.Length - got);
            if (r <= 0) throw new EndOfStreamException("gguf kv section");
            got += r;
        }
        return buf;
    }

    /// <summary>Parse a raw GGUF KV section (wire format: (key, type, value)*, the counts that
    /// normally precede it excluded) back into a key -> value dictionary. This is the inverse of
    /// what a .qw header stores under gguf_kvs, so embedded tokenizer/architecture metadata can
    /// be restored without the original .gguf file.</summary>
    public static Dictionary<string, object> ParseKvSection(byte[] kvBytes)
    {
        var kvs = new Dictionary<string, object>();
        using var ms = new MemoryStream(kvBytes, writable: false);
        using var br = new BinaryReader(ms, Encoding.UTF8, leaveOpen: false);
        while (ms.Position < ms.Length)
        {
            string key = ReadString(br);
            var vt = (GgufValueType)br.ReadUInt32();
            kvs[key] = ReadValue(br, vt);
        }
        return kvs;
    }

    public sealed record GgufTensorInfo(string Name, long[] Dims, int GgmlType, long Offset, long SizeBytes)
    {
        public long NumElements { get { long n = 1; foreach (var d in Dims) n *= d; return n; } }
    }

    public static GgufFile Open(string path)
    {
        var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20);
        var g = new GgufFile { Stream = fs };
        using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: true);

        Span<byte> magic = br.ReadBytes(4);
        if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F')
            throw new InvalidDataException($"{path}: not a GGUF file");
        g.Version = br.ReadUInt32();
        if (g.Version < 2) throw new InvalidDataException($"GGUF v{g.Version} not supported (need v2+)");

        ulong nTensors = br.ReadUInt64();
        ulong nKv = br.ReadUInt64();
        g.KvCount = (long)nKv;

        for (ulong i = 0; i < nKv; i++)
        {
            string key = ReadString(br);
            var vt = (GgufValueType)br.ReadUInt32();
            g.Kvs[key] = ReadValue(br, vt);
        }
        g.KvSectionEnd = br.BaseStream.Position;

        for (ulong i = 0; i < nTensors; i++)
        {
            string name = ReadString(br);
            int nDims = br.ReadInt32();
            var dims = new long[nDims];
            for (int d = 0; d < nDims; d++) dims[d] = br.ReadInt64();   // ne[0] = fastest
            int type = br.ReadInt32();
            long offset = br.ReadInt64();
            long bytes = TensorBytes(ggmlType: type, dims);
            g.Tensors.Add(new GgufTensorInfo(name, dims, type, offset, bytes));
        }

        long align = 32;
        if (g.Kvs.TryGetValue("general.alignment", out var a)) align = Convert.ToInt64(a);
        g.DataStart = (br.BaseStream.Position + align - 1) / align * align;
        return g;
    }

    public object? Kv(string key) => Kvs.TryGetValue(key, out var v) ? v : null;
    public long KvLong(string key, long def = 0) => Kv(key) is object v ? Convert.ToInt64(v) : def;
    public float KvFloat(string key, float def = 0) => Kv(key) is object v ? Convert.ToSingle(v) : def;

    public void Dispose() => Stream?.Dispose();

    // ---------------- GGML type sizes ----------------

    // block layouts per ggml-common.h: Q4_0 {f16 d, u8 qs[16]} = 18 B; Q4_1 adds f16 m = 20 B;
    // Q5_0 {f16 d, u8 qh[4], u8 qs[16]} = 22 B; Q5_1 adds f16 m = 24 B. Q4_0 is NOT the same
    // block as QuantRunner's own Q4_0 (20 B, positive delta) — see GgmlDequant.
    public static (long blockSize, long blockBytes) GgmlBlockSpec(int t) => t switch
    {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        30 => (1, 2),     // BF16
        _ => (0, 0)
    };

    public static long TensorBytes(int ggmlType, long[] dims)
    {
        var (bs, bb) = GgmlBlockSpec(ggmlType);
        if (bs == 0) throw new NotSupportedException($"GGML type {ggmlType} not supported by importer");
        long n = 1; foreach (var d in dims) n *= d;
        return n / bs * bb;
    }

    public static string GgmlTypeName(int t) => t switch
    {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1",
        8 => "Q8_0", 10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K", 30 => "BF16",
        _ => $"ggml:{t}"
    };

    // ---------------- primitives ----------------

    public enum GgufValueType : uint
    {
        U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, Bool = 7,
        String = 8, Array = 9, U64 = 10, I64 = 11, F64 = 12
    }

    internal static string ReadString(BinaryReader br)
    {
        int len = (int)br.ReadUInt64();
        var bytes = br.ReadBytes(len);
        return Encoding.UTF8.GetString(bytes);
    }

    internal static object ReadValue(BinaryReader br, GgufValueType t) => t switch
    {
        GgufValueType.U8 => br.ReadByte(),
        GgufValueType.I8 => br.ReadSByte(),
        GgufValueType.U16 => br.ReadUInt16(),
        GgufValueType.I16 => br.ReadInt16(),
        GgufValueType.U32 => br.ReadUInt32(),
        GgufValueType.I32 => br.ReadInt32(),
        GgufValueType.F32 => br.ReadSingle(),
        GgufValueType.Bool => br.ReadBoolean(),
        GgufValueType.String => ReadString(br),
        GgufValueType.U64 => br.ReadUInt64(),
        GgufValueType.I64 => br.ReadInt64(),
        GgufValueType.F64 => br.ReadDouble(),
        GgufValueType.Array => ReadArray(br),
        _ => throw new InvalidDataException($"gguf value type {t}")
    };

    internal static object ReadArray(BinaryReader br)
    {
        var elemType = (GgufValueType)br.ReadUInt32();
        ulong count = br.ReadUInt64();
        if (elemType == GgufValueType.String)
        {
            var arr = new string[count];
            for (ulong i = 0; i < count; i++) arr[i] = ReadString(br);
            return arr;
        }
        switch (elemType)
        {
            case GgufValueType.U32: { var a = new uint[count]; for (ulong i = 0; i < count; i++) a[i] = br.ReadUInt32(); return a; }
            case GgufValueType.I32: { var a = new int[count]; for (ulong i = 0; i < count; i++) a[i] = br.ReadInt32(); return a; }
            case GgufValueType.F32: { var a = new float[count]; for (ulong i = 0; i < count; i++) a[i] = br.ReadSingle(); return a; }
            case GgufValueType.I64: { var a = new long[count]; for (ulong i = 0; i < count; i++) a[i] = br.ReadInt64(); return a; }
            case GgufValueType.U64: { var a = new ulong[count]; for (ulong i = 0; i < count; i++) a[i] = br.ReadUInt64(); return a; }
            case GgufValueType.F64: { var a = new double[count]; for (ulong i = 0; i < count; i++) a[i] = br.ReadDouble(); return a; }
            default:
                {
                    // small types: read as long
                    var a = new long[count];
                    for (ulong i = 0; i < count; i++) a[i] = Convert.ToInt64(ReadValue(br, elemType));
                    return a;
                }
        }
    }
}
