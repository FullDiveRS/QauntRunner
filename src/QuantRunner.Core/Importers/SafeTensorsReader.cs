using System.Text;
using System.Text.Json;

namespace QuantRunner.Core.Importers;

/// <summary>HuggingFace safetensors reader (header JSON + raw fp data).</summary>
public sealed class SafeTensorsFile : IDisposable
{
    public sealed record TensorInfo(string Name, string Dtype, long[] Shape, long DataStart, long DataEnd);

    private readonly FileStream _fs;
    public Dictionary<string, TensorInfo> Tensors { get; } = new();
    public long DataOffset { get; private set; }

    private SafeTensorsFile(FileStream fs) { _fs = fs; }

    public static SafeTensorsFile Open(string path)
    {
        var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20);
        var st = new SafeTensorsFile(fs);
        Span<byte> lenBuf = stackalloc byte[8];
        if (fs.Read(lenBuf) != 8) throw new InvalidDataException("safetensors: too short");
        long headerLen = BitConverter.ToInt64(lenBuf);
        if (headerLen < 2 || headerLen > 500_000_000) throw new InvalidDataException("safetensors: bad header len");
        var headerBytes = new byte[headerLen];
        int read = 0;
        while (read < headerLen)
        {
            int r = fs.Read(headerBytes, read, (int)(headerLen - read));
            if (r <= 0) throw new EndOfStreamException();
            read += r;
        }
        st.DataOffset = 8 + headerLen;

        var doc = JsonDocument.Parse(headerBytes);
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            if (prop.Name == "__metadata__") continue;
            var v = prop.Value;
            var shape = new List<long>();
            foreach (var s in v.GetProperty("shape").EnumerateArray()) shape.Add(s.GetInt64());
            var offs = v.GetProperty("data_offsets");
            long start = offs[0].GetInt64(), end = offs[1].GetInt64();
            st.Tensors[prop.Name] = new TensorInfo(prop.Name, v.GetProperty("dtype").GetString()!,
                shape.ToArray(), start, end);
        }
        return st;
    }

    /// <summary>Element size in bytes for the dtype (1 for fp8).</summary>
    public static int ElemSize(string dtype) => dtype switch
    {
        "F32" => 4, "F16" => 2, "BF16" => 2, "F64" => 8, "F8_E4M3" => 1, _ => 0
    };

    /// <summary>Read a chunk of a tensor as float32 starting at element `elemStart`.
    /// For F8_E4M3 tensors pass the companion `*.weight_scale_inv` tensor (DeepSeek/GLM fp8
    /// block quantization, 128x128 blocks): value = fp8(w) * scale[row/128, col/128].</summary>
    public unsafe void ReadChunkF32(TensorInfo t, long elemStart, Span<float> dst, TensorInfo? scaleInv = null)
    {
        int elemSize = ElemSize(t.Dtype);
        if (elemSize == 0)
            throw new NotSupportedException($"safetensors dtype {t.Dtype} for tensor '{t.Name}'");
        if (t.Dtype == "F8_E4M3" && scaleInv == null)
            throw new InvalidDataException($"fp8 tensor '{t.Name}' has no weight_scale_inv companion");

        float[]? scales = null;
        int scaleCols = 0, cols = 1;
        if (t.Dtype == "F8_E4M3")
        {
            scales = ReadTensorF32(scaleInv!);           // tiny [rows/128, cols/128] grid
            scaleCols = (int)scaleInv!.Shape[^1];
            cols = (int)t.Shape[^1];
        }

        _fs.Position = DataOffset + t.DataStart + elemStart * elemSize;
        int count = dst.Length;
        var buf = new byte[count * elemSize];
        int got = 0;
        while (got < buf.Length)
        {
            int r = _fs.Read(buf, got, buf.Length - got);
            if (r <= 0) throw new EndOfStreamException();
            got += r;
        }
        fixed (byte* p = buf)
        fixed (float* sc = scales)
        {
            switch (t.Dtype)
            {
                case "F32":
                    {
                        float* s = (float*)p;
                        for (int i = 0; i < count; i++) dst[i] = s[i];
                        break;
                    }
                case "F16":
                    {
                        ushort* s = (ushort*)p;
                        for (int i = 0; i < count; i++) dst[i] = Dtypes.F16ToFloat(s[i]);
                        break;
                    }
                case "BF16":
                    {
                        ushort* s = (ushort*)p;
                        for (int i = 0; i < count; i++) dst[i] = Dtypes.Bf16ToFloat(s[i]);
                        break;
                    }
                case "F8_E4M3":
                    {
                        // row-major element index -> 128x128 block scale
                        for (int i = 0; i < count; i++)
                        {
                            long g = elemStart + i;
                            long row = g / cols, col = g % cols;
                            float scale = sc[(row / 128) * scaleCols + col / 128];
                            dst[i] = Dtypes.F8E4M3ToFloat(p[i]) * scale;
                        }
                        break;
                    }
                default:
                    {
                        double* s = (double*)p;
                        for (int i = 0; i < count; i++) dst[i] = (float)s[i];
                        break;
                    }
            }
        }
    }

    /// <summary>Read one tensor as float32 (converts F16/BF16). Throws for integer dtypes.</summary>
    public unsafe float[] ReadTensorF32(TensorInfo t)
    {
        long n = 1; foreach (var d in t.Shape) n *= d;
        var result = new float[n];
        _fs.Position = DataOffset + t.DataStart;
        int elemSize = t.Dtype switch { "F32" => 4, "F16" => 2, "BF16" => 2, "F64" => 8, _ => 0 };
        if (elemSize == 0)
            throw new NotSupportedException($"safetensors dtype {t.Dtype} for tensor '{t.Name}' — only F32/F16/BF16/F64 supported");

        if (t.Dtype == "F32")
            Fill(_fs, result, 4, (byte* p, float* dst, int cnt) =>
            {
                float* sp = (float*)p;
                for (int i = 0; i < cnt; i++) dst[i] = sp[i];
            });
        else if (t.Dtype == "F16")
            Fill(_fs, result, 2, (byte* p, float* dst, int cnt) =>
            {
                ushort* sp = (ushort*)p;
                for (int i = 0; i < cnt; i++) dst[i] = Dtypes.F16ToFloat(sp[i]);
            });
        else if (t.Dtype == "BF16")
            Fill(_fs, result, 2, (byte* p, float* dst, int cnt) =>
            {
                ushort* sp = (ushort*)p;
                for (int i = 0; i < cnt; i++) dst[i] = Dtypes.Bf16ToFloat(sp[i]);
            });
        else
            Fill(_fs, result, 8, (byte* p, float* dst, int cnt) =>
            {
                double* sp = (double*)p;
                for (int i = 0; i < cnt; i++) dst[i] = (float)sp[i];
            });
        return result;
    }

    private unsafe delegate void ConvertFn(byte* bytes, float* dst, int count);

    private static unsafe void Fill(FileStream fs, Span<float> dst, int elemSize, ConvertFn convert)
    {
        var buf = new byte[1 << 20];
        int done = 0;
        int total = dst.Length;
        while (done < total)
        {
            int count = (int)Math.Min(buf.Length / elemSize, total - done);
            int bytesNeeded = count * elemSize;
            int got = 0;
            while (got < bytesNeeded)
            {
                int r = fs.Read(buf, got, bytesNeeded - got);
                if (r <= 0) throw new EndOfStreamException();
                got += r;
            }
            fixed (byte* bp = buf)
            fixed (float* dp = dst.Slice(done))
                convert(bp, dp, count);
            done += count;
        }
    }

    public void Dispose() => _fs.Dispose();
}
