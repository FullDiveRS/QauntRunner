using System.Diagnostics;
using System.Text.Json;
using QuantRunner.Core;
using QuantRunner.Core.Importers;
using QuantRunner.Core.Memory;
using QuantRunner.Core.Models;
using QuantRunner.Core.Quantization;

namespace QuantRunner.Cli;

internal static class Program
{
    private static int Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        if (args.Length == 0) { Usage(); return 2; }
        try
        {
            switch (args[0])
            {
                case "convert": return CmdConvert(args[1..]);
                case "quantize": return Commands.CmdQuantize(args[1..]);
                case "requant": return Commands.CmdRequant(args[1..]);
                case "analyze": return Commands.CmdAnalyze(args[1..]);
                case "synthesize": return CmdSynthesize(args[1..]);
                case "run": return Commands.CmdRun(args[1..]);
                case "bench": return Commands.CmdBench(args[1..]);
                case "logits": return Commands.CmdLogits(args[1..]);
                case "selftest": return Commands.CmdSelftest();
                case "info": return Commands.CmdInfo(args[1..]);
                default: Usage(); return 2;
            }
        }
        catch (Exception ex)
        {
            var prev = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Error.WriteLine($"error: {ex.Message}");
            if (Environment.GetEnvironmentVariable("QR_DEBUG") == "1") Console.Error.WriteLine(ex.ToString());
            Console.ForegroundColor = prev;
            return 1;
        }
    }

    private static void Usage()
    {
        Console.WriteLine("""
        QuantRunner — streaming quantized LLM runner (C# / .NET 8)

          QuantRunner quantize model.safetensors [shard2 ...] -o model.qw [quant options]
          QuantRunner requant  model.qw -o small.qw --fmt tri [--dry-run] [--from-report r.json]
          QuantRunner analyze  model.safetensors [--formats tri,b1,b1x,t1e,...] [-o report.json]
          QuantRunner convert  <src> -o <dst>            (gguf | qw | hf-dir  ->  qw | gguf | hf-dir)
          QuantRunner synthesize model.qw --dim 512 --layers 4 ...   (synthetic test model)
            [--glm5]            GLM-5.x hybrid shape: mHC + KDA:MLA 3:1 (+ GLM-MoE with --experts)
          QuantRunner info     <model.qw | model.gguf | model.safetensors | hf-dir/>
                                                                     (config + full parameter counts)
          QuantRunner run      <model.qw | model.gguf> -p "prompt" [options]
                     (a gguf is converted once into a cached .qw; tokenizers embedded in
                      the model (GGUF metadata / gguf-imported .qw) load automatically —
                      --tokenizer is only needed for safetensors-quantized models)
            [--draft draft.qw]  greedy speculative decoding (output = target greedy, exactly)
          QuantRunner bench    model.qw [options]                    (disk stream + tok/s + AirLLM table)
          QuantRunner selftest                                       (no model needed)

        quantize: quantize .safetensors shard(s) into a streaming .qw model.
          start : quantize model-00001-of-00062.safetensors -o model.qw [--layers 80]
          append: quantize --append model-00002-of-00062.safetensors -o model.qw
          (repeat per shard, any order — new tensor data is appended at EOF, never copied;
           `info` reports completeness. Config comes from config.json beside the file or
           --config; nested 'text_config' is read automatically. With neither, the
           architecture is inferred from tensor names/shapes — pin a partial shard with
           --layers/--heads/--dim/... Passing --config to --append also heals a stored
           config that is missing fields)

        supported checkpoints: Llama-family naming (model.layers.*, model.language_model.*),
        GQA attention, Mixtral-style MoE (routed experts), fused qkv_proj / W_pack / GLM
        query_key_value, fused gate_up_proj (dense + experts), and GLM-5.x hybrid models
        (KDA linear attention + MLA latent attention + mHC hyper-connections + sigmoid/noaux_tc
        MoE + FP8-with-block-scales source weights). The DSA indexer is not needed for context
        <= index_topk (2048 for GLM-5.3-Flash); beyond that the engine runs full attention over
        the latent cache instead of sparse top-k. Unrecognized tensor names are always reported.

        other commands:
          logits model.qw --tokens 1,2,3   print last-position logits (numeric debugging)

        quantize options:
          --append          add tensors to an existing .qw instead of creating one
          --config <path>   explicit HuggingFace config.json (also heals on --append)
          --fmt q4_k        shorthand: sets both --qattn and --qffn
          --layers/--heads/--kv-heads/--head-dim/--dim/--ffn/--vocab/--experts/--active N
          --reserve-mb 8    header space reserved for later --append entries
          + quant formats (--qattn/--qffn/--qemb/--qhead/--min-elems/--max-ctx)

        requant: requantize an existing .qw into a smaller .qw (dequant -> requant, one streaming pass).
          requant model.qw -o model-small.qw --fmt tri [--qattn b1x] [--qffn t1e] [--qemb q4_k]
          --fmt tri                 target format for attention+ffn weights (--qattn/--qffn to split)
          --qemb q4_k               also shrink embeddings / lm_head (f16 by default, kept unless asked)
          --dry-run                 print the per-tensor from -> to plan, write nothing
          --from-report report.json --max-rell2 0.35
                                    per-tensor formats from an `analyze` report of the ORIGINAL
                                    checkpoint: smallest bpw whose relL2 fits the gate; never grows
          --exclude tok_emb,lm_head | --only l0.wq,l0.w_gate | --min-elems 4096
          kept tensors are byte-exact; norms/router/KDA-gate/mHC stay fp untouched.
          Requant adds a second quantization error on top of the first — verify with
          logits / short generation before shipping (see README for pros and cons).

        convert: any-to-any between GGUF, .qw and HuggingFace safetensors — the destination kind
          is detected from -o (or the legacy --gguf/--hf source flags still work):
            convert model.gguf -o model.qw         gguf -> qw      (quantize on import: --fmt/--qattn/...)
            convert hf-dir/    -o model.qw         safetensors dir -> qw
            convert model.qw   -o model.gguf       qw -> gguf      (llama.cpp-loadable; --ggml-type auto)
            convert model.gguf -o hf-out/          gguf -> split safetensors + config.json (HF-ready)
            convert model.qw   -o hf-out/          qw -> split safetensors + config.json (dequantized)
          safetensors export options:
            --shards N              split into N shards (default 1 -> model.safetensors)
            --layers-per-shard M    group M transformer layers per shard (embeddings first,
                                    final norm + lm_head last) — overrides --shards
            --dtype f16|f32|bf16    dequantized output precision (default f16)
            index.json + config.json + generation_config.json + tokenizer files are generated;
            a .qw imported from GGUF restores the original tokenizer verbatim, a .qw from
            safetensors exports weights only (note tells you what to copy for training).
          gguf export options:
            --ggml-type auto|f16|f32|q8_0   auto keeps layout-identical formats byte-exact,
                                            re-encodes Q4_0 as true ggml Q4_0, TRI/sub-TRI -> F16
        convert quant options:
          --qattn q4_k     attention weights format   (default q4_k)
          --qffn  q4_k     ffn weights format         (default q4_k)
          --qemb  f16      token embedding format     (default f16)
          --qhead f16      lm_head format             (default f16)
          --min-elems 4096 tensors smaller stay fp
        formats: f32 f16 bf16 q4_0 q8_0 q2_k q3_k q4_k q5_k q6_k i8 tri b1 b1x t1e t1e_lo

        run options:
          -p "text"    prompt (or pipe via stdin)
          -n 256       max new tokens
          --temp 0.8   0 = greedy | --topk 40 | --topp 0.95 | --seed -1
          --repeat-penalty 1.1 | --no-prefetch | --prefetch-mb 1024 | --max-ctx 4096
        """);
    }

    // ------------------------------------------------------------------ convert

    private static int CmdConvert(string[] a)
    {
        string? src = null, output = null;
        string? legacyGguf = null, legacyHf = null;
        var opt = new ConvertOptions();
        var hfOpt = new HfExportOptions();
        var ggufOpt = new GgufExportOptions();
        for (int i = 0; i < a.Length; i++)
        {
            switch (a[i])
            {
                case "--gguf": legacyGguf = a[++i]; break;
                case "--hf": legacyHf = a[++i]; break;
                case "-o" or "--out": output = a[++i]; break;
                case "--qattn": opt.Attention = ConvertOptions.ParseQuant(a[++i]); break;
                case "--qffn": opt.Ffn = ConvertOptions.ParseQuant(a[++i]); break;
                case "--fmt": opt.Attention = opt.Ffn = ConvertOptions.ParseQuant(a[++i]); break;
                case "--qemb": opt.Emb = ConvertOptions.ParseQuant(a[++i]); break;
                case "--qhead": opt.Head = ConvertOptions.ParseQuant(a[++i]); break;
                case "--qnorm": opt.Norm = ConvertOptions.ParseQuant(a[++i]); break;
                case "--min-elems": opt.MinQuantElements = long.Parse(a[++i]); break;
                case "--max-ctx": opt.MaxCtx = int.Parse(a[++i]); break;
                case "--shards": hfOpt.Shards = int.Parse(a[++i]); break;
                case "--layers-per-shard": hfOpt.LayersPerShard = int.Parse(a[++i]); break;
                case "--dtype": hfOpt.DType = a[++i]; break;
                case "--ggml-type": ggufOpt.GgmlType = a[++i]; break;
                default:
                    if (!a[i].StartsWith('-') && src == null) src = a[i];
                    else { Console.Error.WriteLine($"error: unknown option '{a[i]}'"); return 2; }
                    break;
            }
        }
        if (legacyGguf != null) src = legacyGguf;
        if (legacyHf != null) src = legacyHf;
        if (src == null || output == null)
        {
            Console.Error.WriteLine("""
                usage: convert <src> -o <dst> [options]
                  src: model.gguf | model.qw | <hf dir>     dst: model.qw | model.gguf | <out dir>/
                examples:
                  convert model.gguf -o model.qw            (gguf -> qw, quantize: --fmt q4_k ...)
                  convert hf-dir -o model.qw                (safetensors dir -> qw)
                  convert model.qw -o model.gguf            (qw -> gguf, --ggml-type auto)
                  convert model.gguf -o hf-out/             (gguf -> split safetensors + config.json)
                  convert model.qw -o hf-out/ --shards 4    (qw -> split safetensors + config.json)
                """);
            return 2;
        }

        // destination kind: .qw = import, .gguf = gguf export, anything else = HF safetensors dir
        bool dstIsQw = output.EndsWith(".qw", StringComparison.OrdinalIgnoreCase);
        bool dstIsGguf = output.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);
        bool srcIsGguf = src.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);
        bool srcIsQw = src.EndsWith(".qw", StringComparison.OrdinalIgnoreCase);

        if (dstIsQw)
        {
            Console.WriteLine($"QuantRunner convert -> {output}");
            var sw = Stopwatch.StartNew();
            ConvertReport rep = srcIsGguf
                ? ModelConverter.ConvertGguf(src, output, opt)
                : ModelConverter.ConvertHfDir(src, output, opt);
            sw.Stop();

            double ratio = rep.OriginalBytes > 0 ? rep.OriginalBytes / (double)Math.Max(1, rep.TotalBytes) : 0;
            Console.WriteLine($"  tensors      : {rep.TensorCount}");
            Console.WriteLine($"  output size  : {Fmt(rep.TotalBytes)}");
            Console.WriteLine($"  source size  : {Fmt(rep.OriginalBytes)}  ({ratio:F2}x smaller)");
            Console.WriteLine($"  time         : {sw.Elapsed.TotalSeconds:F1}s  ({Fmt((long)(rep.TotalBytes / Math.Max(0.001, sw.Elapsed.TotalSeconds)))}/s)");
            Console.WriteLine($"  arch         : {rep.Config.Arch}, layers={rep.Config.NumLayers}, dim={rep.Config.HiddenSize}" +
                              (rep.Config.IsMoe ? $", experts={rep.Config.NumExperts}x{rep.Config.NumActiveExperts} active" : ""));
            return 0;
        }

        if (dstIsGguf)
        {
            if (!srcIsQw)
            {
                Console.Error.WriteLine("error: gguf export needs a .qw source (convert <model.qw> -o <model.gguf>)");
                return 2;
            }
            GgufExporter.Export(src, output, ggufOpt);
            return 0;
        }

        // HF safetensors directory export
        if (!srcIsQw && !srcIsGguf)
        {
            Console.Error.WriteLine("error: safetensors export needs a .qw or .gguf source");
            return 2;
        }
        if (hfOpt.LayersPerShard > 0 && hfOpt.Shards > 1)
        {
            Console.Error.WriteLine("error: use either --shards or --layers-per-shard, not both");
            return 2;
        }
        if (srcIsGguf) HfExporter.FromGguf(src, output, hfOpt);
        else HfExporter.FromQw(src, output, hfOpt);
        return 0;
    }

    // ------------------------------------------------------------------ synthesize

    private static int CmdSynthesize(string[] a)
    {
        string? output = null;
        int dim = 512, layers = 4, heads = 8, kv = -1, ffn = 1408, vocab = 1024, experts = 0, active = 0;
        string quant = "q4_k";
        int seed = 42, hc = 2;
        bool glm5 = false;

        for (int i = 0; i < a.Length; i++)
        {
            switch (a[i])
            {
                case "-o" or "--out": output = a[++i]; break;
                case "--dim": dim = int.Parse(a[++i]); break;
                case "--layers": layers = int.Parse(a[++i]); break;
                case "--heads": heads = int.Parse(a[++i]); break;
                case "--kv-heads": kv = int.Parse(a[++i]); break;
                case "--ffn": ffn = int.Parse(a[++i]); break;
                case "--vocab": vocab = int.Parse(a[++i]); break;
                case "--experts": experts = int.Parse(a[++i]); break;
                case "--active": active = int.Parse(a[++i]); break;
                case "--quant": quant = a[++i]; break;
                case "--seed": seed = int.Parse(a[++i]); break;
                case "--glm5": glm5 = true; break;          // GLM-5.x hybrid: mHC + KDA:MLA 3:1 (+ GLM-MoE with --experts)
                case "--hc": hc = int.Parse(a[++i]); break;
                default:
                    if (!a[i].StartsWith('-') && output == null) output = a[i];
                    break;
            }
        }
        if (output == null) { Console.Error.WriteLine("error: output path required"); return 2; }
        if (kv < 0) kv = heads;
        var fmt = ConvertOptions.ParseQuant(quant);
        if (active > experts) active = experts;

        // K-quant blocks require 256-alignment of every matrix dimension
        dim = Round(dim, 256); ffn = Round(ffn, 256); vocab = Round(vocab, 256);

        if (glm5)
        {
            Synth.WriteGlm5(output, dim, layers, heads, ffn, vocab, experts, active, hc, seed, fmt);
            return 0;
        }

        int headDim = dim / heads;
        if (dim % heads != 0) { Console.Error.WriteLine("error: dim must be divisible by heads"); return 2; }

        var cfg = new ModelConfig
        {
            Arch = experts > 0 ? "moe" : "llama",
            HiddenSize = dim, NumLayers = layers, NumHeads = heads, NumKvHeads = kv,
            HeadDim = headDim, FfnDim = ffn, VocabSize = vocab,
            NumExperts = experts, NumActiveExperts = active, ExpertFfnDim = experts > 0 ? ffn : -1,
            MaxCtx = 2048, SourceDescription = $"synthetic(seed={seed})"
        };

        Console.WriteLine($"QuantRunner synthesize: {cfg.Arch} dim={dim} layers={layers} ffn={ffn} vocab={vocab} " +
                          (experts > 0 ? $"experts={experts}/{active} " : "") + $"quant={fmt}");

        using var builder = new ModelStoreBuilder(output, Synth.ConfigWithKvs(cfg));
        var rng = new Random(seed);
        var plans = new List<(TensorPlan plan, float scale)>();
        void Plan(string name, long[] dims, QuantFormat f, float scale)
        {
            var p = builder.Plan(name, dims, f);
            plans.Add((p, scale));
        }

        Plan("tok_emb", new long[] { vocab, dim }, QuantFormat.F16, 0.02f);
        Plan("final_norm", new long[] { dim }, QuantFormat.F32, 1f);
        for (int l = 0; l < layers; l++)
        {
            Plan($"l{l}.attn_norm", new long[] { dim }, QuantFormat.F32, 1f);
            Plan($"l{l}.wq", new long[] { dim, dim }, fmt, 0.02f);
            Plan($"l{l}.wk", new long[] { kv * headDim, dim }, fmt, 0.02f);
            Plan($"l{l}.wv", new long[] { kv * headDim, dim }, fmt, 0.02f);
            Plan($"l{l}.wo", new long[] { dim, dim }, fmt, 0.02f);
            Plan($"l{l}.ffn_norm", new long[] { dim }, QuantFormat.F32, 1f);
            if (experts > 0)
            {
                Plan($"l{l}.router", new long[] { experts, dim }, QuantFormat.F32, 1f);
                for (int e = 0; e < experts; e++)
                {
                    Plan($"l{l}.e{e}.w_gate", new long[] { ffn, dim }, fmt, 0.02f);
                    Plan($"l{l}.e{e}.w_up", new long[] { ffn, dim }, fmt, 0.02f);
                    Plan($"l{l}.e{e}.w_down", new long[] { dim, ffn }, fmt, 0.02f);
                }
            }
            else
            {
                Plan($"l{l}.w_gate", new long[] { ffn, dim }, fmt, 0.02f);
                Plan($"l{l}.w_up", new long[] { ffn, dim }, fmt, 0.02f);
                Plan($"l{l}.w_down", new long[] { dim, ffn }, fmt, 0.02f);
            }
        }
        Plan("lm_head", new long[] { vocab, dim }, QuantFormat.F16, 0.02f);

        long total = plans.Sum(p => p.plan.SizeBytes);
        Console.WriteLine($"  tensors: {plans.Count}, total {Fmt(total)} ({total * 8.0 / Math.Max(1, plans.Sum(p => p.plan.NumElements)):F2} bits/weight avg)");

        builder.Commit((plan, stream) =>
        {
            var (_, scale) = plans.First(p => p.plan.Name == plan.Name);
            long n = plan.NumElements;
            const long CHUNK = 1 << 20;
            var buf = new float[Math.Min(CHUNK, n)];
            long done = 0;
            while (done < n)
            {
                int cnt = (int)Math.Min(buf.Length, n - done);
                for (int i = 0; i < cnt; i++) buf[i] = NextGaussian(rng) * scale;
                // keep norms at ~1 so RMSNorm stays sane for synthetic runs
                if (plan.Name.Contains("norm") || plan.Name.EndsWith("router"))
                    for (int i = 0; i < cnt; i++) buf[i] = 1f;
                var bytes = new byte[Dtypes.TensorBytes(cnt, plan.Format)];
                BlockQuantizer.Quantize(plan.Format, buf.AsSpan(0, cnt), bytes.AsSpan(0, bytes.Length));
                stream.Write(bytes, 0, bytes.Length);
                done += cnt;
            }
        });
        Console.WriteLine($"  wrote {output} ({Fmt(new FileInfo(output).Length)})");
        return 0;
    }

    internal static int Round(int v, int align) => (v + align - 1) / align * align;

    private static float NextGaussian(Random r)
    {
        double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
    }

    internal static string Fmt(long bytes) => bytes switch
    {
        >= 1L << 40 => $"{bytes / (double)(1L << 40):F2} TiB",
        >= 1L << 30 => $"{bytes / (double)(1L << 30):F2} GiB",
        >= 1L << 20 => $"{bytes / (double)(1L << 20):F1} MiB",
        >= 1 << 10 => $"{bytes / 1024.0:F1} KiB",
        _ => $"{bytes} B"
    };
}
