using System.Text.Json;
using System.Text.Json.Serialization;

namespace QuantRunner.Core.Models;

/// <summary>
/// Architecture config embedded in the .qw header. Covers Llama-family dense models and
/// Mixtral/DeepSeek-style MoE. New architectures can be added via ILayerFactory without
/// touching the core (pluggable generic support).
/// </summary>
public sealed class ModelConfig
{
    [JsonPropertyName("arch")] public string Arch { get; set; } = "llama";   // "llama" | "moe" | custom
    [JsonPropertyName("dim")] public int HiddenSize { get; set; }
    [JsonPropertyName("layers")] public int NumLayers { get; set; }
    [JsonPropertyName("heads")] public int NumHeads { get; set; }
    [JsonPropertyName("kv_heads")] public int NumKvHeads { get; set; } = -1; // -1 = same as heads (MHA)
    [JsonPropertyName("head_dim")] public int HeadDim { get; set; } = -1;    // -1 = HiddenSize / NumHeads
    [JsonPropertyName("ffn")] public long FfnDim { get; set; }
    [JsonPropertyName("vocab")] public int VocabSize { get; set; }
    [JsonPropertyName("rope_theta")] public float RopeTheta { get; set; } = 10000f;
    [JsonPropertyName("rms_eps")] public float RmsEps { get; set; } = 1e-5f;
    [JsonPropertyName("n_experts")] public int NumExperts { get; set; } = 0;
    [JsonPropertyName("n_active")] public int NumActiveExperts { get; set; } = 0;
    [JsonPropertyName("expert_ffn")] public long ExpertFfnDim { get; set; } = -1;
    [JsonPropertyName("tie_emb")] public bool TieEmbeddings { get; set; }
    [JsonPropertyName("max_ctx")] public int MaxCtx { get; set; } = 4096;
    [JsonPropertyName("source")] public string SourceDescription { get; set; } = "";
    [JsonPropertyName("extra")] public Dictionary<string, JsonElement>? Extra { get; set; }

    // ---- GLM-5.x family: per-layer attention kinds + hyper-connections + MLA/KDA dims ----

    /// <summary>Attention kind per layer: "gqa" | "kda" (Kimi delta linear attention) | "mla"
    /// (multi-latent attention / DSA without indexer). Empty => every layer is legacy GQA.</summary>
    [JsonPropertyName("attn_kinds")] public string[] AttnKinds { get; set; } = Array.Empty<string>();

    /// <summary>Hyper-connections (mHC): number of residual streams; 0/1 = classic residual.</summary>
    [JsonPropertyName("hc_mult")] public int HcMult { get; set; }
    [JsonPropertyName("hc_eps")] public float HcEps { get; set; } = 1e-6f;
    [JsonPropertyName("hc_sinkhorn")] public int HcSinkhornIters { get; set; } = 1;

    // MLA (DeepSeek / GLM-5.3 multi-latent attention)
    [JsonPropertyName("q_lora")] public int QLoraRank { get; set; }
    [JsonPropertyName("kv_lora")] public int KvLoraRank { get; set; }
    [JsonPropertyName("qk_nope")] public int QkNopeHeadDim { get; set; }
    [JsonPropertyName("qk_rope")] public int QkRopeHeadDim { get; set; }
    [JsonPropertyName("v_head_dim")] public int VHeadDim { get; set; }

    // KDA (Kimi delta attention, GLM-5.3 linear_attn_config)
    [JsonPropertyName("lin_heads")] public int LinearHeads { get; set; }
    [JsonPropertyName("lin_head_dim")] public int LinearHeadDim { get; set; }
    [JsonPropertyName("lin_conv")] public int LinearConvKernel { get; set; } = 4;
    /// <summary>Gate lower bound: negative = bounded sigmoid decay (GLM-5.3: -5); 0 = softplus decay path (Kimi Linear).</summary>
    [JsonPropertyName("lin_lower_bound")] public float LinearLowerBound { get; set; }

    // GLM-style MoE (sigmoid router, noaux_tc bias, routed scaling, shared expert, limited SwiGLU)
    [JsonPropertyName("moe_sigmoid")] public bool MoeSigmoid { get; set; }
    [JsonPropertyName("routed_scale")] public float RoutedScale { get; set; } = 1f;
    [JsonPropertyName("n_shared")] public int NumSharedExperts { get; set; }
    /// <summary>SwiGLU clamp limit (GLM-5.x: 10). 0 = plain silu(g)*u.</summary>
    [JsonPropertyName("swiglu_limit")] public float SwigluLimit { get; set; }

    [JsonIgnore] public int EffectiveHeadDim => HeadDim > 0 ? HeadDim : HiddenSize / NumHeads;
    [JsonIgnore] public int EffectiveKvHeads => NumKvHeads < 0 ? NumHeads : NumKvHeads;
    [JsonIgnore] public long EffectiveExpertFfn => ExpertFfnDim > 0 ? ExpertFfnDim : FfnDim;
    [JsonIgnore] public bool IsMoe => Arch == "moe" || (NumExperts > 0 && NumActiveExperts > 0);

    /// <summary>True when any layer uses the GLM-5.x forward (mHC streams and/or KDA/MLA attention).</summary>
    [JsonIgnore] public bool IsGlm5 => AttnKinds.Length > 0 || HcMult > 1;
    [JsonIgnore] public bool HasMhc => HcMult > 1;
    [JsonIgnore] public bool HasKda => Array.IndexOf(AttnKinds, "kda") >= 0;
    [JsonIgnore] public bool HasMla => Array.IndexOf(AttnKinds, "mla") >= 0;

    public string AttnKind(int layer) =>
        AttnKinds.Length == 0 ? "gqa" : AttnKinds[Math.Min(layer, AttnKinds.Length - 1)];

    public string ToJson() => JsonSerializer.Serialize(this,
        new JsonSerializerOptions { WriteIndented = true, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull });

    public static ModelConfig FromJson(JsonElement el) => JsonSerializer.Deserialize<ModelConfig>(el.GetRawText())
        ?? throw new InvalidDataException("bad model config");
}
