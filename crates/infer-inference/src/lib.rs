// rust 1.98 prefers `as_chunks::<N>()` over `chunks_exact(N)`. These are
// dequantisation loops where `chunks_exact(4)` reads as "4 bytes per f32" and
// the turbofished const-generic form does not; the generated code is
// equivalent. Upstream larql made the same call for the same lint.
#![allow(clippy::chunks_exact_to_as_chunks)]
// `ModelWeightsVariant`'s F32 and Quantized arms differ in size by design:
// one holds expanded f32 tensors, the other packed quantised blocks. Boxing
// the larger arm would add an indirection on the hot decode path to satisfy a
// lint about an enum constructed once per model load.
#![allow(clippy::large_enum_variant)]

extern crate blas_src;

pub mod attention;
pub mod capture;
pub mod engines;
pub mod error;
pub mod experts;
pub mod ffn;
pub mod forward;
pub mod graph_ffn;
pub mod layer_graph;
pub mod model;
pub mod prompt;
pub mod residual;
pub mod tokenizer;
pub mod trace;
pub mod trie;
pub mod vindex;
pub mod walker;

// Re-export dependencies for downstream crates.
pub use infer_models;
pub use infer_vindex;
pub use ndarray;
pub use safetensors;
pub use tokenizers;

// Backend re-exports (from infer-compute).
pub use infer_compute::cpu::ops::moe::{
    cpu_moe_forward, run_single_expert, run_single_expert_with_norm,
};
pub use infer_compute::Activation as ComputeActivation;
pub use infer_compute::CpuBackend;
pub use infer_compute::MoeLayerWeights;
pub use infer_compute::{
    cpu_backend, default_backend, dot_proj_gpu, matmul_gpu, ComputeBackend, MatMulOp,
};

/// Map a model's activation function to the compute-layer `Activation` enum.
pub fn activation_from_arch(
    arch: &dyn infer_models::ModelArchitecture,
) -> infer_compute::Activation {
    match arch.activation() {
        infer_models::Activation::GeluTanh => infer_compute::Activation::GeluTanh,
        _ => infer_compute::Activation::Silu,
    }
}
#[cfg(feature = "metal")]
pub use infer_compute::MetalBackend;

// Re-export essentials at crate root.
pub use attention::AttentionWeights;
pub use capture::{
    CaptureCallbacks, CaptureConfig, InferenceModel, TopKEntry, VectorFileHeader, VectorRecord,
};
pub use engines::{MarkovResidualEngine, RsPrefillResult, RsStore};
pub use error::InferenceError;
pub use ffn::{
    FfnBackend, LayerFfnRouter, MoeRouterWeights, RemoteFfnConfig, RemoteFfnError,
    RemoteLatencyStats, RemoteMoeBackend, RemoteMoeError, RemoteWalkBackend, ShardConfig,
    SparseFfn, WeightFfn,
};
pub use forward::{
    apply_knn_override, calibrate_scalar_gains, capture_decoy_residuals,
    capture_ffn_activation_matrix, capture_residuals, capture_spec_residuals,
    estimate_ffn_covariance, forward_raw_logits, forward_to_layer, generate_cached_constrained,
    generate_with_markov_rs, hidden_to_raw_logits, infer_patched, infer_patched_q4k,
    logit_lens_top1, predict, predict_from_hidden, predict_from_hidden_with_ffn, predict_with_ffn,
    predict_with_ffn_attention, predict_with_ffn_trace, predict_with_router, predict_with_strategy,
    run_memit, run_memit_with_target_opt, trace_forward, trace_forward_full,
    trace_forward_with_ffn, walk_trace_from_residuals, InferPatchedResult, KnnOverride,
    LayerAttentionCapture, LayerMode, MemitFact, MemitFactResult, MemitResult, PredictResult,
    PredictResultWithAttention, PredictResultWithResiduals, RawForward, SpecCapture, TargetDelta,
    TargetDeltaOpts, TraceResult, KNN_COSINE_THRESHOLD,
};
pub use graph_ffn::{GateIndex, IndexBuildCallbacks, SilentIndexCallbacks};
pub use layer_graph::{
    build_adaptive_graph,
    detect_template,
    generate,
    // Expert grid generation
    grid::{generate_with_remote_moe, GridGenerateResult},
    hybrid::predict_hybrid,
    predict_honest,
    predict_pipeline,
    predict_split_cached,
    predict_split_pass,
    predict_with_graph,
    predict_with_graph_vindex_logits,
    trace_with_graph,
    AttentionCache,
    CachedLayerGraph,
    DenseLayerGraph,
    GenerateResult,
    GuidedWalkLayerGraph,
    // Production
    LayerGraph,
    LayerOutput,
    PerLayerGraph,
    PipelinedLayerGraph,
    // Analysis/validation
    TemplatePattern,
    TemplateUniverse,
    WalkLayerGraph,
};
pub use model::{load_model_dir, resolve_model_path, ModelWeights};
pub use tokenizer::{decode_token, decode_token_raw, encode_prompt, load_tokenizer};
pub use trace::{
    trace as trace_decomposed, trace_residuals, AnswerWaypoint, BoundaryStore, BoundaryWriter,
    ContextStore, ContextTier, ContextWriter, LayerSummary, ResidualTrace, TraceNode,
    TracePositions, TraceStore, TraceWriter,
};
pub use vindex::{predict_q4k, FfnL1Cache, WalkFfn, WalkFfnConfig};

// Walker re-exports.
pub use walker::attention_walker::{AttentionLayerResult, AttentionWalker};
pub use walker::vector_extractor::{
    ExtractCallbacks, ExtractConfig, ExtractSummary, VectorExtractor,
};
pub use walker::weight_walker::{
    walk_model, LayerResult, LayerStats, WalkCallbacks, WalkConfig, WeightWalker,
};
