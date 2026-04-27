use pgrx::error;

/// Unified error type for pg_infer operations.
#[derive(Debug, thiserror::Error)]
pub enum PgInferError {
    #[error("INFER vindex error: {0}")]
    Vindex(#[from] infer_vindex::VindexError),

    #[error("INFER model error: {0}")]
    Model(#[from] infer_models::ModelError),

    #[error("model not found: {name}")]
    ModelNotFound { name: String },

    #[error("no default model configured — SET infer.default_model first")]
    NoDefaultModel,

    #[cfg_attr(
        not(feature = "inference"),
        allow(dead_code)
    )]
    #[error("model requires extract level '{needed}' but was built at '{have}'")]
    InsufficientExtractLevel { needed: String, have: String },

    #[error("tokenization failed: {0}")]
    Tokenize(String),

    #[error("empty prompt")]
    EmptyPrompt,

    #[error("path not permitted: {path} is not under allowed directory {allowed}")]
    PathNotPermitted { path: String, allowed: String },

    #[error("{0}")]
    Internal(String),

    #[error("SPI error: {0}")]
    Spi(#[from] pgrx::spi::SpiError),
}

/// Convert a `PgInferError` into a PostgreSQL `ereport(ERROR, ...)` and
/// diverge.  Use this at the boundary where you want to surface the error
/// to the SQL caller.
#[allow(dead_code)]
pub fn report(e: PgInferError) -> ! {
    match e {
        PgInferError::ModelNotFound { ref name } => {
            error!("INFER: model '{}' not found — register it with infer_create_model()", name);
        }
        PgInferError::NoDefaultModel => {
            error!(
                "INFER: no default model configured. \
                 Use SET infer.default_model = 'name'; or pass model => 'name'"
            );
        }
        PgInferError::InsufficientExtractLevel {
            ref needed,
            ref have,
        } => {
            error!(
                "INFER: operation requires extract level '{}' but model was built at '{}'",
                needed, have
            );
        }
        PgInferError::EmptyPrompt => {
            error!("INFER: prompt is empty after tokenization");
        }
        PgInferError::PathNotPermitted { ref path, ref allowed } => {
            error!(
                "INFER: path '{}' is not under allowed directory '{}'",
                path, allowed
            );
        }
        _ => {
            error!("INFER: {}", e);
        }
    }
}
