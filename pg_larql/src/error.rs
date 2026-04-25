use pgrx::error;

/// Unified error type for pg_larql operations.
#[derive(Debug, thiserror::Error)]
pub enum PgLarqlError {
    #[error("LARQL vindex error: {0}")]
    Vindex(#[from] larql_vindex::VindexError),

    #[error("LARQL model error: {0}")]
    Model(#[from] larql_models::ModelError),

    #[error("model not found: {name}")]
    ModelNotFound { name: String },

    #[error("no default model configured — SET larql.default_model first")]
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

    #[error("{0}")]
    Internal(String),

    #[error("SPI error: {0}")]
    Spi(#[from] pgrx::spi::SpiError),
}

/// Convert a `PgLarqlError` into a PostgreSQL `ereport(ERROR, ...)` and
/// diverge.  Use this at the boundary where you want to surface the error
/// to the SQL caller.
#[allow(dead_code)]
pub fn report(e: PgLarqlError) -> ! {
    match e {
        PgLarqlError::ModelNotFound { ref name } => {
            error!("LARQL: model '{}' not found — register it with larql_create_model()", name);
        }
        PgLarqlError::NoDefaultModel => {
            error!(
                "LARQL: no default model configured. \
                 Use SET larql.default_model = 'name'; or pass model => 'name'"
            );
        }
        PgLarqlError::InsufficientExtractLevel {
            ref needed,
            ref have,
        } => {
            error!(
                "LARQL: operation requires extract level '{}' but model was built at '{}'",
                needed, have
            );
        }
        PgLarqlError::EmptyPrompt => {
            error!("LARQL: prompt is empty after tokenization");
        }
        _ => {
            error!("LARQL: {}", e);
        }
    }
}
