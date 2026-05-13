use std::path::PathBuf;

/// Errors that can occur during model loading, tokenization, or inference.
///
/// # Examples
///
/// ```
/// use infer_inference::InferenceError;
///
/// let err = InferenceError::MissingTensor("lm_head.weight".to_string());
/// assert!(err.to_string().contains("lm_head.weight"));
///
/// let err = InferenceError::Parse("unexpected EOF".to_string());
/// assert!(format!("{err}").contains("parse error"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("not a directory: {0}")]
    NotADirectory(PathBuf),
    #[error("no safetensors files in {0}")]
    NoSafetensors(PathBuf),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("vindex error: {0}")]
    Vindex(#[from] infer_vindex::VindexError),
    #[error("model error: {0}")]
    Model(#[from] infer_models::ModelError),
}
