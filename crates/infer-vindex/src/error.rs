use std::path::PathBuf;

use crate::config::ExtractLevel;

#[derive(Debug, thiserror::Error)]
pub enum VindexError {
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
    /// The container is a generation this build cannot read.
    ///
    /// pg_infer reads VINDEX2 (`index.json` schema 1–2). VINDEX3 is a
    /// different container generation — schema 3+, its own index shape,
    /// LYRW v2 physical layout, a system graph — not a newer revision of
    /// the same format. Refusing by version is the point: a V3 index
    /// deserialized against the V2 struct loses every field it does not
    /// share and would then be read as a malformed V2.
    #[error(
        "vindex format version {found} is not supported (this build reads \
         {min}..={max}). Version 3+ is the VINDEX3 container generation, \
         which has a different on-disk layout; rebuild the vindex with a \
         VINDEX2 extract, or use a newer pg_infer."
    )]
    UnsupportedFormatVersion { found: u32, min: u32, max: u32 },
    /// The container declares a quantization layout whose *interpretation*
    /// this build does not implement.
    ///
    /// Not a missing feature so much as a refusal to guess: `fp4` and
    /// `bitnet_layout` carry block geometry, scale dtypes and per-projection
    /// precision that determine how the weight bytes decode. Upstream's own
    /// spec says readers must dispatch on the declared tag and must not
    /// sniff filenames. Ignoring the field (which is what happens without
    /// this check, since nothing here sets `deny_unknown_fields`) means
    /// decoding those bytes under the wrong geometry and returning
    /// confident nonsense.
    #[error(
        "vindex declares `{field}`, whose weight layout this build cannot \
         decode. Refusing rather than misreading the weights: the field \
         carries block geometry and scale dtypes that change how bytes are \
         interpreted. Use a vindex without {field}, or a pg_infer that \
         implements it."
    )]
    UnsupportedQuantLayout { field: &'static str },
    #[error("requires extract level '{needed}' but vindex was built at '{have}'")]
    InsufficientExtractLevel {
        needed: ExtractLevel,
        have: ExtractLevel,
    },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("model error: {0}")]
    Model(#[from] infer_models::ModelError),
}
