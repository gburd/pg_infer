//! Build pipeline — extract model weights into vindex format.

pub mod build;
pub mod build_from_vectors;
pub mod build_helpers;
pub mod callbacks;
pub mod streaming;

pub use build::build_vindex;
pub use build::build_vindex_resume;
pub use build::resolve_gate_dtype;
pub use build_from_vectors::build_vindex_from_vectors;
pub use streaming::build_vindex_streaming;
pub use streaming::build_vindex_resume as build_vindex_streaming_resume;
pub use callbacks::{IndexBuildCallbacks, SilentBuildCallbacks};
