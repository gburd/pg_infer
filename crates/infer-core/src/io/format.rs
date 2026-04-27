use std::path::Path;

/// Serialization format for graph files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Pretty-printed JSON (.infer.json)
    Json,
    /// MessagePack binary (.infer.bin)
    #[cfg(feature = "msgpack")]
    MessagePack,
    /// Packed binary with string interning (.infer.pak)
    Packed,
}

impl Format {
    /// Detect format from file extension.
    /// Returns None if the extension is unrecognised.
    pub fn from_path(path: impl AsRef<Path>) -> Option<Self> {
        let path = path.as_ref();
        let name = path.file_name()?.to_str()?;

        if name.ends_with(".infer.json") || name.ends_with(".json") {
            return Some(Self::Json);
        }

        #[cfg(feature = "msgpack")]
        if name.ends_with(".infer.bin") || name.ends_with(".bin") || name.ends_with(".msgpack") {
            return Some(Self::MessagePack);
        }

        if name.ends_with(".infer.pak") || name.ends_with(".pak") {
            return Some(Self::Packed);
        }

        None
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::Json => ".infer.json",
            #[cfg(feature = "msgpack")]
            Self::MessagePack => ".infer.bin",
            Self::Packed => ".infer.pak",
        }
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json => write!(f, "json"),
            #[cfg(feature = "msgpack")]
            Self::MessagePack => write!(f, "msgpack"),
            Self::Packed => write!(f, "packed"),
        }
    }
}
