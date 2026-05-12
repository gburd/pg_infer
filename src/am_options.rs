//! Reloptions parsing for the `infer` access method.
//!
//! Supports `CREATE INDEX ... USING infer (col) WITH (model = 'name')`.
//! If no model is specified, falls back to `infer.default_model` GUC at
//! index creation time (baked into the metapage).

use pgrx::pg_sys;

use crate::error::PgInferError;

/// Magic bytes for the infer metapage: "INFR" = 0x494E4652.
pub const INFER_META_MAGIC: u32 = 0x494E_4652;

/// Metapage version.
pub const INFER_META_VERSION: u32 = 1;

/// Maximum length of a model name stored in the metapage.
pub const MODEL_NAME_MAX_LEN: usize = 248;

/// On-disk metapage format stored as block 0 of the index.
///
/// Total size: 256 bytes, fits within any PG page.
#[repr(C)]
pub struct InferMetaPage {
    pub magic: u32,
    pub version: u32,
    pub model_name: [u8; MODEL_NAME_MAX_LEN],
}

impl InferMetaPage {
    /// Create a new metapage with the given model name.
    pub fn new(model_name: &str) -> Result<Self, PgInferError> {
        let bytes = model_name.as_bytes();
        if bytes.len() >= MODEL_NAME_MAX_LEN {
            return Err(PgInferError::Internal(format!(
                "model name too long ({} bytes, max {})",
                bytes.len(),
                MODEL_NAME_MAX_LEN - 1
            )));
        }
        let mut name_buf = [0u8; MODEL_NAME_MAX_LEN];
        name_buf[..bytes.len()].copy_from_slice(bytes);
        Ok(Self {
            magic: INFER_META_MAGIC,
            version: INFER_META_VERSION,
            model_name: name_buf,
        })
    }

    /// Extract model name from a metapage buffer.
    pub fn model_name_str(&self) -> Result<&str, PgInferError> {
        if self.magic != INFER_META_MAGIC {
            return Err(PgInferError::Internal(
                "infer index: invalid metapage magic".into(),
            ));
        }
        let end = self
            .model_name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(MODEL_NAME_MAX_LEN);
        std::str::from_utf8(&self.model_name[..end])
            .map_err(|e| PgInferError::Internal(format!("infer index: invalid model name: {e}")))
    }
}

/// Read the model name from an index's metapage (block 0).
///
/// # Safety
///
/// Caller must ensure `index_rel` is a valid, open index relation.
pub unsafe fn read_model_from_metapage(index_rel: pg_sys::Relation) -> Result<String, PgInferError> {
    let buf = pg_sys::ReadBuffer(index_rel, 0);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);

    let page = pg_sys::BufferGetPage(buf);
    let data = pg_sys::PageGetContents(page) as *const InferMetaPage;
    let meta = &*data;
    let name = meta.model_name_str()?.to_string();

    pg_sys::UnlockReleaseBuffer(buf);
    Ok(name)
}

/// Parse the `WITH (model = '...')` reloptions from a CREATE INDEX.
///
/// Returns the model name if specified, otherwise `None` (caller should
/// fall back to GUC).
///
/// PostgreSQL stores reloptions as a bytea varlena.  Since pgrx doesn't
/// expose a high-level reloptions API for custom text options, we scan the
/// raw bytes for the `model=` pattern.
pub fn parse_model_from_reloptions(reloptions: pg_sys::Datum) -> Option<String> {
    if reloptions.is_null() {
        return None;
    }

    unsafe {
        let ptr = reloptions.cast_mut_ptr::<pg_sys::varlena>();
        if ptr.is_null() {
            return None;
        }
        // Decode varlena header.  Standard 4-byte header: first byte's top
        // bit distinguishes 1-byte and 4-byte headers.
        let header_ptr = ptr as *const u8;
        let first_byte = *header_ptr;
        let (data_ptr, data_len): (*const u8, usize) = if first_byte & 0x01 != 0 {
            // 1-byte header (short varlena): length in top 7 bits,
            // includes the 1-byte header itself.
            let total = (first_byte >> 1) as usize;
            if total <= 1 {
                return None;
            }
            (header_ptr.add(1), total - 1)
        } else {
            // 4-byte header: full 32-bit word, length includes header.
            let word = *(ptr as *const u32);
            let total = (word >> 2) as usize;
            if total <= 4 {
                return None;
            }
            (header_ptr.add(4), total - 4)
        };
        if data_len == 0 {
            return None;
        }
        let slice = std::slice::from_raw_parts(data_ptr, data_len);
        let text = std::str::from_utf8(slice).ok()?;
        // Look for model= in the options text
        for part in text.split(',') {
            let trimmed = part.trim();
            if let Some(value) = trimmed
                .strip_prefix("model=")
                .or_else(|| trimmed.strip_prefix("model ="))
            {
                let model = value.trim().trim_matches('\'').trim_matches('"');
                if !model.is_empty() {
                    return Some(model.to_string());
                }
            }
        }
        None
    }
}
