//! Page format structs for the v2 `infer` index with HNSW + SQ8 embeddings.
//!
//! Version 2 extends the original single-metapage format with:
//! - Embedding pages: SQ8-quantized vectors for each indexed row
//! - HNSW pages: serialized navigable small-world graph for O(log N) ANN
//!
//! Layout on disk (block numbers):
//! ```text
//! Block 0:           MetaPageV2 (512 bytes in page content area)
//! Blocks [embed_start..embed_end):  Packed EmbeddingEntry records
//! Blocks [hnsw_start..hnsw_end):    Serialized HNSW neighbor lists
//! ```

use pgrx::pg_sys;

use crate::am_options::INFER_META_MAGIC;

/// Version 2 metapage magic stays the same; the `version` field discriminates.
pub const INFER_META_VERSION_V2: u32 = 2;

/// Version 2 metapage with HNSW + embedding support.
///
/// Total struct size: 512 bytes.  Stored in the page content area of block 0.
/// Fields are ordered to avoid internal padding (u32 fields first, then u16, then u8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct InferMetaPageV2 {
    // --- u32-aligned fields (offset 0) ---
    /// Magic bytes: 0x494E4652 ("INFR").
    pub magic: u32,
    /// Version number: 2 for this format.
    pub version: u32,
    /// NUL-terminated model name (max 247 chars + NUL).
    pub model_name: [u8; 248],
    /// Embedding dimensionality.
    pub embedding_dim: u32,
    /// Total number of indexed embeddings.
    pub num_embeddings: u32,
    /// First block number of the embedding pages range (inclusive).
    pub embed_pages_start: u32,
    /// End block number of embedding pages (exclusive).
    pub embed_pages_end: u32,
    /// First block number of the HNSW pages range (inclusive).
    pub hnsw_pages_start: u32,
    /// End block number of HNSW pages (exclusive).
    pub hnsw_pages_end: u32,
    /// Node ID of the HNSW entry point.
    pub hnsw_entry_point: u32,
    // --- u16-aligned fields (offset 288) ---
    /// Maximum level in the HNSW graph.
    pub hnsw_max_level: u16,
    /// HNSW M parameter (max neighbors per node per layer).
    pub hnsw_m: u16,
    /// HNSW ef_construction parameter.
    pub hnsw_ef_construction: u16,
    // --- u8 fields (offset 294) ---
    /// Quantization format: 0=f32, 1=f16, 2=sq8.
    pub quantization: u8,
    /// Padding to bring total struct size to 512 bytes.
    /// Fields end at offset 291; 512 - 291 = 221 bytes of padding.
    pub _pad: [u8; 221],
}

// Compile-time size assertion.
const _: () = assert!(std::mem::size_of::<InferMetaPageV2>() == 512);

impl InferMetaPageV2 {
    /// Create a new v2 metapage.
    pub fn new(
        model_name: &str,
        embedding_dim: u32,
        hnsw_m: u16,
        hnsw_ef_construction: u16,
    ) -> Result<Self, crate::error::PgInferError> {
        let bytes = model_name.as_bytes();
        if bytes.len() >= 248 {
            return Err(crate::error::PgInferError::Internal(format!(
                "model name too long ({} bytes, max 247)",
                bytes.len()
            )));
        }
        let mut name_buf = [0u8; 248];
        name_buf[..bytes.len()].copy_from_slice(bytes);

        Ok(Self {
            magic: INFER_META_MAGIC,
            version: INFER_META_VERSION_V2,
            model_name: name_buf,
            embedding_dim,
            num_embeddings: 0,
            embed_pages_start: 0,
            embed_pages_end: 0,
            hnsw_pages_start: 0,
            hnsw_pages_end: 0,
            hnsw_entry_point: 0,
            hnsw_max_level: 0,
            hnsw_m,
            hnsw_ef_construction,
            quantization: 2, // SQ8 by default
            _pad: [0u8; 221],
        })
    }

    /// Extract model name from this metapage.
    pub fn model_name_str(&self) -> Result<&str, crate::error::PgInferError> {
        if self.magic != INFER_META_MAGIC {
            return Err(crate::error::PgInferError::Internal(
                "infer index v2: invalid metapage magic".into(),
            ));
        }
        let end = self
            .model_name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(248);
        std::str::from_utf8(&self.model_name[..end])
            .map_err(|e| crate::error::PgInferError::Internal(format!("invalid model name: {e}")))
    }

    /// Check if this is a valid v2 metapage.
    pub fn is_valid_v2(&self) -> bool {
        self.magic == INFER_META_MAGIC && self.version == INFER_META_VERSION_V2
    }
}

/// Single embedding entry stored in index pages.
///
/// Each entry is a fixed-size header followed by `embedding_dim` u8 quantized values.
/// The total size per entry is: `size_of::<EmbeddingEntryHeader>() + embedding_dim`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmbeddingEntryHeader {
    /// Heap tuple identifier (6 bytes).
    pub tid: pg_sys::ItemPointerData,
    /// Padding to 8-byte alignment.
    pub _pad: [u8; 2],
    /// SQ8 scale: minimum value of the original f32 vector.
    pub min: f32,
    /// SQ8 scale: maximum value of the original f32 vector.
    pub max: f32,
}

impl EmbeddingEntryHeader {
    /// Size of the header in bytes (without the quantized data).
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

/// Calculate how many embedding entries fit in one 8KB page (minus page header).
///
/// Returns the number of entries that fit, given the embedding dimension.
pub fn entries_per_page(embedding_dim: u32) -> usize {
    let page_content_size =
        pg_sys::BLCKSZ as usize - std::mem::size_of::<pg_sys::PageHeaderData>();
    let entry_size = EmbeddingEntryHeader::SIZE + embedding_dim as usize;
    if entry_size == 0 {
        return 0;
    }
    page_content_size / entry_size
}

/// Calculate total pages needed for N embeddings of given dimension.
pub fn pages_needed_for_embeddings(num_embeddings: u32, embedding_dim: u32) -> u32 {
    let per_page = entries_per_page(embedding_dim);
    if per_page == 0 {
        return 0;
    }
    ((num_embeddings as usize + per_page - 1) / per_page) as u32
}

/// Read metapage version from block 0 (returns 1 or 2).
///
/// # Safety
///
/// Caller must ensure `index_rel` is a valid, open index relation with at
/// least one block.
pub unsafe fn read_metapage_version(index_rel: pg_sys::Relation) -> Result<u32, crate::error::PgInferError> {
    let buf = pg_sys::ReadBuffer(index_rel, 0);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);

    let page = pg_sys::BufferGetPage(buf);
    let data = pg_sys::PageGetContents(page) as *const InferMetaPageV2;

    let magic = (*data).magic;
    let version = (*data).version;

    pg_sys::UnlockReleaseBuffer(buf);

    if magic != INFER_META_MAGIC {
        return Err(crate::error::PgInferError::Internal(
            "infer index: invalid metapage magic".into(),
        ));
    }
    Ok(version)
}

/// Read the full v2 metapage from block 0.
///
/// # Safety
///
/// Caller must ensure `index_rel` is a valid, open v2 index relation.
pub unsafe fn read_metapage_v2(
    index_rel: pg_sys::Relation,
) -> Result<InferMetaPageV2, crate::error::PgInferError> {
    let buf = pg_sys::ReadBuffer(index_rel, 0);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);

    let page = pg_sys::BufferGetPage(buf);
    let data = pg_sys::PageGetContents(page) as *const InferMetaPageV2;
    let meta = *data;

    pg_sys::UnlockReleaseBuffer(buf);

    if !meta.is_valid_v2() {
        return Err(crate::error::PgInferError::Internal(
            "infer index: not a valid v2 metapage".into(),
        ));
    }
    Ok(meta)
}
