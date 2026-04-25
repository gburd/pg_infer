//! PostgreSQL page format for VINDEX data.
//!
//! All vindex data is stored in standard 8KB PostgreSQL pages, managed by the
//! buffer manager and WAL-logged via GenericXLog.  Each page carries a 16-byte
//! `InferPageOpaque` in the special area that identifies the page type.

use pgrx::pg_sys;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic number for the metapage: "INFR" in ASCII.
pub const INFER_MAGIC: u32 = 0x494E4652;

/// Current on-disk format version.
pub const INFER_FORMAT_VERSION: u32 = 1;

/// Size of the special area at the end of each page.
pub const INFER_OPAQUE_SIZE: usize = std::mem::size_of::<InferPageOpaque>();

/// Usable data bytes per page: BLCKSZ - PageHeaderData(24) - opaque(16).
pub const INFER_USABLE_PER_PAGE: usize = pg_sys::BLCKSZ as usize - 24 - INFER_OPAQUE_SIZE;

/// Invalid block number sentinel (same as PG's InvalidBlockNumber / P_NEW).
pub const INVALID_BLOCK_NUMBER: u32 = 0xFFFFFFFF;

// ---------------------------------------------------------------------------
// Page type discriminator
// ---------------------------------------------------------------------------

/// Page types stored in `InferPageOpaque::page_type`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageType {
    Meta = 1,
    LayerDir = 2,
    Gate = 3,
    Embed = 4,
    DownMeta = 5,
    Blob = 6,
}

impl TryFrom<u8> for PageType {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            1 => Ok(Self::Meta),
            2 => Ok(Self::LayerDir),
            3 => Ok(Self::Gate),
            4 => Ok(Self::Embed),
            5 => Ok(Self::DownMeta),
            6 => Ok(Self::Blob),
            other => Err(other),
        }
    }
}

// ---------------------------------------------------------------------------
// InferPageOpaque — 16 bytes, stored at pd_special
// ---------------------------------------------------------------------------

/// Special area at the end of every infer index page.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct InferPageOpaque {
    /// Discriminator: see [`PageType`].
    pub page_type: u8,
    /// Flags: bit 0 = compressed, bit 1 = quantized, others reserved.
    pub flags: u8,
    /// Layer index (0xFFFF for non-layer pages).
    pub layer_id: u16,
    /// Next block in a chain (for blob pages).  `INVALID_BLOCK_NUMBER` if none.
    pub next_blkno: u32,
    /// Reserved for future use.
    pub reserved: [u8; 8],
}

const _: () = assert!(std::mem::size_of::<InferPageOpaque>() == 16);

impl InferPageOpaque {
    pub fn new(page_type: PageType) -> Self {
        Self {
            page_type: page_type as u8,
            flags: 0,
            layer_id: 0xFFFF,
            next_blkno: INVALID_BLOCK_NUMBER,
            reserved: [0u8; 8],
        }
    }
}

// ---------------------------------------------------------------------------
// Metapage (block 0) — stored in the data area of a META page
// ---------------------------------------------------------------------------

/// Metapage payload.  Stored starting at the data area of block 0.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct InferMetaPage {
    /// Magic number: [`INFER_MAGIC`].
    pub magic: u32,
    /// On-disk format version: [`INFER_FORMAT_VERSION`].
    pub format_version: u32,
    /// Null-terminated model name.
    pub model_name: [u8; 128],
    /// Number of transformer layers.
    pub num_layers: u32,
    /// Hidden dimension of the model.
    pub hidden_size: u32,
    /// Number of SAE features per layer.
    pub features_per_layer: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Embedding scale factor.
    pub embed_scale: f32,
    /// Gate vector data type: 0=f32, 1=f16, 2=q4k.
    pub gate_dtype: u8,
    /// Embedding data type: 0=f32, 1=f16.
    pub embed_dtype: u8,
    /// Number of top-K entries stored per feature in down_meta.
    pub down_top_k: u16,
    /// Extract level: 0=browse, 1=inference, 2=all.
    pub extract_level: u8,
    /// Padding for alignment.
    pub _pad: [u8; 3],
    // Section block ranges
    pub layer_dir_blk: u32,
    pub gate_start_blk: u32,
    pub gate_end_blk: u32,
    pub embed_start_blk: u32,
    pub embed_end_blk: u32,
    pub down_start_blk: u32,
    pub down_end_blk: u32,
    pub tok_start_blk: u32,
    pub tok_end_blk: u32,
    /// Maximum gate score observed across all features (for adaptive thresholding).
    pub max_gate_score: f32,
    /// Mean gate score across all features.
    pub mean_gate_score: f32,
    /// Total number of pages in the index.
    pub total_pages: u32,
    /// Source path/URI used during build (null-terminated).
    pub source_uri: [u8; 256],
}

const _: () = assert!(std::mem::size_of::<InferMetaPage>() <= INFER_USABLE_PER_PAGE);

// ---------------------------------------------------------------------------
// Layer directory entry — 20 bytes per layer
// ---------------------------------------------------------------------------

/// One entry in the layer directory page (block 1).
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct LayerDirEntry {
    /// Layer index.
    pub layer_id: u16,
    /// Number of SAE features in this layer.
    pub num_features: u16,
    /// First block of gate vector pages for this layer.
    pub gate_start_blk: u32,
    /// Number of gate pages for this layer.
    pub gate_page_count: u16,
    /// First block of down_meta pages for this layer.
    pub down_start_blk: u32,
    /// Number of down_meta pages for this layer.
    pub down_page_count: u16,
    /// Reserved.
    pub reserved: [u8; 4],
}

const _: () = assert!(std::mem::size_of::<LayerDirEntry>() == 20);

/// Maximum layers that fit in one directory page.
#[allow(dead_code)]
pub const MAX_LAYERS_PER_DIR_PAGE: usize = INFER_USABLE_PER_PAGE / std::mem::size_of::<LayerDirEntry>();

// ---------------------------------------------------------------------------
// Gate page header — 8 bytes at start of data area
// ---------------------------------------------------------------------------

/// Small header at the start of a gate vector page's data area.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GatePageHeader {
    /// Layer this page belongs to.
    pub layer_id: u16,
    /// Number of vectors stored in this page (≤ vectors_per_gate_page()).
    pub num_vectors: u16,
    /// Index of the first feature in this page (within the layer).
    pub first_feature_idx: u32,
}

const _: () = assert!(std::mem::size_of::<GatePageHeader>() == 8);

/// Bytes available for vectors after the GatePageHeader.
pub const GATE_DATA_BYTES: usize = INFER_USABLE_PER_PAGE - std::mem::size_of::<GatePageHeader>();

/// Number of f16 gate vectors that fit in one page, given `hidden_size`.
pub fn vectors_per_gate_page(hidden_size: usize) -> usize {
    let bytes_per_vec = hidden_size * 2; // f16 = 2 bytes
    if bytes_per_vec == 0 {
        return 0;
    }
    GATE_DATA_BYTES / bytes_per_vec
}

/// Number of gate pages needed for `num_features` vectors.
pub fn gate_pages_needed(num_features: usize, hidden_size: usize) -> usize {
    let per_page = vectors_per_gate_page(hidden_size);
    if per_page == 0 {
        return 0;
    }
    (num_features + per_page - 1) / per_page
}

// ---------------------------------------------------------------------------
// Embedding page header — same layout as gate pages
// ---------------------------------------------------------------------------

/// Embedding pages reuse the same header layout as gate pages.
pub type EmbedPageHeader = GatePageHeader;

/// Bytes available for embeddings after the header.
#[allow(dead_code)]
pub const EMBED_DATA_BYTES: usize = GATE_DATA_BYTES;

/// Number of f16 embedding vectors per page.
pub fn embeds_per_page(hidden_size: usize) -> usize {
    vectors_per_gate_page(hidden_size) // same size
}

/// Number of embedding pages needed for `vocab_size` tokens.
pub fn embed_pages_needed(vocab_size: usize, hidden_size: usize) -> usize {
    let per_page = embeds_per_page(hidden_size);
    if per_page == 0 {
        return 0;
    }
    (vocab_size + per_page - 1) / per_page
}

// ---------------------------------------------------------------------------
// Down meta record — 88 bytes per feature
// ---------------------------------------------------------------------------

/// Per-feature metadata: top token and top-K output token logits.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DownMetaRecord {
    /// Top output token ID for this feature.
    pub top_token_id: u32,
    /// Activation score for the top token.
    pub c_score: f32,
    /// Top-K output tokens (token_id, logit) pairs.  Fixed at 10.
    pub top_k: [(u32, f32); 10],
}

const _: () = assert!(std::mem::size_of::<DownMetaRecord>() == 88);

/// Small header at the start of a down_meta page's data area.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DownMetaPageHeader {
    /// Layer this page belongs to.
    pub layer_id: u16,
    /// Number of records in this page.
    pub num_records: u16,
    /// Index of the first feature in this page.
    pub first_feature_idx: u32,
}

const _: () = assert!(std::mem::size_of::<DownMetaPageHeader>() == 8);

/// Bytes available for down_meta records after the header.
#[allow(dead_code)]
pub const DOWN_META_DATA_BYTES: usize =
    INFER_USABLE_PER_PAGE - std::mem::size_of::<DownMetaPageHeader>();

/// Number of DownMetaRecords per page.
#[allow(dead_code)]
pub const DOWN_META_RECORDS_PER_PAGE: usize =
    DOWN_META_DATA_BYTES / std::mem::size_of::<DownMetaRecord>();

/// Number of down_meta pages needed for `num_features` records.
#[allow(dead_code)]
pub fn down_meta_pages_needed(num_features: usize) -> usize {
    (num_features + DOWN_META_RECORDS_PER_PAGE - 1) / DOWN_META_RECORDS_PER_PAGE
}

// ---------------------------------------------------------------------------
// Blob page (tokenizer, etc.) — 8128 bytes per chunk, chained
// ---------------------------------------------------------------------------

/// Header for blob pages (tokenizer JSON, etc.).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlobPageHeader {
    /// Total size of the blob across all pages.
    pub total_size: u32,
    /// Number of data bytes in THIS page.
    pub page_bytes: u32,
}

const _: () = assert!(std::mem::size_of::<BlobPageHeader>() == 8);

/// Bytes available for blob data per page after the header.
pub const BLOB_DATA_BYTES: usize = INFER_USABLE_PER_PAGE - std::mem::size_of::<BlobPageHeader>();

/// Number of blob pages needed for `total_bytes`.
pub fn blob_pages_needed(total_bytes: usize) -> usize {
    if total_bytes == 0 {
        return 0;
    }
    (total_bytes + BLOB_DATA_BYTES - 1) / BLOB_DATA_BYTES
}

// ---------------------------------------------------------------------------
// Page read/write helpers (unsafe, operate on raw page pointers)
// ---------------------------------------------------------------------------

/// Get a mutable pointer to the `InferPageOpaque` in a page's special area.
///
/// # Safety
///
/// `page` must be a valid PostgreSQL page that was initialized with
/// `INFER_OPAQUE_SIZE` as the special size.
pub unsafe fn page_get_opaque(page: pg_sys::Page) -> *mut InferPageOpaque {
    let special = pg_sys::PageGetSpecialPointer(page);
    special as *mut InferPageOpaque
}

/// Get a pointer to the data area right after PageHeader.
///
/// # Safety
///
/// `page` must be a valid, initialized PostgreSQL page.
pub unsafe fn page_get_data(page: pg_sys::Page) -> *mut u8 {
    pg_sys::PageGetContents(page) as *mut u8
}

/// Initialize a new page with our opaque area and set the page type.
///
/// # Safety
///
/// `page` must be a valid, writable buffer page. Caller must be inside
/// a GenericXLog context or otherwise holding appropriate locks.
pub unsafe fn init_page(page: pg_sys::Page, page_type: PageType) {
    pg_sys::PageInit(page, pg_sys::BLCKSZ as usize, INFER_OPAQUE_SIZE);
    let opaque = page_get_opaque(page);
    (*opaque) = InferPageOpaque::new(page_type);
}

/// Write a `T` struct at the given byte offset within the data area.
///
/// # Safety
///
/// Caller must ensure `offset + size_of::<T>() <= INFER_USABLE_PER_PAGE`.
pub unsafe fn write_struct_at<T: Copy>(page: pg_sys::Page, offset: usize, value: &T) {
    let dst = page_get_data(page).add(offset);
    std::ptr::copy_nonoverlapping(value as *const T as *const u8, dst, std::mem::size_of::<T>());
}

/// Read a `T` struct from the given byte offset within the data area.
///
/// # Safety
///
/// Caller must ensure `offset + size_of::<T>() <= INFER_USABLE_PER_PAGE`
/// and that the data at that offset is a valid `T`.
#[allow(dead_code)]
pub unsafe fn read_struct_at<T: Copy>(page: pg_sys::Page, offset: usize) -> T {
    let src = page_get_data(page).add(offset);
    std::ptr::read_unaligned(src as *const T)
}

/// Write raw bytes at the given offset within the data area.
///
/// # Safety
///
/// Caller must ensure `offset + len <= INFER_USABLE_PER_PAGE`.
pub unsafe fn write_bytes_at(page: pg_sys::Page, offset: usize, data: &[u8]) {
    let dst = page_get_data(page).add(offset);
    std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
}

/// Read raw bytes from the given offset within the data area.
///
/// # Safety
///
/// Caller must ensure `offset + len <= INFER_USABLE_PER_PAGE`.
#[allow(dead_code)]
pub unsafe fn read_bytes_at(page: pg_sys::Page, offset: usize, len: usize) -> Vec<u8> {
    let src = page_get_data(page).add(offset);
    let mut buf = vec![0u8; len];
    std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), len);
    buf
}
