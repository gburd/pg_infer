//! Read vindex data from PostgreSQL index pages.
//!
//! Provides [`PageBackend`], which decodes gate vectors, embeddings,
//! tokenizer, and down_meta data stored in PG buffer-managed pages written
//! by [`crate::build::build_index`].

use std::collections::HashMap;
use std::sync::Mutex;

use half::f16;
use ndarray::{Array1, Array2, ArrayView2};
use pgrx::pg_sys;

use infer_models::TopKEntry;
use infer_vindex::FeatureMeta;

use crate::error::PgInferError;
use crate::pages::*;

// ---------------------------------------------------------------------------
// PageBackend
// ---------------------------------------------------------------------------

/// Backend state for a model stored in PG index pages.
///
/// On first access, the metapage, layer directory, embeddings, tokenizer,
/// and down_meta are read from pages into Rust heap memory.  Gate vectors
/// are decoded lazily per-layer and cached.
pub struct PageBackend {
    /// OID of the infer index relation.
    pub index_oid: pg_sys::Oid,
    /// Copy of the metapage.
    pub meta: InferMetaPage,
    /// Layer directory entries.
    pub layer_dir: Vec<LayerDirEntry>,
    /// Raw down_meta binary data (reconstructed from blob pages).
    down_meta_data: Vec<u8>,
    /// Per-layer (byte_offset, num_features) into `down_meta_data`.
    down_meta_offsets: Vec<(usize, usize)>,
    /// Record size for down_meta: `8 + top_k_count * 8`.
    down_meta_record_size: usize,
    /// Per-layer decoded f32 gate vectors (lazy-loaded).
    gate_cache: Mutex<HashMap<u16, Vec<f32>>>,
}

// SAFETY: PageBackend is only accessed from a single PG backend (one OS
// process, one thread).  The Mutex satisfies Rust's Send requirement for
// statics but has zero real contention.
unsafe impl Send for PageBackend {}
unsafe impl Sync for PageBackend {}

impl PageBackend {
    /// Load a `PageBackend` by reading from the PG index pages identified
    /// by `index_oid`.
    ///
    /// Returns `(backend, embeddings, embed_scale, tokenizer)`.
    ///
    /// # Safety
    ///
    /// Must be called within a valid PG transaction context.
    pub unsafe fn load(
        index_oid: pg_sys::Oid,
    ) -> Result<
        (
            Self,
            Array2<f32>,
            f32,
            infer_vindex::tokenizers::Tokenizer,
        ),
        PgInferError,
    > {
        let rel = pg_sys::relation_open(index_oid, pg_sys::AccessShareLock as _);

        // Read metapage (block 0).
        let meta = read_meta(rel)?;
        if meta.magic != INFER_MAGIC {
            pg_sys::relation_close(rel, pg_sys::AccessShareLock as _);
            return Err(PgInferError::Internal(format!(
                "invalid infer index: bad magic number 0x{:08X} (expected 0x{:08X}). \
                 This may indicate a stale index from a previous build — try: \
                 SELECT infer_drop_model('...'); then re-register.",
                meta.magic, INFER_MAGIC,
            )));
        }

        // Layer directory (block 1).
        let layer_dir = read_layer_dir(rel, meta.num_layers as usize)?;

        // Embeddings.
        let embeddings = read_embeddings(rel, &meta)?;
        let embed_scale = meta.embed_scale;

        // Tokenizer blob.
        let tok_json = read_blob_data(rel, meta.tok_start_blk, meta.tok_end_blk);
        let tokenizer = infer_vindex::tokenizers::Tokenizer::from_bytes(&tok_json)
            .map_err(|e| PgInferError::Internal(format!("parse tokenizer: {}", e)))?;

        // Down meta blob.
        let down_meta_data = read_blob_data(rel, meta.down_start_blk, meta.down_end_blk);

        pg_sys::relation_close(rel, pg_sys::AccessShareLock as _);

        // Parse down_meta layer offsets.
        let top_k_count = meta.down_top_k as usize;
        let record_size = 8 + top_k_count * 8;
        let down_meta_offsets = parse_down_meta_offsets(
            &down_meta_data,
            meta.num_layers as usize,
            record_size,
        );

        let backend = PageBackend {
            index_oid,
            meta,
            layer_dir,
            down_meta_data,
            down_meta_offsets,
            down_meta_record_size: record_size,
            gate_cache: Mutex::new(HashMap::new()),
        };

        Ok((backend, embeddings, embed_scale, tokenizer))
    }

    /// Compute gate KNN for a single layer.
    ///
    /// On first call per layer, reads gate pages and decodes f16 → f32.
    /// Subsequent calls use the cached decoded data.
    pub fn gate_knn(
        &self,
        layer: usize,
        query: &Array1<f32>,
        top_k: usize,
        hidden_size: usize,
    ) -> Vec<(usize, f32)> {
        let mut cache = self.gate_cache.lock().expect("gate cache poisoned");

        if !cache.contains_key(&(layer as u16)) {
            let data = unsafe { self.read_gate_layer(layer, hidden_size) };
            cache.insert(layer as u16, data);
        }

        let gate_data = cache.get(&(layer as u16)).expect("just inserted");
        let num_features = gate_data.len() / hidden_size;
        if num_features == 0 || hidden_size == 0 {
            return vec![];
        }

        let gates = match ArrayView2::from_shape((num_features, hidden_size), gate_data) {
            Ok(v) => v,
            Err(_) => return vec![],
        };
        let scores = gates.dot(query);
        top_k_from_scores(&scores, top_k)
    }

    /// Decode feature metadata from the down_meta binary blob.
    pub fn feature_meta(
        &self,
        layer: usize,
        feature_idx: usize,
        tokenizer: &infer_vindex::tokenizers::Tokenizer,
    ) -> Option<FeatureMeta> {
        if layer >= self.down_meta_offsets.len() {
            return None;
        }
        let (base_offset, num_features) = self.down_meta_offsets[layer];
        if feature_idx >= num_features {
            return None;
        }

        let record_offset = base_offset + feature_idx * self.down_meta_record_size;
        parse_one_feature(
            &self.down_meta_data,
            record_offset,
            self.meta.down_top_k as usize,
            tokenizer,
        )
    }

    // -----------------------------------------------------------------------
    // Internal page readers
    // -----------------------------------------------------------------------

    /// Read and decode all gate vectors for one layer from PG pages.
    ///
    /// # Safety
    ///
    /// Must be called within a valid PG transaction context.
    unsafe fn read_gate_layer(&self, layer: usize, hidden_size: usize) -> Vec<f32> {
        if layer >= self.layer_dir.len() {
            return vec![];
        }

        let entry = &self.layer_dir[layer];
        let gate_start = entry.gate_start_blk;
        let gate_count = entry.gate_page_count;
        let num_features = entry.num_features as usize;

        let mut result = Vec::with_capacity(num_features * hidden_size);

        let rel = pg_sys::relation_open(self.index_oid, pg_sys::AccessShareLock as _);

        for blk_offset in 0..gate_count {
            let blkno = gate_start + blk_offset as u32;
            let buf = pg_sys::ReadBuffer(rel, blkno);
            pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
            let page = pg_sys::BufferGetPage(buf);

            let data_ptr = page_get_data(page);
            let hdr: GatePageHeader =
                std::ptr::read_unaligned(data_ptr as *const GatePageHeader);
            let num_vectors = hdr.num_vectors as usize;

            let hdr_size = std::mem::size_of::<GatePageHeader>();
            let f16_ptr = data_ptr.add(hdr_size);
            let f16_bytes = num_vectors * hidden_size * 2;
            let f16_slice = std::slice::from_raw_parts(f16_ptr, f16_bytes);

            // Decode f16 → f32.
            for chunk in f16_slice.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                result.push(f16::from_bits(bits).to_f32());
            }

            pg_sys::UnlockReleaseBuffer(buf);
        }

        pg_sys::relation_close(rel, pg_sys::AccessShareLock as _);

        result
    }
}

// ---------------------------------------------------------------------------
// Page reading helpers
// ---------------------------------------------------------------------------

/// Read the metapage (block 0) from the index relation.
///
/// # Safety
///
/// `rel` must be a valid, open relation with at least block 0.
unsafe fn read_meta(rel: pg_sys::Relation) -> Result<InferMetaPage, PgInferError> {
    let buf = pg_sys::ReadBuffer(rel, 0);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = pg_sys::BufferGetPage(buf);

    let data_ptr = page_get_data(page);
    let meta: InferMetaPage = std::ptr::read_unaligned(data_ptr as *const InferMetaPage);

    pg_sys::UnlockReleaseBuffer(buf);
    Ok(meta)
}

/// Read the layer directory (block 1).
///
/// # Safety
///
/// `rel` must be a valid, open relation with at least block 1.
unsafe fn read_layer_dir(
    rel: pg_sys::Relation,
    num_layers: usize,
) -> Result<Vec<LayerDirEntry>, PgInferError> {
    let buf = pg_sys::ReadBuffer(rel, 1);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = pg_sys::BufferGetPage(buf);

    let data_ptr = page_get_data(page);
    let entry_size = std::mem::size_of::<LayerDirEntry>();
    let mut entries = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let entry: LayerDirEntry =
            std::ptr::read_unaligned(data_ptr.add(i * entry_size) as *const LayerDirEntry);
        entries.push(entry);
    }

    pg_sys::UnlockReleaseBuffer(buf);
    Ok(entries)
}

/// Read embedding pages and decode f16 → f32 into an Array2.
///
/// # Safety
///
/// `rel` must be a valid, open relation.
unsafe fn read_embeddings(
    rel: pg_sys::Relation,
    meta: &InferMetaPage,
) -> Result<Array2<f32>, PgInferError> {
    let hidden_size = meta.hidden_size as usize;
    let vocab_size = meta.vocab_size as usize;
    let total_floats = vocab_size * hidden_size;

    let mut data = Vec::with_capacity(total_floats);

    for blkno in meta.embed_start_blk..meta.embed_end_blk {
        let buf = pg_sys::ReadBuffer(rel, blkno);
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
        let page = pg_sys::BufferGetPage(buf);

        let data_ptr = page_get_data(page);
        let hdr: GatePageHeader =
            std::ptr::read_unaligned(data_ptr as *const GatePageHeader);
        let num_vectors = hdr.num_vectors as usize;

        let hdr_size = std::mem::size_of::<GatePageHeader>();
        let f16_ptr = data_ptr.add(hdr_size);
        let f16_bytes = num_vectors * hidden_size * 2;
        let f16_slice = std::slice::from_raw_parts(f16_ptr, f16_bytes);

        for chunk in f16_slice.chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            data.push(f16::from_bits(bits).to_f32());
        }

        pg_sys::UnlockReleaseBuffer(buf);
    }

    // Truncate to exact vocab_size × hidden_size (last page may have padding).
    data.truncate(total_floats);

    Array2::from_shape_vec((vocab_size, hidden_size), data).map_err(|e| {
        PgInferError::Internal(format!("embedding shape mismatch: {}", e))
    })
}

/// Read sequential blob pages and reconstruct the original byte slice.
///
/// # Safety
///
/// `rel` must be a valid, open relation.
unsafe fn read_blob_data(
    rel: pg_sys::Relation,
    start_blk: u32,
    end_blk: u32,
) -> Vec<u8> {
    let mut data = Vec::new();

    for blkno in start_blk..end_blk {
        let buf = pg_sys::ReadBuffer(rel, blkno);
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
        let page = pg_sys::BufferGetPage(buf);

        let data_ptr = page_get_data(page);
        let hdr: BlobPageHeader =
            std::ptr::read_unaligned(data_ptr as *const BlobPageHeader);

        let hdr_size = std::mem::size_of::<BlobPageHeader>();
        let blob_ptr = data_ptr.add(hdr_size);
        let chunk = std::slice::from_raw_parts(blob_ptr, hdr.page_bytes as usize);
        data.extend_from_slice(chunk);

        pg_sys::UnlockReleaseBuffer(buf);
    }

    data
}

// ---------------------------------------------------------------------------
// Down meta binary parsing
// ---------------------------------------------------------------------------

/// Parse the down_meta binary header to compute per-layer byte offsets.
///
/// The binary format (from infer-vindex):
/// ```text
/// Header (16 bytes):
///   magic: u32 = 0x444D4554 ("DMET")
///   version: u32 = 1
///   num_layers: u32
///   top_k_count: u32
/// Per-layer:
///   num_features: u32
///   num_features × record (8 + top_k_count × 8 bytes each)
/// ```
fn parse_down_meta_offsets(
    data: &[u8],
    num_layers: usize,
    record_size: usize,
) -> Vec<(usize, usize)> {
    if data.len() < 16 {
        return vec![(0, 0); num_layers];
    }

    // Validate magic.
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != 0x444D4554 {
        // Not a valid down_meta binary; return empty offsets.
        return vec![(0, 0); num_layers];
    }

    let mut offsets = Vec::with_capacity(num_layers);
    let mut cursor = 16usize; // skip header

    for _ in 0..num_layers {
        if cursor + 4 > data.len() {
            offsets.push((0, 0));
            continue;
        }
        let nf = u32::from_le_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]) as usize;
        cursor += 4;

        let records_start = cursor;
        offsets.push((records_start, nf));

        // Advance past all records for this layer.
        cursor += nf * record_size;
    }

    offsets
}

/// Parse one FeatureMeta record from the raw down_meta bytes.
fn parse_one_feature(
    data: &[u8],
    offset: usize,
    top_k_count: usize,
    tokenizer: &infer_vindex::tokenizers::Tokenizer,
) -> Option<FeatureMeta> {
    let record_size = 8 + top_k_count * 8;
    if offset + record_size > data.len() {
        return None;
    }

    let top_token_id = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]);

    let c_score = f32::from_le_bytes([
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ]);

    let mut top_k = Vec::with_capacity(top_k_count);
    for i in 0..top_k_count {
        let base = offset + 8 + i * 8;
        let tid = u32::from_le_bytes([
            data[base],
            data[base + 1],
            data[base + 2],
            data[base + 3],
        ]);
        let logit = f32::from_le_bytes([
            data[base + 4],
            data[base + 5],
            data[base + 6],
            data[base + 7],
        ]);

        // Skip zero entries (empty padding).
        if tid == 0 && logit == 0.0 {
            continue;
        }

        let token = tokenizer
            .decode(&[tid], true)
            .unwrap_or_else(|_| format!("T{tid}"))
            .trim()
            .to_string();

        top_k.push(TopKEntry {
            token,
            token_id: tid,
            logit,
        });
    }

    let top_token = tokenizer
        .decode(&[top_token_id], true)
        .unwrap_or_else(|_| format!("T{top_token_id}"))
        .trim()
        .to_string();

    Some(FeatureMeta {
        top_token,
        top_token_id,
        c_score,
        top_k,
    })
}

// ---------------------------------------------------------------------------
// Top-K selection (matches infer-vindex's algorithm)
// ---------------------------------------------------------------------------

/// Select the top-K entries by absolute score magnitude.
fn top_k_from_scores(scores: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    let k = top_k.min(indexed.len());
    if k > 0 && k < indexed.len() {
        indexed.select_nth_unstable_by(k, |a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
    }
    indexed.sort_unstable_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed
}
