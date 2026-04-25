//! Index build logic: reads vindex files and writes PostgreSQL pages via
//! GenericXLog.

use std::path::Path;

use pgrx::pg_sys;

use crate::error::PgInferError;
use crate::pages::*;

/// P_NEW block number — tells ReadBufferExtended to extend the relation.
const P_NEW: pg_sys::BlockNumber = INVALID_BLOCK_NUMBER;

// ---------------------------------------------------------------------------
// Public entry point (called from am.rs ambuild callback)
// ---------------------------------------------------------------------------

/// Build the infer index by reading a vindex directory and writing its
/// contents into PostgreSQL pages.
///
/// # Safety
///
/// Must be called from the ambuild callback with valid relation pointers.
pub unsafe fn build_index(
    index_relation: pg_sys::Relation,
    source_path: &str,
) -> Result<pg_sys::IndexBuildResult, PgInferError> {
    let path = Path::new(source_path);

    // Load vindex config to get dimensions.
    let config = larql_vindex::load_vindex_config(path)?;
    let num_layers = config.num_layers;
    let hidden_size = config.hidden_size;
    let vocab_size = config.vocab_size;
    let embed_scale = config.embed_scale;

    // Get features_per_layer from the first layer's info.
    let features_per_layer = config
        .layers
        .first()
        .map(|l| l.num_features)
        .unwrap_or(0);

    // Load the raw data files we need.
    let gate_data = std::fs::read(path.join("gate_vectors.bin"))
        .map_err(|e| PgInferError::Internal(format!("read gate_vectors.bin: {}", e)))?;
    let embed_data = std::fs::read(path.join("embeddings.bin"))
        .map_err(|e| PgInferError::Internal(format!("read embeddings.bin: {}", e)))?;
    let tok_data = std::fs::read(path.join("tokenizer.json"))
        .map_err(|e| PgInferError::Internal(format!("read tokenizer.json: {}", e)))?;

    // down_meta is NDJSON, read as raw bytes for blob storage.
    let down_data = std::fs::read(path.join("down_meta.bin"))
        .unwrap_or_default(); // may not exist

    let bytes_per_gate_vec = hidden_size * 2; // f16
    let bytes_per_embed_vec = hidden_size * 2;
    let vecs_per_gate_pg = vectors_per_gate_page(hidden_size);
    let vecs_per_embed_pg = embeds_per_page(hidden_size);

    // Calculate page counts for each section.
    // Gate pages: per-layer, using each layer's feature count.
    let mut total_gate_pages = 0usize;
    for li in &config.layers {
        total_gate_pages += gate_pages_needed(li.num_features, hidden_size);
    }
    let embed_page_count = embed_pages_needed(vocab_size, hidden_size);
    let down_page_count = blob_pages_needed(down_data.len()); // store as blob
    let tok_page_count = blob_pages_needed(tok_data.len());

    // Block allocation plan.
    let _meta_blk: u32 = 0;
    let layer_dir_blk: u32 = 1;
    let gate_start_blk: u32 = 2;
    let gate_end_blk = gate_start_blk + total_gate_pages as u32;
    let embed_start_blk = gate_end_blk;
    let embed_end_blk = embed_start_blk + embed_page_count as u32;
    let down_start_blk = embed_end_blk;
    let down_end_blk = down_start_blk + down_page_count as u32;
    let tok_start_blk = down_end_blk;
    let tok_end_blk = tok_start_blk + tok_page_count as u32;
    let total_pages = tok_end_blk;

    // Build the metapage struct.
    let mut meta = InferMetaPage {
        magic: INFER_MAGIC,
        format_version: INFER_FORMAT_VERSION,
        model_name: [0u8; 128],
        num_layers: num_layers as u32,
        hidden_size: hidden_size as u32,
        features_per_layer: features_per_layer as u32,
        vocab_size: vocab_size as u32,
        embed_scale,
        gate_dtype: 1, // f16
        embed_dtype: 1, // f16
        down_top_k: config.down_top_k as u16,
        extract_level: 0, // browse
        _pad: [0u8; 3],
        layer_dir_blk,
        gate_start_blk,
        gate_end_blk,
        embed_start_blk,
        embed_end_blk,
        down_start_blk,
        down_end_blk,
        tok_start_blk,
        tok_end_blk,
        max_gate_score: 0.0,
        mean_gate_score: 0.0,
        total_pages,
        source_uri: [0u8; 256],
    };

    // Copy model name into metapage.
    let model_bytes = config.model.as_bytes();
    let copy_len = model_bytes.len().min(meta.model_name.len() - 1);
    meta.model_name[..copy_len].copy_from_slice(&model_bytes[..copy_len]);

    // Copy source URI.
    let src_bytes = source_path.as_bytes();
    let copy_len = src_bytes.len().min(meta.source_uri.len() - 1);
    meta.source_uri[..copy_len].copy_from_slice(&src_bytes[..copy_len]);

    // --- Write pages ---

    // Block 0: Metapage
    write_new_page(index_relation, PageType::Meta, |page| {
        write_struct_at(page, 0, &meta);
    })?;

    // Block 1: Layer directory
    write_new_page(index_relation, PageType::LayerDir, |page| {
        let mut gate_blk_cursor = gate_start_blk;
        for (i, li) in config.layers.iter().enumerate() {
            let layer_gate_count = gate_pages_needed(li.num_features, hidden_size);

            let entry = LayerDirEntry {
                layer_id: i as u16,
                num_features: li.num_features as u16,
                gate_start_blk: gate_blk_cursor,
                gate_page_count: layer_gate_count as u16,
                down_start_blk: down_start_blk, // down is stored as blob
                down_page_count: 0,
                reserved: [0u8; 4],
            };

            let offset = i * std::mem::size_of::<LayerDirEntry>();
            write_struct_at(page, offset, &entry);

            gate_blk_cursor += layer_gate_count as u32;
        }
    })?;

    // Gate vector pages — per layer, reading from gate_vectors.bin.
    for li in &config.layers {
        let layer_data_start = li.offset as usize;
        let _layer_data_len = li.length as usize;
        let nf = li.num_features;
        let mut feature_idx = 0usize;

        while feature_idx < nf {
            let vecs_this_page = vecs_per_gate_pg.min(nf - feature_idx);

            write_new_page(index_relation, PageType::Gate, |page| {
                let hdr = GatePageHeader {
                    layer_id: li.layer as u16,
                    num_vectors: vecs_this_page as u16,
                    first_feature_idx: feature_idx as u32,
                };
                write_struct_at(page, 0, &hdr);

                let hdr_size = std::mem::size_of::<GatePageHeader>();
                let data_start = layer_data_start + feature_idx * bytes_per_gate_vec;
                let data_len = vecs_this_page * bytes_per_gate_vec;

                if data_start + data_len <= gate_data.len() {
                    write_bytes_at(page, hdr_size, &gate_data[data_start..data_start + data_len]);
                }

                let opaque = page_get_opaque(page);
                (*opaque).layer_id = li.layer as u16;
            })?;

            feature_idx += vecs_this_page;
        }
    }

    // Embedding pages
    let mut token_idx = 0usize;
    while token_idx < vocab_size {
        let vecs_this_page = vecs_per_embed_pg.min(vocab_size - token_idx);

        write_new_page(index_relation, PageType::Embed, |page| {
            let hdr = EmbedPageHeader {
                layer_id: 0xFFFF,
                num_vectors: vecs_this_page as u16,
                first_feature_idx: token_idx as u32,
            };
            write_struct_at(page, 0, &hdr);

            let hdr_size = std::mem::size_of::<EmbedPageHeader>();
            let data_offset = token_idx * bytes_per_embed_vec;
            let data_len = vecs_this_page * bytes_per_embed_vec;

            if data_offset + data_len <= embed_data.len() {
                write_bytes_at(page, hdr_size, &embed_data[data_offset..data_offset + data_len]);
            }
        })?;

        token_idx += vecs_this_page;
    }

    // Down meta as blob pages (NDJSON data)
    write_blob_pages(index_relation, &down_data)?;

    // Tokenizer blob pages
    write_blob_pages(index_relation, &tok_data)?;

    // Build the result.
    let result = pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBuildResult>())
        as *mut pg_sys::IndexBuildResult;
    (*result).heap_tuples = 0.0;
    (*result).index_tuples = total_pages as f64;

    Ok(*result)
}

// ---------------------------------------------------------------------------
// Blob page writer (for tokenizer, down_meta, etc.)
// ---------------------------------------------------------------------------

/// Write a byte slice as a chain of blob pages.
///
/// # Safety
///
/// Must be called with a valid, locked relation.
unsafe fn write_blob_pages(
    rel: pg_sys::Relation,
    data: &[u8],
) -> Result<(), PgInferError> {
    if data.is_empty() {
        return Ok(());
    }

    let total_size = data.len();
    let mut offset = 0usize;

    while offset < total_size {
        let chunk_len = BLOB_DATA_BYTES.min(total_size - offset);
        let has_next = offset + chunk_len < total_size;

        // We need to know the current block number for the next_blkno chain.
        // Since we're extending one block at a time, we can read it after buffer
        // allocation.  For simplicity, set next_blkno after the fact.
        write_new_page(rel, PageType::Blob, |page| {
            let hdr = BlobPageHeader {
                total_size: total_size as u32,
                page_bytes: chunk_len as u32,
            };
            write_struct_at(page, 0, &hdr);

            let hdr_size = std::mem::size_of::<BlobPageHeader>();
            write_bytes_at(page, hdr_size, &data[offset..offset + chunk_len]);

            // We can't easily set next_blkno here because we don't know the
            // next block number yet.  Leave it as INVALID_BLOCK_NUMBER; the
            // reader can follow sequential blocks using the block ranges in
            // the metapage instead of chaining.
            let _ = has_next;
        })?;

        offset += chunk_len;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Page write helper
// ---------------------------------------------------------------------------

/// Extend the relation by one block, initialize it, run the callback to
/// fill in data, then WAL-log via GenericXLog.
///
/// # Safety
///
/// Must be called with a valid relation.
unsafe fn write_new_page<F>(
    rel: pg_sys::Relation,
    page_type: PageType,
    fill: F,
) -> Result<(), PgInferError>
where
    F: FnOnce(pg_sys::Page),
{
    // Extend the relation to get a new block.
    pg_sys::LockRelationForExtension(rel, pg_sys::ExclusiveLock as pg_sys::LOCKMODE);

    let buf = pg_sys::ReadBufferExtended(
        rel,
        pg_sys::ForkNumber::MAIN_FORKNUM,
        P_NEW,
        pg_sys::ReadBufferMode::RBM_ZERO_AND_LOCK,
        std::ptr::null_mut(),
    );

    pg_sys::UnlockRelationForExtension(rel, pg_sys::ExclusiveLock as pg_sys::LOCKMODE);

    // Start WAL-logged page modification.
    let state = pg_sys::GenericXLogStart(rel);
    let page = pg_sys::GenericXLogRegisterBuffer(
        state,
        buf,
        pg_sys::GENERIC_XLOG_FULL_IMAGE as i32,
    );

    // Initialize the page with our special area.
    init_page(page, page_type);

    // Let the caller fill in the data.
    fill(page);

    // Finish WAL logging and release.
    pg_sys::GenericXLogFinish(state);
    pg_sys::UnlockReleaseBuffer(buf);

    Ok(())
}
