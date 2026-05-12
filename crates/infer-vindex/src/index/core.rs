//! VectorIndex struct and core operations.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ndarray::Array2;

// Re-export all shared types from types.rs.
pub use super::types::*;

/// The full model as a local vector index.
///
/// Gate vectors for KNN matching + down token metadata for output lookup.
/// Supports two storage modes:
/// - **Heap**: gate vectors copied into per-layer Array2 (in-memory builds, mutations)
/// - **Mmap**: gate vectors sliced directly from mmap'd file (zero-copy, zero heap)
pub struct VectorIndex {
    /// Per-layer gate vectors (heap mode): gate_vectors[layer] is (num_features, hidden_size).
    pub(crate) gate_vectors: Vec<Option<Array2<f32>>>,

    /// Mmap'd gate vector bytes (zero-copy mode). When set, gate_knn slices
    /// directly from this instead of using gate_vectors heap arrays.
    /// For f32: bytes are reinterpreted as &[f32] directly (zero-copy).
    /// For f16: bytes are decoded per-layer on demand.
    /// Arc for Clone support — the mmap is shared, not copied.
    pub(crate) gate_mmap_bytes: Option<Arc<memmap2::Mmap>>,

    /// Storage dtype for mmap'd data (needed for f16 decoding).
    pub(crate) gate_mmap_dtype: crate::config::dtype::StorageDtype,

    /// Per-layer slice info for mmap mode.
    pub(crate) gate_mmap_slices: Vec<GateLayerSlice>,

    /// Per-layer, per-feature output token metadata from down projections.
    /// down_meta[layer][feature] = FeatureMeta with top tokens.
    /// Heap mode: populated during builds or when loaded from JSONL.
    pub(crate) down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,

    /// Mmap'd down_meta.bin bytes (zero-copy mode).
    /// When set, feature_meta() reads records on demand from the mmap.
    pub(crate) down_meta_mmap: Option<Arc<DownMetaMmap>>,

    /// Number of layers in the model.
    pub num_layers: usize,

    /// Hidden dimension.
    pub hidden_size: usize,

    /// Down vector overrides: custom output vectors for specific features.
    /// When set, sparse_ffn_forward uses this instead of the model's down weight row.
    /// Key: (layer, feature), Value: hidden_size f32 vector.
    pub(crate) down_overrides: HashMap<(usize, usize), Vec<f32>>,

    /// Up vector overrides: custom up vectors for specific features.
    /// Parallel to down_overrides — when set, walk_ffn_sparse uses this
    /// instead of the model's up_features row at that slot. INSERT
    /// writes to this so the slot's activation = silu(gate·x) * (up·x)
    /// reflects the constellation, not the original weak free-slot up.
    /// Key: (layer, feature), Value: hidden_size f32 vector.
    pub(crate) up_overrides: HashMap<(usize, usize), Vec<f32>>,

    /// Lazy decode cache for f16 gate vectors. Each layer decoded once on first
    /// KNN call, then reused. Eliminates repeated f16→f32 conversion.
    pub(crate) f16_decode_cache: Mutex<Vec<Option<Vec<f32>>>>,
    /// LRU queue for `f16_decode_cache`. Back is oldest, front is newest.
    /// Used with `gate_cache_max_layers` to cap decoded-gate heap growth
    /// (a 31B f16 gate table decodes to ~26 GB if all 60 layers are kept).
    pub(crate) gate_cache_lru: Mutex<std::collections::VecDeque<usize>>,
    /// Cap on live entries in `f16_decode_cache`. 0 = unlimited (default —
    /// historical behaviour, max speed). Set via `set_gate_cache_max_layers`
    /// to bound RSS growth. When an insert would exceed the cap, the
    /// least-recently-used layer is dropped.
    pub(crate) gate_cache_max_layers: std::sync::atomic::AtomicUsize,
    pub(crate) warmed_gates: std::sync::RwLock<Vec<Option<Vec<f32>>>>,
    pub(crate) down_features_mmap: Option<Arc<memmap2::Mmap>>,
    pub(crate) up_features_mmap: Option<Arc<memmap2::Mmap>>,
    pub(crate) hnsw_cache: Mutex<Vec<Option<super::hnsw::HnswLayer>>>,
    pub(crate) hnsw_enabled: std::sync::atomic::AtomicBool,
    pub(crate) hnsw_ef_search: std::sync::atomic::AtomicUsize,
    /// Mmap'd lm_head (output projection): [vocab_size, hidden_size], f32.
    pub(crate) lm_head_mmap: Option<Arc<memmap2::Mmap>>,
    /// Mmap'd lm_head as f16 — typically the tied-embedding case where the
    /// vindex's `embeddings.bin` is the output projection. Carried by
    /// `VectorIndex` so `lm_head_knn_backend` can dispatch to Metal's
    /// `f16_gemv` without materialising a 5.6 GB f32 clone on 31B.
    pub(crate) lm_head_f16_mmap: Option<Arc<memmap2::Mmap>>,
    pub vocab_size: usize,
    /// Interleaved FFN data: [gate|up|down] per layer in one contiguous file.
    pub(crate) interleaved_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_0 quantized interleaved FFN data (7x smaller, dequant on read).
    pub(crate) interleaved_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_K/Q6_K quantized interleaved FFN data (Ollama-compatible, matches attn format).
    pub(crate) interleaved_q4k_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-matrix (offset, length, format) entries for `interleaved_q4k.bin`,
    /// 3 per layer in [gate, up, down] order. Required because the Ollama
    /// strategy mixes Q4_K (gate/up) with Q6_K (down), so layer stride is
    /// not uniform and callers cannot compute offsets from shape alone.
    pub(crate) interleaved_q4k_manifest: Option<Vec<(usize, usize, String)>>,
    /// Per-layer lazy decode cache for Q4K/Q6K FFN tensors.
    /// `q4k_ffn_cache[layer][c]` is the dequantised `[intermediate × hidden]`
    /// matrix for component `c` (0=gate, 1=up, 2=down). Populated on first
    /// access via `q4k_ffn_layer`. Backs `walk_ffn_sparse`'s f32 view when
    /// no native f32 mmap exists (Q4K-only vindexes).
    #[allow(clippy::type_complexity)]
    pub(crate) q4k_ffn_cache: Mutex<Vec<[Option<Arc<Vec<f32>>>; 3]>>,

    /// Relation cluster labels: maps (layer, feature) → cluster label string.
    /// Loaded from `feature_clusters.jsonl` + `relation_clusters.json` at vindex load time.
    /// Used by `feature_meta()` to populate `FeatureMeta.relation`.
    pub(crate) relation_labels: HashMap<(usize, usize), String>,

    /// Layer range owned by this index instance (start inclusive, end exclusive).
    /// `None` means all layers are owned (default, no sharding).
    /// Set via `load_vindex_with_range` to restrict which layers are served,
    /// preventing accidental page faults into out-of-shard mmap regions.
    pub(crate) layer_range: Option<(usize, usize)>,

    /// Q4_0 gate vectors mmap — for fast Q4 KNN via infer-compute.
    pub(crate) gate_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-layer byte offset + byte length in gate_q4_mmap.
    pub(crate) gate_q4_slices: Vec<GateQ4Slice>,
    /// Q4_0 lm_head mmap — for GPU Q4 logits (replaces CPU f32 lm_head KNN).
    pub(crate) lm_head_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_0 lm_head synthesized in RAM from f16 embeddings at load time.
    pub(crate) lm_head_q4_synth: Option<Arc<Vec<u8>>>,
    /// Q4_K/Q6_K attention weights (Ollama-compatible).
    pub(crate) attn_q4k_mmap: Option<Arc<memmap2::Mmap>>,
    pub(crate) attn_q4k_manifest: Option<Vec<(usize, usize, String)>>,
    /// Q4_0 attention weights mmap — for GPU full pipeline.
    pub(crate) attn_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-matrix (offset, length) in attn_q4_mmap — from manifest.
    pub(crate) attn_q4_manifest: Option<Vec<(usize, usize)>>,
    /// Q8_0 attention weights mmap — higher precision for attention projections.
    pub(crate) attn_q8_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-matrix (offset, vals_len, scales_len) in attn_q8_mmap.
    pub(crate) attn_q8_manifest: Option<Vec<(usize, usize, usize)>>,
}

impl Clone for VectorIndex {
    fn clone(&self) -> Self {
        use std::sync::atomic::Ordering;
        Self {
            gate_vectors: self.gate_vectors.clone(),
            gate_mmap_bytes: self.gate_mmap_bytes.clone(),
            gate_mmap_dtype: self.gate_mmap_dtype,
            gate_mmap_slices: self.gate_mmap_slices.clone(),
            down_meta: self.down_meta.clone(),
            down_meta_mmap: self.down_meta_mmap.clone(),
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            down_overrides: self.down_overrides.clone(),
            up_overrides: self.up_overrides.clone(),
            f16_decode_cache: Mutex::new(vec![None; self.num_layers]),
            gate_cache_lru: Mutex::new(std::collections::VecDeque::new()),
            gate_cache_max_layers: std::sync::atomic::AtomicUsize::new(
                self.gate_cache_max_layers.load(std::sync::atomic::Ordering::Relaxed),
            ),
            warmed_gates: std::sync::RwLock::new(vec![None; self.num_layers]),
            down_features_mmap: self.down_features_mmap.clone(),
            up_features_mmap: self.up_features_mmap.clone(),
            hnsw_cache: Mutex::new((0..self.num_layers).map(|_| None).collect()),
            hnsw_enabled: std::sync::atomic::AtomicBool::new(
                self.hnsw_enabled.load(Ordering::Relaxed)
            ),
            hnsw_ef_search: std::sync::atomic::AtomicUsize::new(
                self.hnsw_ef_search.load(Ordering::Relaxed)
            ),
            lm_head_mmap: self.lm_head_mmap.clone(),
            lm_head_f16_mmap: self.lm_head_f16_mmap.clone(),
            vocab_size: self.vocab_size,
            interleaved_mmap: self.interleaved_mmap.clone(),
            interleaved_q4_mmap: self.interleaved_q4_mmap.clone(),
            interleaved_q4k_mmap: self.interleaved_q4k_mmap.clone(),
            interleaved_q4k_manifest: self.interleaved_q4k_manifest.clone(),
            q4k_ffn_cache: Mutex::new(
                (0..self.num_layers).map(|_| [None, None, None]).collect(),
            ),
            gate_q4_mmap: self.gate_q4_mmap.clone(),
            gate_q4_slices: self.gate_q4_slices.clone(),
            lm_head_q4_mmap: self.lm_head_q4_mmap.clone(),
            lm_head_q4_synth: self.lm_head_q4_synth.clone(),
            attn_q4k_mmap: self.attn_q4k_mmap.clone(),
            attn_q4k_manifest: self.attn_q4k_manifest.clone(),
            attn_q4_mmap: self.attn_q4_mmap.clone(),
            attn_q4_manifest: self.attn_q4_manifest.clone(),
            attn_q8_mmap: self.attn_q8_mmap.clone(),
            attn_q8_manifest: self.attn_q8_manifest.clone(),
            relation_labels: self.relation_labels.clone(),
            layer_range: self.layer_range,
        }
    }
}

impl VectorIndex {
    /// Create a new VectorIndex from heap-allocated components (in-memory builds).
    pub fn new(
        gate_vectors: Vec<Option<Array2<f32>>>,
        down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,
        num_layers: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            gate_vectors,
            gate_mmap_bytes: None,
            gate_mmap_dtype: crate::config::dtype::StorageDtype::F32,
            gate_mmap_slices: Vec::new(),
            down_meta,
            down_meta_mmap: None,
            num_layers,
            hidden_size,
            down_overrides: HashMap::new(),
            up_overrides: HashMap::new(),
            f16_decode_cache: Mutex::new(vec![None; num_layers]),
            gate_cache_lru: Mutex::new(std::collections::VecDeque::new()),
            gate_cache_max_layers: std::sync::atomic::AtomicUsize::new(0),
            warmed_gates: std::sync::RwLock::new(vec![None; num_layers]),
            down_features_mmap: None,
            up_features_mmap: None,
            hnsw_cache: Mutex::new((0..num_layers).map(|_| None).collect()),
            hnsw_enabled: std::sync::atomic::AtomicBool::new(false),
            hnsw_ef_search: std::sync::atomic::AtomicUsize::new(200),
            lm_head_mmap: None,
            lm_head_f16_mmap: None,
            vocab_size: 0,
            interleaved_mmap: None,
            interleaved_q4_mmap: None,
            interleaved_q4k_mmap: None,
            interleaved_q4k_manifest: None,
            q4k_ffn_cache: Mutex::new((0..num_layers).map(|_| [None, None, None]).collect()),
            relation_labels: HashMap::new(),
            layer_range: None,
            gate_q4_mmap: None,
            gate_q4_slices: Vec::new(),
            lm_head_q4_mmap: None,
            lm_head_q4_synth: None,
            attn_q4k_mmap: None,
            attn_q4k_manifest: None,
            attn_q4_mmap: None,
            attn_q4_manifest: None,
            attn_q8_mmap: None,
            attn_q8_manifest: None,
        }
    }

    /// Create a VectorIndex with zero-copy mmap'd gate vectors and down_meta.
    /// No heap allocation — everything read on demand from mmap'd files.
    pub fn new_mmap(
        gate_mmap: memmap2::Mmap,
        gate_slices: Vec<GateLayerSlice>,
        dtype: crate::config::dtype::StorageDtype,
        down_meta_mmap: Option<DownMetaMmap>,
        num_layers: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            gate_vectors: vec![None; num_layers],
            gate_mmap_bytes: Some(Arc::new(gate_mmap)),
            gate_mmap_dtype: dtype,
            gate_mmap_slices: gate_slices,
            down_meta: vec![None; num_layers],
            down_meta_mmap: down_meta_mmap.map(Arc::new),
            num_layers,
            hidden_size,
            down_overrides: HashMap::new(),
            up_overrides: HashMap::new(),
            f16_decode_cache: Mutex::new(vec![None; num_layers]),
            gate_cache_lru: Mutex::new(std::collections::VecDeque::new()),
            gate_cache_max_layers: std::sync::atomic::AtomicUsize::new(0),
            warmed_gates: std::sync::RwLock::new(vec![None; num_layers]),
            down_features_mmap: None,
            up_features_mmap: None,
            hnsw_cache: Mutex::new((0..num_layers).map(|_| None).collect()),
            hnsw_enabled: std::sync::atomic::AtomicBool::new(false),
            hnsw_ef_search: std::sync::atomic::AtomicUsize::new(200),
            lm_head_mmap: None,
            lm_head_f16_mmap: None,
            vocab_size: 0,
            interleaved_mmap: None,
            interleaved_q4_mmap: None,
            interleaved_q4k_mmap: None,
            interleaved_q4k_manifest: None,
            q4k_ffn_cache: Mutex::new((0..num_layers).map(|_| [None, None, None]).collect()),
            relation_labels: HashMap::new(),
            layer_range: None,
            gate_q4_mmap: None,
            gate_q4_slices: Vec::new(),
            lm_head_q4_mmap: None,
            lm_head_q4_synth: None,
            attn_q4k_mmap: None,
            attn_q4k_manifest: None,
            attn_q4_mmap: None,
            attn_q4_manifest: None,
            attn_q8_mmap: None,
            attn_q8_manifest: None,
        }
    }

    /// Returns true if this index uses mmap'd gate vectors (zero heap copy).
    pub fn is_mmap(&self) -> bool {
        self.gate_mmap_bytes.is_some()
    }

    /// Load relation cluster labels from `relation_clusters.json` and
    /// `feature_clusters.jsonl` in the given directory. Populates the
    /// `relation_labels` map so `feature_meta()` can return relation labels.
    ///
    /// Silently returns Ok(0) if either file is missing (optional data).
    pub fn load_relation_labels(&mut self, dir: &std::path::Path) -> Result<usize, crate::VindexError> {
        use std::io::BufRead;

        let clusters_path = dir.join("relation_clusters.json");
        let assignments_path = dir.join("feature_clusters.jsonl");

        if !clusters_path.exists() || !assignments_path.exists() {
            return Ok(0);
        }

        // Parse cluster result to get labels per cluster index.
        let clusters_text = std::fs::read_to_string(&clusters_path)
            .map_err(crate::VindexError::Io)?;
        let cluster_result: crate::clustering::ClusterResult = serde_json::from_str(&clusters_text)
            .map_err(|e| crate::VindexError::Parse(format!("relation_clusters.json: {e}")))?;

        // Build labels array: use explicit labels if present, otherwise
        // join top_tokens as a fallback label.
        let labels: Vec<String> = if !cluster_result.labels.is_empty() {
            cluster_result.labels
        } else {
            cluster_result
                .top_tokens
                .iter()
                .map(|tokens| tokens.join("/"))
                .collect()
        };

        // Parse feature assignments (NDJSON: {"l": layer, "f": feature, "c": cluster})
        let file = std::fs::File::open(&assignments_path)
            .map_err(crate::VindexError::Io)?;
        let reader = std::io::BufReader::new(file);
        let mut count = 0usize;

        for line in reader.lines() {
            let line = line.map_err(crate::VindexError::Io)?;
            if line.is_empty() {
                continue;
            }
            let record: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| crate::VindexError::Parse(format!("feature_clusters.jsonl: {e}")))?;
            let layer = record["l"].as_u64().unwrap_or(0) as usize;
            let feature = record["f"].as_u64().unwrap_or(0) as usize;
            let cluster = record["c"].as_u64().unwrap_or(0) as usize;

            if let Some(label) = labels.get(cluster) {
                if !label.is_empty() {
                    self.relation_labels.insert((layer, feature), label.clone());
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Estimated heap bytes used by gate vectors (0 if mmap'd).
    pub fn gate_heap_bytes(&self) -> usize {
        if self.is_mmap() {
            return 0;
        }
        self.gate_vectors.iter()
            .filter_map(|v| v.as_ref())
            .map(|m| m.len() * std::mem::size_of::<f32>())
            .sum()
    }

    /// Approximate resident bytes: mmap regions + decoded gate cache.
    ///
    /// Used by pg_infer's backend LRU cache for eviction decisions.
    pub fn approx_resident_bytes(&self) -> usize {
        let mut total = 0usize;

        // Mmap'd gate vectors
        if let Some(ref m) = self.gate_mmap_bytes {
            total += m.len();
        }

        // Heap-mode gate vectors
        total += self.gate_heap_bytes();

        // f16 decode cache (decoded layers kept in memory)
        if let Ok(cache) = self.f16_decode_cache.lock() {
            for entry in cache.iter().flatten() {
                total += entry.len() * std::mem::size_of::<f32>();
            }
        }

        // Down-meta mmap
        if let Some(ref m) = self.down_meta_mmap {
            total += m.mmap.len();
        }

        // Interleaved FFN mmaps
        if let Some(ref m) = self.interleaved_mmap {
            total += m.len();
        }
        if let Some(ref m) = self.interleaved_q4_mmap {
            total += m.len();
        }
        if let Some(ref m) = self.interleaved_q4k_mmap {
            total += m.len();
        }

        // lm_head mmaps
        if let Some(ref m) = self.lm_head_mmap {
            total += m.len();
        }
        if let Some(ref m) = self.lm_head_f16_mmap {
            total += m.len();
        }

        // Down/up features mmaps
        if let Some(ref m) = self.down_features_mmap {
            total += m.len();
        }
        if let Some(ref m) = self.up_features_mmap {
            total += m.len();
        }

        total
    }

    /// Returns true if `layer` is owned by this shard (always true when no
    /// range is set). Use this to guard accessor calls and reject requests
    /// for layers outside the server's owned range before touching mmap pages.
    pub fn is_layer_owned(&self, layer: usize) -> bool {
        match self.layer_range {
            None => true,
            Some((start, end)) => layer >= start && layer < end,
        }
    }

    /// Returns the owned layer range `(start_inclusive, end_exclusive)`, or
    /// `None` if all layers are served.
    pub fn owned_layer_range(&self) -> Option<(usize, usize)> {
        self.layer_range
    }

    /// Set the owned layer range (used by `load_vindex_with_range`).
    pub(crate) fn set_layer_range(&mut self, range: (usize, usize)) {
        self.layer_range = Some(range);
    }
}
