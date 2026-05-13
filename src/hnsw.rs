//! Hierarchical Navigable Small World (HNSW) graph for approximate nearest
//! neighbor search.
//!
//! This module provides:
//! - `HnswBuilder`: in-memory graph construction used during index build.
//! - `HnswSearcher`: read-only searcher that operates on serialized page data.
//!
//! The implementation follows the original HNSW paper (Malkov & Yashunin, 2018)
//! with configurable M (max neighbors per layer) and ef_construction parameters.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::error::PgInferError;
use crate::sq8::asymmetric_distance_sq8_squared;

// ---------------------------------------------------------------------------
// HnswBuilder: in-memory construction
// ---------------------------------------------------------------------------

/// A node in the HNSW graph.
#[derive(Clone)]
struct HnswNode {
    /// Neighbors at each level. Level 0 has up to 2*M neighbors; higher levels
    /// have up to M neighbors.
    neighbors: Vec<Vec<u32>>,
    /// The level this node was inserted at.
    level: usize,
}

/// In-memory HNSW graph builder.
///
/// Builds the graph incrementally using `insert()`, then serializes to bytes
/// for storage in index pages.
pub struct HnswBuilder {
    /// All nodes in insertion order. Index = node ID.
    nodes: Vec<HnswNode>,
    /// Maximum number of neighbors per node per layer (M for layer > 0).
    m: usize,
    /// Maximum neighbors at level 0 (typically 2*M).
    m_max0: usize,
    /// Construction beam width.
    ef_construction: usize,
    /// Entry point node ID.
    entry_point: Option<u32>,
    /// Maximum level in the graph.
    max_level: usize,
    /// Reciprocal of ln(M) for level generation.
    level_mult: f64,
    /// Embeddings: SQ8 quantized data for distance computation.
    /// Each entry: (quantized_bytes, min, max).
    embeddings: Vec<(Vec<u8>, f32, f32)>,
    /// Embedding dimensionality.
    dim: usize,
}

/// Search result: (node_id, distance).
#[derive(Clone, Copy, PartialEq)]
pub struct HnswResult {
    pub id: u32,
    pub distance: f32,
}

impl Eq for HnswResult {}

impl PartialOrd for HnswResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HnswResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare by distance (smaller = better); break ties by id.
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl HnswBuilder {
    /// Create a new HNSW builder.
    ///
    /// - `m`: max neighbors per layer (typical: 16)
    /// - `ef_construction`: beam width during construction (typical: 200)
    /// - `dim`: embedding dimensionality
    pub fn new(m: usize, ef_construction: usize, dim: usize) -> Self {
        let m = m.max(2);
        let level_mult = 1.0 / (m as f64).ln();
        Self {
            nodes: Vec::new(),
            m,
            m_max0: 2 * m,
            ef_construction,
            entry_point: None,
            max_level: 0,
            level_mult,
            embeddings: Vec::new(),
            dim,
        }
    }

    /// Number of nodes inserted so far.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Entry point node ID.
    pub fn entry_point(&self) -> Option<u32> {
        self.entry_point
    }

    /// Maximum level in the current graph.
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Insert a new node with SQ8-quantized embedding data.
    ///
    /// Returns the assigned node ID.
    pub fn insert(&mut self, quantized: Vec<u8>, min: f32, max: f32) -> u32 {
        let node_id = self.nodes.len() as u32;
        let level = self.random_level();

        // Store the embedding.
        self.embeddings.push((quantized, min, max));

        // Create the node with empty neighbor lists.
        let node = HnswNode {
            neighbors: vec![Vec::new(); level + 1],
            level,
        };
        self.nodes.push(node);

        if self.entry_point.is_none() {
            // First node: just set as entry point.
            self.entry_point = Some(node_id);
            self.max_level = level;
            return node_id;
        }

        let ep = self.entry_point.expect("entry_point checked above");

        // Phase 1: greedily traverse from top level down to (level + 1).
        let mut current_ep = ep;
        for lc in (level + 1..=self.max_level).rev() {
            current_ep = self.greedy_closest(node_id, current_ep, lc);
        }

        // Phase 2: search and connect at levels [0..level].
        for lc in (0..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(node_id, current_ep, self.ef_construction, lc);
            let m_max = if lc == 0 { self.m_max0 } else { self.m };

            // Select best neighbors (simple: take closest M).
            let neighbors: Vec<u32> = candidates
                .iter()
                .take(m_max)
                .map(|r| r.id)
                .collect();

            // Connect node_id -> neighbors.
            self.nodes[node_id as usize].neighbors[lc] = neighbors.clone();

            // Connect neighbors -> node_id (bidirectional).
            for &neighbor_id in &neighbors {
                let n_level = self.nodes[neighbor_id as usize].level;
                if lc <= n_level {
                    let n_m_max = if lc == 0 { self.m_max0 } else { self.m };
                    let n_neighbors = &mut self.nodes[neighbor_id as usize].neighbors[lc];
                    n_neighbors.push(node_id);

                    // Prune if over capacity.
                    if n_neighbors.len() > n_m_max {
                        self.prune_neighbors(neighbor_id, lc, n_m_max);
                    }
                }
            }

            // Update entry point for next lower level.
            if !candidates.is_empty() {
                current_ep = candidates[0].id;
            }
        }

        // Update global entry point if new node has a higher level.
        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(node_id);
        }

        node_id
    }

    /// Search for the `k` nearest neighbors to a query vector (f32).
    ///
    /// Uses the specified `ef` beam width (should be >= k).
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<HnswResult> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        // Phase 1: greedily traverse from top level to level 1.
        let mut current_ep = ep;
        for lc in (1..=self.max_level).rev() {
            current_ep = self.greedy_closest_query(query, current_ep, lc);
        }

        // Phase 2: search at level 0 with beam width ef.
        let candidates = self.search_layer_query(query, current_ep, ef, 0);

        // Return top-k.
        candidates.into_iter().take(k).collect()
    }

    /// Serialize the HNSW graph to bytes for storage in index pages.
    ///
    /// Format:
    /// ```text
    /// [4 bytes] num_nodes (u32 LE)
    /// [4 bytes] max_level (u32 LE)
    /// For each node (in order):
    ///   [1 byte]  node_level (u8)
    ///   For each layer 0..=node_level:
    ///     [2 bytes] num_neighbors (u16 LE)
    ///     [4 * num_neighbors bytes] neighbor IDs (u32 LE each)
    /// ```
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Header.
        buf.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.max_level as u32).to_le_bytes());

        // Nodes.
        for node in &self.nodes {
            buf.push(node.level as u8);
            for layer_neighbors in &node.neighbors {
                buf.extend_from_slice(&(layer_neighbors.len() as u16).to_le_bytes());
                for &n in layer_neighbors {
                    buf.extend_from_slice(&n.to_le_bytes());
                }
            }
        }

        buf
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Generate a random level using exponential distribution.
    fn random_level(&self) -> usize {
        // Use a simple LCG seeded from node count for determinism in tests.
        // In production this could use thread_rng, but for reproducibility
        // in a PostgreSQL extension we use a deterministic PRNG.
        let seed = self.nodes.len() as u64;
        let r = ((seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407))
            as f64)
            / (u64::MAX as f64);
        let r = r.clamp(f64::MIN_POSITIVE, 1.0);
        let level = (-r.ln() * self.level_mult).floor() as usize;
        level.min(32) // cap at reasonable max
    }

    /// Compute squared distance between two stored nodes.
    fn distance_between(&self, a: u32, b: u32) -> f32 {
        let (ref q_a, min_a, max_a) = self.embeddings[a as usize];
        let (ref q_b, min_b, max_b) = self.embeddings[b as usize];
        // Dequantize both and compute L2 squared.
        // For build-time this is acceptable; search uses asymmetric.
        let a_f32 = crate::sq8::dequantize_sq8(q_a, min_a, max_a);
        let b_f32 = crate::sq8::dequantize_sq8(q_b, min_b, max_b);
        a_f32
            .iter()
            .zip(b_f32.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }

    /// Compute squared distance between a stored node and the node being inserted.
    fn distance_to_node(&self, query_id: u32, target_id: u32) -> f32 {
        self.distance_between(query_id, target_id)
    }

    /// Compute squared distance between a query vector (f32) and a stored node.
    fn distance_query_to_node(&self, query: &[f32], node_id: u32) -> f32 {
        let (ref stored, min, max) = self.embeddings[node_id as usize];
        asymmetric_distance_sq8_squared(query, stored, min, max)
    }

    /// Greedy search: find the closest node to `query_id` starting from `ep` at `level`.
    fn greedy_closest(&self, query_id: u32, ep: u32, level: usize) -> u32 {
        let mut current = ep;
        let mut current_dist = self.distance_to_node(query_id, current);

        loop {
            let mut changed = false;
            let neighbors = &self.nodes[current as usize].neighbors;
            if level < neighbors.len() {
                for &neighbor in &neighbors[level] {
                    let dist = self.distance_to_node(query_id, neighbor);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    /// Greedy search with f32 query vector.
    fn greedy_closest_query(&self, query: &[f32], ep: u32, level: usize) -> u32 {
        let mut current = ep;
        let mut current_dist = self.distance_query_to_node(query, current);

        loop {
            let mut changed = false;
            let neighbors = &self.nodes[current as usize].neighbors;
            if level < neighbors.len() {
                for &neighbor in &neighbors[level] {
                    let dist = self.distance_query_to_node(query, neighbor);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    /// Search at a single layer: returns candidates sorted by distance (ascending).
    fn search_layer(&self, query_id: u32, ep: u32, ef: usize, level: usize) -> Vec<HnswResult> {
        let mut visited = vec![false; self.nodes.len()];
        visited[ep as usize] = true;

        let ep_dist = self.distance_to_node(query_id, ep);

        // Min-heap for candidates (closest first).
        let mut candidates: BinaryHeap<Reverse<HnswResult>> = BinaryHeap::new();
        candidates.push(Reverse(HnswResult {
            id: ep,
            distance: ep_dist,
        }));

        // Max-heap for results (farthest first, so we can bound).
        let mut results: BinaryHeap<HnswResult> = BinaryHeap::new();
        results.push(HnswResult {
            id: ep,
            distance: ep_dist,
        });

        while let Some(Reverse(current)) = candidates.pop() {
            // If the closest candidate is farther than the farthest result, stop.
            if let Some(farthest) = results.peek() {
                if current.distance > farthest.distance && results.len() >= ef {
                    break;
                }
            }

            let neighbors = &self.nodes[current.id as usize].neighbors;
            if level >= neighbors.len() {
                continue;
            }

            for &neighbor in &neighbors[level] {
                if visited[neighbor as usize] {
                    continue;
                }
                visited[neighbor as usize] = true;

                let dist = self.distance_to_node(query_id, neighbor);

                let should_add = if results.len() < ef {
                    true
                } else if let Some(farthest) = results.peek() {
                    dist < farthest.distance
                } else {
                    true
                };

                if should_add {
                    candidates.push(Reverse(HnswResult {
                        id: neighbor,
                        distance: dist,
                    }));
                    results.push(HnswResult {
                        id: neighbor,
                        distance: dist,
                    });
                    if results.len() > ef {
                        results.pop(); // remove farthest
                    }
                }
            }
        }

        // Collect and sort by distance ascending.
        let mut result_vec: Vec<HnswResult> = results.into_vec();
        result_vec.sort();
        result_vec
    }

    /// Search at a single layer with f32 query vector.
    fn search_layer_query(
        &self,
        query: &[f32],
        ep: u32,
        ef: usize,
        level: usize,
    ) -> Vec<HnswResult> {
        let mut visited = vec![false; self.nodes.len()];
        visited[ep as usize] = true;

        let ep_dist = self.distance_query_to_node(query, ep);

        let mut candidates: BinaryHeap<Reverse<HnswResult>> = BinaryHeap::new();
        candidates.push(Reverse(HnswResult {
            id: ep,
            distance: ep_dist,
        }));

        let mut results: BinaryHeap<HnswResult> = BinaryHeap::new();
        results.push(HnswResult {
            id: ep,
            distance: ep_dist,
        });

        while let Some(Reverse(current)) = candidates.pop() {
            if let Some(farthest) = results.peek() {
                if current.distance > farthest.distance && results.len() >= ef {
                    break;
                }
            }

            let neighbors = &self.nodes[current.id as usize].neighbors;
            if level >= neighbors.len() {
                continue;
            }

            for &neighbor in &neighbors[level] {
                if visited[neighbor as usize] {
                    continue;
                }
                visited[neighbor as usize] = true;

                let dist = self.distance_query_to_node(query, neighbor);

                let should_add = if results.len() < ef {
                    true
                } else if let Some(farthest) = results.peek() {
                    dist < farthest.distance
                } else {
                    true
                };

                if should_add {
                    candidates.push(Reverse(HnswResult {
                        id: neighbor,
                        distance: dist,
                    }));
                    results.push(HnswResult {
                        id: neighbor,
                        distance: dist,
                    });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut result_vec: Vec<HnswResult> = results.into_vec();
        result_vec.sort();
        result_vec
    }

    /// Prune a node's neighbor list at the given level to at most `m_max`.
    fn prune_neighbors(&mut self, node_id: u32, level: usize, m_max: usize) {
        let neighbors = self.nodes[node_id as usize].neighbors[level].clone();
        if neighbors.len() <= m_max {
            return;
        }

        // Score all neighbors by distance to node_id and keep the closest.
        let mut scored: Vec<(u32, f32)> = neighbors
            .iter()
            .map(|&n| (n, self.distance_between(node_id, n)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(m_max);

        self.nodes[node_id as usize].neighbors[level] =
            scored.into_iter().map(|(id, _)| id).collect();
    }
}

// ---------------------------------------------------------------------------
// HnswSearcher: read-only searcher from serialized data
// ---------------------------------------------------------------------------

/// Read-only HNSW searcher that operates on deserialized graph data.
///
/// Used during index scans to find approximate nearest neighbors without
/// loading the full builder into memory.
pub struct HnswSearcher {
    /// Neighbor lists: nodes[node_id][level] = vec of neighbor IDs.
    neighbors: Vec<Vec<Vec<u32>>>,
    /// Number of nodes.
    num_nodes: usize,
    /// Maximum level.
    max_level: usize,
    /// Entry point node ID.
    entry_point: u32,
}

impl HnswSearcher {
    /// Deserialize an HNSW graph from bytes (as produced by `HnswBuilder::serialize`).
    pub fn from_bytes(data: &[u8], entry_point: u32) -> Result<Self, PgInferError> {
        if data.len() < 8 {
            return Err(PgInferError::Internal(
                "HNSW data too short for header".into(),
            ));
        }

        let num_nodes = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let max_level = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        let mut offset = 8;
        let mut neighbors = Vec::with_capacity(num_nodes);

        for _ in 0..num_nodes {
            if offset >= data.len() {
                return Err(PgInferError::Internal("HNSW data truncated".into()));
            }
            let node_level = data[offset] as usize;
            offset += 1;

            let mut node_neighbors = Vec::with_capacity(node_level + 1);
            for _ in 0..=node_level {
                if offset + 2 > data.len() {
                    return Err(PgInferError::Internal("HNSW data truncated".into()));
                }
                let num_n = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
                offset += 2;

                if offset + 4 * num_n > data.len() {
                    return Err(PgInferError::Internal("HNSW data truncated".into()));
                }
                let mut layer_neighbors = Vec::with_capacity(num_n);
                for _ in 0..num_n {
                    let id = u32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ]);
                    layer_neighbors.push(id);
                    offset += 4;
                }
                node_neighbors.push(layer_neighbors);
            }
            neighbors.push(node_neighbors);
        }

        Ok(Self {
            neighbors,
            num_nodes,
            max_level,
            entry_point,
        })
    }

    /// Search the graph for k nearest neighbors to a query.
    ///
    /// `distance_fn` computes the distance from the query to a given node ID.
    /// This allows the caller to provide the actual distance computation
    /// (asymmetric SQ8 from index pages).
    pub fn search<F>(
        &self,
        k: usize,
        ef: usize,
        distance_fn: &F,
    ) -> Vec<HnswResult>
    where
        F: Fn(u32) -> f32,
    {
        if self.num_nodes == 0 {
            return Vec::new();
        }

        // Phase 1: greedily descend from top level to level 1.
        let mut current_ep = self.entry_point;
        for lc in (1..=self.max_level).rev() {
            current_ep = self.greedy_closest(current_ep, lc, distance_fn);
        }

        // Phase 2: search at level 0.
        self.search_layer(current_ep, ef, 0, distance_fn, k)
    }

    fn greedy_closest<F>(&self, ep: u32, level: usize, distance_fn: &F) -> u32
    where
        F: Fn(u32) -> f32,
    {
        let mut current = ep;
        let mut current_dist = distance_fn(current);

        loop {
            let mut changed = false;
            if let Some(node_layers) = self.neighbors.get(current as usize) {
                if let Some(layer_neighbors) = node_layers.get(level) {
                    for &neighbor in layer_neighbors {
                        let dist = distance_fn(neighbor);
                        if dist < current_dist {
                            current = neighbor;
                            current_dist = dist;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    fn search_layer<F>(
        &self,
        ep: u32,
        ef: usize,
        level: usize,
        distance_fn: &F,
        k: usize,
    ) -> Vec<HnswResult>
    where
        F: Fn(u32) -> f32,
    {
        let mut visited = vec![false; self.num_nodes];
        visited[ep as usize] = true;

        let ep_dist = distance_fn(ep);

        let mut candidates: BinaryHeap<Reverse<HnswResult>> = BinaryHeap::new();
        candidates.push(Reverse(HnswResult {
            id: ep,
            distance: ep_dist,
        }));

        let mut results: BinaryHeap<HnswResult> = BinaryHeap::new();
        results.push(HnswResult {
            id: ep,
            distance: ep_dist,
        });

        while let Some(Reverse(current)) = candidates.pop() {
            if let Some(farthest) = results.peek() {
                if current.distance > farthest.distance && results.len() >= ef {
                    break;
                }
            }

            if let Some(node_layers) = self.neighbors.get(current.id as usize) {
                if let Some(layer_neighbors) = node_layers.get(level) {
                    for &neighbor in layer_neighbors {
                        if neighbor as usize >= self.num_nodes {
                            continue;
                        }
                        if visited[neighbor as usize] {
                            continue;
                        }
                        visited[neighbor as usize] = true;

                        let dist = distance_fn(neighbor);

                        let should_add = if results.len() < ef {
                            true
                        } else if let Some(farthest) = results.peek() {
                            dist < farthest.distance
                        } else {
                            true
                        };

                        if should_add {
                            candidates.push(Reverse(HnswResult {
                                id: neighbor,
                                distance: dist,
                            }));
                            results.push(HnswResult {
                                id: neighbor,
                                distance: dist,
                            });
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut result_vec: Vec<HnswResult> = results.into_vec();
        result_vec.sort();
        result_vec.truncate(k);
        result_vec
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sq8::quantize_sq8;

    /// Generate a random-ish f32 vector for testing.
    fn gen_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut v = Vec::with_capacity(dim);
        let mut state = seed;
        for _ in 0..dim {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            v.push(val);
        }
        v
    }

    /// Brute-force k-nearest neighbor search for ground truth.
    fn brute_force_knn(query: &[f32], embeddings: &[(Vec<u8>, f32, f32)], k: usize) -> Vec<u32> {
        let mut distances: Vec<(u32, f32)> = embeddings
            .iter()
            .enumerate()
            .map(|(i, (q, min, max))| {
                (
                    i as u32,
                    asymmetric_distance_sq8_squared(query, q, *min, *max),
                )
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.iter().take(k).map(|(id, _)| *id).collect()
    }

    #[test]
    fn test_hnsw_builder_empty() {
        let builder = HnswBuilder::new(16, 200, 32);
        assert!(builder.is_empty());
        assert_eq!(builder.len(), 0);
        assert_eq!(builder.entry_point(), None);
    }

    #[test]
    fn test_hnsw_builder_single_insert() {
        let mut builder = HnswBuilder::new(16, 200, 4);
        let vec = vec![1.0, 2.0, 3.0, 4.0];
        let (quantized, min, max) = quantize_sq8(&vec);
        let id = builder.insert(quantized, min, max);
        assert_eq!(id, 0);
        assert_eq!(builder.len(), 1);
        assert_eq!(builder.entry_point(), Some(0));
    }

    #[test]
    fn test_hnsw_search_basic() {
        let dim = 32;
        let n = 100;
        let mut builder = HnswBuilder::new(16, 50, dim);

        // Insert N vectors.
        for i in 0..n {
            let vec = gen_vector(dim, i as u64 * 17 + 42);
            let (quantized, min, max) = quantize_sq8(&vec);
            builder.insert(quantized, min, max);
        }

        assert_eq!(builder.len(), n);

        // Search for a known vector (the first one we inserted).
        let query = gen_vector(dim, 42);
        let results = builder.search(&query, 5, 50);

        assert!(!results.is_empty());
        // The first inserted vector should be among top results.
        assert!(
            results.iter().any(|r| r.id == 0),
            "expected node 0 in results: {:?}",
            results.iter().map(|r| r.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_hnsw_recall_synthetic() {
        // Test recall@10 on random data.
        let dim = 64;
        let n = 500;
        let k = 10;
        let ef = 64;
        let mut builder = HnswBuilder::new(16, 100, dim);

        let mut embeddings_for_bf = Vec::new();

        for i in 0..n {
            let vec = gen_vector(dim, i as u64 * 31 + 7);
            let (quantized, min, max) = quantize_sq8(&vec);
            embeddings_for_bf.push((quantized.clone(), min, max));
            builder.insert(quantized, min, max);
        }

        // Run multiple queries and measure average recall.
        let num_queries = 20;
        let mut total_recall = 0.0;

        for q in 0..num_queries {
            let query = gen_vector(dim, (n + q) as u64 * 53 + 13);

            // HNSW results.
            let hnsw_results: Vec<u32> = builder
                .search(&query, k, ef)
                .iter()
                .map(|r| r.id)
                .collect();

            // Brute force ground truth.
            let bf_results = brute_force_knn(&query, &embeddings_for_bf, k);

            // Recall = |intersection| / k.
            let hits = hnsw_results
                .iter()
                .filter(|id| bf_results.contains(id))
                .count();
            total_recall += hits as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        // With M=16, ef_construction=100, ef_search=64 on 500 points,
        // recall@10 should be >0.7 (typically >0.8).
        assert!(
            avg_recall > 0.6,
            "HNSW recall@{k} too low: {avg_recall:.3} (expected > 0.6)"
        );
    }

    #[test]
    fn test_hnsw_serialize_deserialize() {
        let dim = 16;
        let n = 50;
        let mut builder = HnswBuilder::new(8, 50, dim);

        let mut embeddings_for_search = Vec::new();

        for i in 0..n {
            let vec = gen_vector(dim, i as u64 * 11 + 3);
            let (quantized, min, max) = quantize_sq8(&vec);
            embeddings_for_search.push((quantized.clone(), min, max));
            builder.insert(quantized, min, max);
        }

        // Serialize.
        let data = builder.serialize();
        assert!(!data.is_empty());

        let entry_point = builder.entry_point().expect("should have entry point");

        // Deserialize.
        let searcher = HnswSearcher::from_bytes(&data, entry_point)
            .expect("deserialization should succeed");

        // Search using the searcher.
        let query = gen_vector(dim, 999);
        let distance_fn = |node_id: u32| -> f32 {
            let (ref stored, min, max) = embeddings_for_search[node_id as usize];
            asymmetric_distance_sq8_squared(&query, stored, min, max)
        };

        let results = searcher.search(5, 32, &distance_fn);
        assert!(!results.is_empty());

        // Verify results are sorted by distance.
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn test_hnsw_distance_ordering() {
        // Insert a target and verify the search finds it closest.
        let dim = 8;
        let mut builder = HnswBuilder::new(8, 50, dim);

        // Insert a target at [1,1,1,...].
        let target = vec![1.0; dim];
        let (tq, tmin, tmax) = quantize_sq8(&target);
        builder.insert(tq, tmin, tmax);

        // Insert random noise far from target.
        for i in 1..50 {
            let v = gen_vector(dim, i * 7 + 100);
            let (q, min, max) = quantize_sq8(&v);
            builder.insert(q, min, max);
        }

        // Query with target vector.
        let results = builder.search(&target, 1, 32);
        assert_eq!(results[0].id, 0, "target should be the nearest neighbor");
    }
}
