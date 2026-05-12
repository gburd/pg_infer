//! Adaptive cost estimation for the `infer` access method.
//!
//! The cost model is dynamic: derived from model/backend characteristics
//! queried at plan time, not hardcoded constants.  This ensures the planner
//! makes correct decisions about when to use the infer AM vs seq scan + sort.

use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::am_options;

/// Cost estimation callback for the `infer` AM.
///
/// Provides the planner with accurate per-row costs so it can choose
/// between Index Scan (using our AM) and SeqScan + Sort.  The key
/// insight: `amcanorderbyop = true` means we return results in distance
/// order, so the executor can stop at LIMIT without sorting.
///
/// # Safety
///
/// Called by the PostgreSQL planner.  All pointer arguments are valid
/// for the duration of this call.
#[allow(clippy::too_many_arguments)]
#[pg_guard]
pub unsafe extern "C-unwind" fn infer_amcostestimate(
    _root: *mut pg_sys::PlannerInfo,
    path: *mut pg_sys::IndexPath,
    _loop_count: f64,
    index_startup_cost: *mut pg_sys::Cost,
    index_total_cost: *mut pg_sys::Cost,
    index_selectivity: *mut pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    let index_info = (*path).indexinfo;
    let tuples = (*index_info).tuples;

    // Read model name from the index metapage to determine cost params.
    let cost_params = match read_cost_params(index_info) {
        Some(p) => p,
        None => {
            // Conservative fallback: assume expensive local backend.
            CostParams {
                embed_cost: 100.0,
                per_row_cost: 50.0,
            }
        }
    };

    // Startup cost: embed the query (one tokenize + one embed lookup).
    let startup = cost_params.embed_cost;

    // Total cost scales with number of tuples.
    // The planner will compare this against SeqScan + Sort costs.
    let total = startup + (tuples * cost_params.per_row_cost);

    *index_startup_cost = startup;
    *index_total_cost = total;
    *index_selectivity = 1.0; // scans all rows; LIMIT prunes at executor
    *index_correlation = 0.0; // output in distance order, not physical order
    *index_pages = 1.0; // just the metapage
}

/// Cost parameters derived from model metadata at plan time.
struct CostParams {
    /// One-time query embedding cost (PG cost units).
    embed_cost: f64,
    /// Marginal cost per candidate row (PG cost units).
    per_row_cost: f64,
}

/// Read cost parameters by looking up the model from the index's metapage
/// and querying its characteristics from `infer.models`.
unsafe fn read_cost_params(index_info: *mut pg_sys::IndexOptInfo) -> Option<CostParams> {
    // Read model name from the index relation's metapage.
    let index_rel = pg_sys::RelationIdGetRelation((*index_info).indexoid);
    if index_rel.is_null() {
        return None;
    }

    let model_name = am_options::read_model_from_metapage(index_rel).ok();
    pg_sys::RelationClose(index_rel);

    let model_name = model_name?;

    // Query model metadata from infer.models via SPI.
    let meta = query_model_metadata(&model_name)?;

    // Layers actually queried (respects similarity_max_layers GUC).
    let max_layers = crate::gucs::similarity_max_layers();
    let layers_queried = if max_layers > 0 && (meta.num_layers as usize) > max_layers {
        max_layers as f64
    } else {
        meta.num_layers as f64
    };

    // Model-size factor: normalized to Qwen 0.5B (24 layers, 4864 hidden).
    let hidden_size = meta.hidden_size as f64;
    let size_factor = (hidden_size / 4864.0) * (layers_queried / 24.0);

    let is_remote = meta.backend == "remote";

    if is_remote {
        // Remote backend: batch-amortized cost.
        // Base: ~15ms/row cached for 0.5B, scales sub-linearly.
        // PG cost units: 1 unit ~ cpu_tuple_cost ~ 0.01ms.
        let per_row_ms = 15.0 * size_factor.sqrt();
        let per_row_cost = per_row_ms / 0.01;
        // Batch amortization: sending N candidates in one POST is ~50x
        // cheaper per row than individual calls.
        let amortized = per_row_cost / 50.0;
        Some(CostParams {
            embed_cost: 50.0,
            per_row_cost: amortized,
        })
    } else {
        // Local backend: per-row gate-KNN is the bottleneck.
        // Base: ~180ms/row for 0.5B.
        let per_row_ms = 180.0 * size_factor;
        let per_row_cost = per_row_ms / 0.01;
        Some(CostParams {
            embed_cost: 100.0,
            per_row_cost,
        })
    }
}

/// Model metadata fetched from `infer.models` at plan time.
struct ModelMeta {
    backend: String,
    num_layers: i32,
    hidden_size: i32,
}

/// Query model metadata from `infer.models` via SPI.
///
/// Returns `None` if the model doesn't exist or SPI fails (plan-time
/// safety: we never error during cost estimation).
fn query_model_metadata(model_name: &str) -> Option<ModelMeta> {
    Spi::connect(|client| {
        let mut result = client
            .select(
                "SELECT backend, num_layers, hidden_size \
                 FROM infer.models WHERE model_name = $1",
                Some(1),
                &[pgrx::datum::DatumWithOid::from(model_name)],
            )
            .ok()?;
        if let Some(row) = result.next() {
            let backend: String = row.get(1).ok()?.unwrap_or_else(|| "local".to_string());
            let num_layers: i32 = row.get(2).ok()?.unwrap_or(24);
            let hidden_size: i32 = row.get(3).ok()?.unwrap_or(4864);
            return Some(ModelMeta {
                backend,
                num_layers,
                hidden_size,
            });
        }
        None
    })
}
