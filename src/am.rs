//! Access method handler for the `infer` virtual index.
//!
//! The `infer` AM exists purely for planner integration.  It tells the
//! planner:
//! 1. The cost of evaluating `<~>` per row (via `amcostestimate`).
//! 2. That results can be returned in distance order (`amcanorderbyop`).
//! 3. How to short-circuit with LIMIT (executor stops calling `amgettuple`).
//!
//! The index itself stores only a single metapage with the model name.
//! No vindex data is stored in PostgreSQL pages.

use pgrx::pg_sys;
use pgrx::prelude::*;

use crate::am_build;
use crate::am_cost;
use crate::am_scan;

/// AM handler function: returns the `IndexAmRoutine` describing our AM.
///
/// Registered as `infer_am_handler(internal)` via extension SQL.
#[pg_extern(sql = false)]
fn infer_am_handler(
    _fcinfo: pg_sys::FunctionCallInfo,
) -> pgrx::PgBox<pg_sys::IndexAmRoutine> {
    let mut routine = unsafe {
        pgrx::PgBox::<pg_sys::IndexAmRoutine>::alloc_node(pg_sys::NodeTag::T_IndexAmRoutine)
    };

    // Strategy and support counts.
    routine.amstrategies = 1; // one strategy: distance ordering
    routine.amsupport = 0; // no support functions beyond the handler

    // Capabilities.
    routine.amcanorder = false; // can't return in table (physical) order
    routine.amcanorderbyop = true; // CAN return in operator-distance order
    routine.amcanbackward = false;
    routine.amcanunique = false;
    routine.amcanmulticol = false;
    routine.amoptionalkey = true; // scan works without a predicate
    routine.amsearchnulls = false;
    routine.amcanparallel = false; // future work
    routine.amcaninclude = false;
    routine.amusemaintenanceworkmem = true; // v2 HNSW build benefits from maintenance_work_mem
    routine.amsummarizing = false;

    // Callbacks.
    routine.ambuild = Some(am_build::infer_ambuild);
    routine.ambuildempty = Some(am_build::infer_ambuildempty);
    routine.aminsert = Some(am_build::infer_aminsert); // v2: supports incremental insert
    routine.ambulkdelete = Some(am_build::infer_ambulkdelete);
    routine.amvacuumcleanup = Some(am_build::infer_amvacuumcleanup);
    routine.amcostestimate = Some(am_cost::infer_amcostestimate);
    routine.amoptions = Some(am_build::infer_amoptions);
    routine.ambeginscan = Some(am_scan::infer_ambeginscan);
    routine.amrescan = Some(am_scan::infer_amrescan);
    routine.amgettuple = Some(am_scan::infer_amgettuple);
    routine.amgetbitmap = None; // not supported (batch-then-iterate doesn't map to bitmap)
    routine.amendscan = Some(am_scan::infer_amendscan);
    routine.ammarkpos = None;
    routine.amrestrpos = None;

    // PG18-specific fields.
    routine.aminsertcleanup = None;
    routine.amparallelrescan = None;
    routine.amestimateparallelscan = None;
    routine.aminitparallelscan = None;

    routine.into_pg_boxed()
}

// Register operators, the access method, and operator class via extension SQL.
//
// All operator/AM registration is in ONE block to guarantee correct SQL
// ordering (pgrx can't express inter-block SQL dependencies).  Within a
// single block, statements execute sequentially:
// 1. Operators (need the underlying C functions to exist)
// 2. COST annotations
// 3. AM handler function + access method
// 4. Operator family and class (needs both operators and AM to exist)
extension_sql!(
    r#"
-- Distance operator (lower = more similar).  Used with ORDER BY + LIMIT.
CREATE OPERATOR <~> (
    LEFTARG  = text,
    RIGHTARG = text,
    FUNCTION = infer_distance,
    COMMUTATOR = <~>
);

-- Similarity score operator (higher = more similar).
-- Used with WHERE filters: WHERE col <~ 'query' > 15.0
CREATE OPERATOR <~ (
    LEFTARG  = text,
    RIGHTARG = text,
    FUNCTION = infer_similarity,
    COMMUTATOR = <~
);

-- Apply high cost to prevent the planner from using these in nested loops.
-- Without this, PG assumes COST=1 and will happily call them millions of times.
ALTER FUNCTION infer_distance(text, text) COST 10000;
ALTER FUNCTION infer_similarity(text, text) COST 10000;
ALTER FUNCTION similar_to(text, text, text) COST 10000;

-- Register the AM handler function.
CREATE FUNCTION infer_am_handler(internal) RETURNS index_am_handler
    LANGUAGE c AS 'MODULE_PATHNAME', 'infer_am_handler_wrapper';

-- Create the access method.
CREATE ACCESS METHOD infer TYPE INDEX HANDLER infer_am_handler;

-- Operator family for semantic distance ordering.
CREATE OPERATOR FAMILY infer_ops USING infer;

-- Default operator class for text columns.
CREATE OPERATOR CLASS infer_text_ops DEFAULT FOR TYPE text
    USING infer FAMILY infer_ops AS
        OPERATOR 1 <~> (text, text) FOR ORDER BY float_ops;
"#,
    name = "infer_access_method",
    requires = [infer_am_handler, infer_distance, infer_similarity, similar_to],
);
