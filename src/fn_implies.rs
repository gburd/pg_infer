use pgrx::prelude::*;

use crate::error::PgInferError;
use crate::registry;

/// Test whether the model's knowledge supports a directional relationship
/// from `subject` to `object`.
///
/// Internally runs `describe(subject)` and checks whether `object` appears
/// as a target with confidence above a threshold.
///
/// ```sql
/// SELECT implies('France', 'Paris');         -- true
/// SELECT implies('France', 'banana');        -- false
/// ```
#[pg_extern]
fn implies(
    subject: &str,
    object: &str,
    model: default!(Option<&str>, "NULL"),
) -> Result<bool, Box<dyn std::error::Error>> {
    let model_name = registry::resolve_model_name(model)?;

    let result = registry::with_model(&model_name, |handle| {
        implies_impl(handle, subject, object)
    })?;

    Ok(result)
}

fn implies_impl(
    handle: &registry::ModelHandle,
    subject: &str,
    object: &str,
) -> Result<bool, PgInferError> {
    let object_lower = object.to_lowercase();

    // Reuse the describe implementation with adaptive thresholding.
    let edges = crate::fn_describe::mmap_describe(handle, subject, None)?;

    Ok(edges
        .iter()
        .any(|e| e.target.to_lowercase() == object_lower))
}
