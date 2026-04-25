use pgrx::prelude::*;

use crate::error::PgLarqlError;
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
) -> Result<bool, PgLarqlError> {
    let object_lower = object.to_lowercase();
    let threshold = 10.0_f64;

    // Reuse the describe implementation to get edges for the subject.
    let edges = crate::fn_describe::describe_impl(handle, subject)?;

    // Check if any target matches the object.
    for (_relation, target, confidence, _layer) in &edges {
        if confidence >= &threshold && target.to_lowercase() == object_lower {
            return Ok(true);
        }
    }

    Ok(false)
}
