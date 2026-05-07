use pgrx::prelude::*;

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

    let result = registry::with_backend(&model_name, |backend| {
        backend.implies(subject, object)
    })?;

    Ok(result)
}
