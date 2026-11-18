use pgrx::prelude::*;

use crate::registry;

/// Test whether the model's knowledge supports a directional relationship
/// from `subject` to `object`.
///
/// Runs `describe(subject)` and reports whether `object` appears among the
/// targets, matched case-insensitively.  Note there is no confidence test
/// here: the filtering has already happened inside `describe()`, via the gate
/// threshold and its coherence check, so this is a membership question and
/// not a comparison against a score.
///
/// ```sql
/// SELECT implies('France', 'Paris');         -- true
/// SELECT implies('France', 'banana');        -- false
/// ```
#[pg_extern(stable, parallel_safe)]
#[tracing::instrument(skip_all, fields(subject = subject, object = object, model = model.unwrap_or("default")))]
fn implies(
    subject: &str,
    object: &str,
    model: default!(Option<&str>, "NULL"),
) -> Result<bool, Box<dyn std::error::Error>> {
    let model_name = registry::resolve_model_name(model)?;

    let result = registry::with_backend(&model_name, |backend| backend.implies(subject, object))?;

    Ok(result)
}

/// Operator function for `@>` (semantic implication).
///
/// `'France' @> 'Paris'` returns true if the model's knowledge supports
/// a directional relationship from left to right.  Uses the default model.
#[pg_extern]
fn infer_implies_op(left: &str, right: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let model_name = registry::resolve_model_name(None)?;
    let result = registry::with_backend(&model_name, |backend| backend.implies(left, right))?;
    Ok(result)
}

// Register the @> operator for semantic implication.
extension_sql!(
    r#"
CREATE OPERATOR @> (
    LEFTARG  = text,
    RIGHTARG = text,
    FUNCTION = infer_implies_op
);
"#,
    name = "infer_implies_operator",
    requires = [infer_implies_op],
);
