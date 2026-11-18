//! Relation classification by a System-One typed-decision model.
//!
//! `describe()` returns edges, and every edge needs a `relation` label.  The
//! fallback in [`crate::relation_classify`] decides that label from the
//! target's capitalisation, two hardcoded gate-score thresholds, and a
//! twenty-word English geography list.  That is English-only by construction
//! and the thresholds are tuned to Gemma-class gate magnitudes, so on other
//! models the labels degrade while the scores stay correct.
//!
//! This module asks a model instead.  A Laya-style server
//! (<https://pglaya.com>, protocol `/v1/systemone`) takes a *state* plus typed
//! *questions* and answers them in a single forward pass with calibrated
//! probabilities over a closed label set.  That is exactly the shape of the
//! problem here: the label set is fixed, and the evidence (target token, its
//! coherent secondaries, where in the network the edge was found) is a short
//! piece of prose.
//!
//! # What this costs, measured
//!
//! Measured on an EC2 g6.8xlarge against `laya-typed-decisions` (421M,
//! ModernBERT-large), labelling the 20 edges of one `describe()`:
//!
//! | host | 1 edge | 20 edges | vs a 15 ms `describe()` |
//! |---|---|---|---|
//! | 32-vCPU CPU, fp32 | 191 ms | 1833 ms | 122x the whole query |
//! | L4 GPU, bf16 | 20 ms | 67 ms | 4.5x the whole query |
//!
//! Batching works -- 20 edges cost 3.4x one edge on GPU, not 20x -- because
//! every option for one edge is scored at its own `[MASK]` marker in a single
//! sequence.  But the floor is a 421M-parameter encoder forward pass, and
//! `describe()` is a 13--18 ms query, so even the GPU path multiplies latency
//! by 4.5x and the CPU path by 122x.
//!
//! That is why this is not the default and is not wired into `describe()`.
//! It is worth having where a caller wants a calibrated label badly enough to
//! pay for it, and it is worth keeping out of the hot path everywhere else.
//!
//! NB: callers get the heuristic labels unchanged when no server is
//! configured or reachable.  Classification must never be the reason a query
//! fails, so every error path here falls back rather than propagating.

// Nothing here is called yet, and that is the finding rather than an
// oversight: the measured cost above rules it out of `describe()`'s hot
// path, so it stays a working, tested prototype instead of being wired in
// behind a switch nobody should turn on.  Drop this when a caller lands.
#![allow(dead_code)]

use std::time::Duration;

use infer_client::transport::{CancelToken, CancellableClient};

/// The closed label set.  Kept identical to the heuristic's vocabulary so the
/// two are interchangeable and a caller cannot tell which produced a row
/// except by the confidence column.
pub(crate) const RELATIONS: &[&str] = &[
    "related_to",
    "part_of",
    "instance_of",
    "located_in",
    "has_property",
    "similar_to",
];

/// One edge's worth of evidence, phrased for an encoder.
pub(crate) struct Evidence<'a> {
    pub entity: &'a str,
    pub target: &'a str,
    pub secondaries: &'a [&'a str],
    pub layer: i32,
    pub num_layers: usize,
}

/// A label plus how much the model believes it.
pub(crate) struct Verdict {
    pub relation: String,
    pub confidence: f64,
}

/// Describe where in the network an edge was found, in words rather than a
/// layer index.
///
/// The band carries the real signal (early layers encode syntax, middle
/// layers hold factual associations, late layers shape the output
/// distribution), and a bare integer means nothing to a text encoder.  This
/// mirrors the band split the heuristic already uses.
fn band_phrase(layer: i32, num_layers: usize) -> &'static str {
    if num_layers == 0 {
        return "an unknown part of the network";
    }
    let band = num_layers / 3;
    let layer = layer.max(0) as usize;
    if layer < band {
        "an early layer, where the model handles grammar and word form"
    } else if layer >= num_layers.saturating_sub(band) {
        "a late layer, where the model is choosing its next word"
    } else {
        "a middle layer, where the model stores facts about the world"
    }
}

/// Turn one edge into the prose state the encoder scores.
///
/// Deliberately prose, not JSON.  Laya's own guidance is that handing the
/// encoder a struct gives it a document with almost no meaning in it, and that
/// naming the situation the way a person would is worth more than the choice
/// of checkpoint.
fn state_for(ev: &Evidence<'_>) -> String {
    let mut s = format!(
        "A language model was asked about \"{}\". Inside {}, it strongly \
         associated \"{}\" with \"{}\".",
        ev.entity,
        band_phrase(ev.layer, ev.num_layers),
        ev.entity,
        ev.target
    );
    if !ev.secondaries.is_empty() {
        s.push_str(&format!(
            " Words that came up alongside \"{}\" were: {}.",
            ev.target,
            ev.secondaries.join(", ")
        ));
    }
    s
}

/// The question, with options as phrases rather than enum tokens.
///
/// Laya's README reports that replacing bare tokens with descriptive phrases
/// took their pong demo from 3/5 to 5/5, so the options read as English and
/// are mapped back to the bare label by position.
fn question_options() -> Vec<&'static str> {
    vec![
        "the two things are loosely related",
        "the second thing is a part or region of the first",
        "the first thing is an example or instance of the second",
        "the first thing is located in the second",
        "the second thing is a property or attribute of the first",
        "the two things are similar kinds of thing",
    ]
}

/// Classify a whole batch of edges in ONE request.
///
/// Returns `None` when the server is unset, unreachable, or answers in a
/// shape we do not recognise; the caller then keeps its heuristic labels.  On
/// success the returned vector is positionally aligned with `batch`.
pub(crate) fn classify_batch(
    server_url: &str,
    batch: &[Evidence<'_>],
    timeout: Duration,
) -> Option<Vec<Verdict>> {
    if server_url.is_empty() || batch.is_empty() {
        return None;
    }

    let opts = question_options();
    // One state per edge, each with a single choice question. `laya` cannot
    // reliably separate rows packed into a shared state (its own docs say so,
    // hence `state_mode = native`), so the batching that matters is at the
    // request level, not inside one state.
    let items: Vec<serde_json::Value> = batch
        .iter()
        .map(|ev| {
            serde_json::json!({
                "state": state_for(ev),
                "questions": {
                    "relation": {
                        "type": "choice",
                        "question": "How are the two things related?",
                        "criteria": opts,
                    }
                }
            })
        })
        .collect();

    let client = CancellableClient::connect(server_url, timeout).ok()?;
    let cancel = CancelToken::new();
    let resp: serde_json::Value = client
        .post_json(
            "/v1/systemone/batch",
            serde_json::json!({ "items": items }),
            &cancel,
        )
        .ok()?;

    let answers = resp
        .get("results")
        .or_else(|| resp.get("items"))?
        .as_array()?;
    if answers.len() != batch.len() {
        // A partial answer cannot be aligned positionally, and guessing the
        // alignment would mislabel edges silently.
        return None;
    }

    let mut out = Vec::with_capacity(answers.len());
    for a in answers {
        let ans = a.pointer("/answers/relation")?;
        let chosen = ans.get("choice")?.as_str()?;
        // Map the phrase back to the bare label by position.
        let idx = opts.iter().position(|o| *o == chosen)?;
        out.push(Verdict {
            relation: RELATIONS[idx].to_string(),
            confidence: ans
                .get("confidence")
                .and_then(|c| c.as_f64())
                .unwrap_or(f64::NAN),
        });
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_phrase_splits_into_three() {
        // 30 layers -> bands of 10.
        assert!(band_phrase(0, 30).contains("grammar"));
        assert!(band_phrase(15, 30).contains("facts"));
        assert!(band_phrase(29, 30).contains("next word"));
    }

    #[test]
    fn band_phrase_survives_degenerate_input() {
        assert!(band_phrase(0, 0).contains("unknown"));
        // A layer past the end must not panic or index out of range.
        assert!(band_phrase(99, 3).contains("next word"));
        assert!(band_phrase(-1, 30).contains("grammar"));
    }

    #[test]
    fn state_reads_as_prose_and_carries_the_evidence() {
        let secs = ["city", "capital"];
        let ev = Evidence {
            entity: "France",
            target: "Paris",
            secondaries: &secs,
            layer: 15,
            num_layers: 30,
        };
        let s = state_for(&ev);
        assert!(s.contains("France") && s.contains("Paris"));
        assert!(s.contains("city, capital"));
        assert!(s.contains("facts about the world"));
        // Prose, not a serialised struct.  The colon in "were: city, capital"
        // is prose punctuation, so only brace/bracket syntax is disqualifying.
        assert!(!s.contains('{') && !s.contains('[') && !s.contains("\":\""));
    }

    #[test]
    fn state_omits_the_clause_when_there_are_no_secondaries() {
        let ev = Evidence {
            entity: "France",
            target: "Paris",
            secondaries: &[],
            layer: 15,
            num_layers: 30,
        };
        assert!(!state_for(&ev).contains("came up alongside"));
    }

    #[test]
    fn options_align_with_the_label_set() {
        // The mapping back from phrase to label is positional, so a drift
        // between these two lists would silently mislabel every edge.
        assert_eq!(question_options().len(), RELATIONS.len());
    }

    #[test]
    fn empty_input_and_unset_server_fall_back() {
        assert!(classify_batch("", &[], Duration::from_millis(1)).is_none());
        let ev = Evidence {
            entity: "a",
            target: "b",
            secondaries: &[],
            layer: 0,
            num_layers: 1,
        };
        assert!(classify_batch("", std::slice::from_ref(&ev), Duration::from_millis(1)).is_none());
    }
}
