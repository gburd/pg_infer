//! Shared token filtering helpers.
//!
//! Matches the filtering logic in LARQL
//! `crates/larql-lql/src/executor/helpers.rs:42-102`.

/// Heuristic: is a token readable enough to show to the user?
/// Filters out encoding garbage, isolated combining marks, etc.
pub(crate) fn is_readable_token(tok: &str) -> bool {
    let tok = tok.trim();
    if tok.is_empty() || tok.len() > 30 {
        return false;
    }
    let readable = tok
        .chars()
        .filter(|c| {
            c.is_ascii_alphanumeric()
                || *c == ' '
                || *c == '-'
                || *c == '\''
                || *c == '.'
                || *c == ','
        })
        .count();
    let total = tok.chars().count();
    readable * 2 >= total && total > 0
}

/// Stricter filter: content words only.
/// Must look like a real word — no code tokens, no encoding fragments.
pub(crate) fn is_content_token(tok: &str) -> bool {
    let tok = tok.trim();
    if !is_readable_token(tok) {
        return false;
    }
    let chars: Vec<char> = tok.chars().collect();
    if chars.len() < 3 || chars.len() > 25 {
        return false;
    }
    // Must be mostly alphabetic
    let alpha = chars.iter().filter(|c| c.is_ascii_alphabetic()).count();
    if alpha < chars.len() * 2 / 3 {
        return false;
    }
    // Reject camelCase code tokens
    for w in chars.windows(2) {
        if w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase() {
            return false;
        }
    }
    // Reject if no ASCII letter (encoding fragment)
    if !chars.iter().any(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    // Filter English stop words and common function words
    let lower = tok.to_lowercase();
    !is_stop_word(&lower)
}

/// Check if a lowercased word is a stop word.
///
/// 79 English + 4 French words, matching the LARQL list exactly.
pub(crate) fn is_stop_word(lower: &str) -> bool {
    matches!(
        lower,
        "the" | "and" | "for" | "but" | "not" | "you" | "all" | "can"
        | "her" | "was" | "one" | "our" | "out" | "are" | "has" | "his"
        | "how" | "its" | "may" | "new" | "now" | "old" | "see" | "way"
        | "who" | "did" | "get" | "let" | "say" | "she" | "too" | "use"
        | "from" | "have" | "been" | "will" | "with" | "this" | "that"
        | "they" | "were" | "some" | "them" | "than" | "when"
        | "what" | "your" | "each" | "make" | "like" | "just" | "over"
        | "such" | "take" | "also" | "into" | "only" | "very" | "more"
        | "does" | "most" | "about" | "which" | "their" | "would" | "there"
        | "could" | "other" | "after" | "being" | "where" | "these" | "those"
        | "first" | "should" | "because" | "through" | "before"
        | "par" | "aux" | "che" | "del"
    )
}
