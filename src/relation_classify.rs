/// Heuristic relation classifier based on layer position and token properties.
///
/// Uses layer bands (syntax / knowledge / output) combined with target token
/// characteristics (capitalization, geographic context) and gate activation
/// strength to assign a relation label.
pub(crate) fn classify_relation(
    layer: usize,
    num_layers: usize,
    target: &str,
    gate_score: f32,
    secondaries: &[&str],
) -> &'static str {
    // Guard against zero-layer models (shouldn't happen, but be safe).
    if num_layers == 0 {
        return "related_to";
    }

    let band_size = num_layers / 3;
    let is_syntax_band = layer < band_size;
    let is_output_band = layer >= num_layers - band_size;
    // Knowledge band is everything in the middle.
    let is_knowledge_band = !is_syntax_band && !is_output_band;

    let target_capitalized = target
        .chars()
        .next()
        .map(|c| c.is_uppercase())
        .unwrap_or(false);

    if is_syntax_band {
        return "related_to";
    }

    if is_output_band {
        return "similar_to";
    }

    // Knowledge band logic.
    debug_assert!(is_knowledge_band);

    // High gate scores with capitalized targets suggest structural relationships.
    if gate_score > 30.0 && target_capitalized {
        return "part_of";
    }

    if target_capitalized {
        if has_geographic_term(secondaries) {
            return "located_in";
        }
        return "instance_of";
    }

    // Non-capitalized targets in knowledge band.
    if gate_score > 20.0 {
        return "has_property";
    }

    "related_to"
}

/// Check if any secondary tokens suggest a geographic context.
fn has_geographic_term(secondaries: &[&str]) -> bool {
    const GEO_HINTS: &[&str] = &[
        "city", "country", "state", "region", "continent", "capital",
        "province", "territory", "island", "ocean", "sea", "river",
        "mountain", "lake", "north", "south", "east", "west",
        "europe", "asia", "africa", "america",
    ];

    for secondary in secondaries {
        let lower = secondary.to_lowercase();
        for hint in GEO_HINTS {
            if lower.contains(hint) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntax_band() {
        assert_eq!(classify_relation(0, 30, "test", 10.0, &[]), "related_to");
        assert_eq!(classify_relation(5, 30, "Paris", 50.0, &["city"]), "related_to");
    }

    #[test]
    fn test_output_band() {
        assert_eq!(classify_relation(25, 30, "dog", 10.0, &[]), "similar_to");
        assert_eq!(classify_relation(29, 30, "Paris", 50.0, &["city"]), "similar_to");
    }

    #[test]
    fn test_knowledge_band_high_score_capitalized() {
        assert_eq!(classify_relation(15, 30, "Europe", 35.0, &["continent"]), "part_of");
    }

    #[test]
    fn test_knowledge_band_geographic() {
        assert_eq!(classify_relation(15, 30, "Paris", 10.0, &["city", "capital"]), "located_in");
    }

    #[test]
    fn test_knowledge_band_instance_of() {
        assert_eq!(classify_relation(15, 30, "Einstein", 10.0, &["physicist"]), "instance_of");
    }

    #[test]
    fn test_knowledge_band_has_property() {
        assert_eq!(classify_relation(15, 30, "tall", 25.0, &["height"]), "has_property");
    }

    #[test]
    fn test_knowledge_band_default() {
        assert_eq!(classify_relation(15, 30, "warm", 5.0, &[]), "related_to");
    }

    #[test]
    fn test_zero_layers() {
        assert_eq!(classify_relation(0, 0, "test", 10.0, &[]), "related_to");
    }
}
