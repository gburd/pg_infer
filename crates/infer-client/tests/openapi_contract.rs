//! Wire-contract test: pg_infer's DTOs vs larql-server's own OpenAPI spec.
//!
//! ## Why this exists
//!
//! `mock_server_integration.rs` asserts that pg_infer can parse JSON that
//! `mock_server_integration.rs` wrote.  Both halves are ours, so the pair
//! agrees by construction and a divergence from the *real* server is
//! invisible.  Three wire bugs shipped behind that green test, two of which
//! had never worked against a real server at any point:
//!
//!   * `/v1/relations` — `RelationSummary::token` is required, the server
//!     emits `name`.  Missing required field ⇒ every `show_relations()`
//!     against a real larql-server errors.
//!   * `/v1/models` — pg_infer expects `{"models": [...]}`, the server
//!     emits the OpenAI envelope `{"object": "list", "data": [...]}`.
//!   * `/v1/warmup` — every field pg_infer reads is `#[serde(default)]` and
//!     none of them exist on the server's response, so warmup silently
//!     reported "0 warmed, 0 already cached" forever.
//!
//! So this test does not hand-write any JSON.  It reads the schemas from
//! `fixtures/larql-openapi.json`, which is generated *by the server from
//! its own handlers* (utoipa `ApiDoc::openapi()`, served live at
//! `GET /v1/openapi.json`), and checks that:
//!
//!   1. every field the server marks `required` is a field our DTO can
//!      actually accept — catches a rename like `token`/`name`; and
//!   2. a synthetic response built to the schema's shape (required fields
//!      only, no extras) deserializes into our DTO — catches a required
//!      field of ours that the server never sends.
//!
//! Check (2) is the load-bearing one: it fails exactly when our DTO
//! demands something the server does not promise.
//!
//! ## Refreshing the fixture
//!
//! The fixture is the contract, so it is committed rather than fetched at
//! test time (offline, hermetic, and it diffs in review).  To re-pin
//! against a newer larql-server:
//!
//! ```sh
//! larql-server <vindex> &                      # or any profile
//! curl -s localhost:8080/v1/openapi.json \
//!   | jq . > crates/infer-client/tests/fixtures/larql-openapi.json
//! ```
//!
//! A field this test flags is a real incompatibility: either the server
//! changed, or we were wrong.  Fix the DTO — do not silence the check.

use infer_client::{
    DescribeResponse, InferResponse, RelationsResponse, StatsResponse, WarmupResponse,
};
use serde_json::{json, Map, Value};

const SPEC: &str = include_str!("fixtures/larql-openapi.json");

fn spec() -> Value {
    serde_json::from_str(SPEC).expect("fixture is valid JSON")
}

/// Resolve a `#/components/schemas/X` reference one level.
fn resolve<'a>(spec: &'a Value, schema: &'a Value) -> &'a Value {
    match schema.get("$ref").and_then(Value::as_str) {
        Some(r) => {
            let name = r.rsplit('/').next().unwrap_or_default();
            spec.pointer(&format!("/components/schemas/{name}"))
                .unwrap_or(schema)
        }
        None => schema,
    }
}

fn schema_of<'a>(spec: &'a Value, name: &str) -> &'a Value {
    spec.pointer(&format!("/components/schemas/{name}"))
        .unwrap_or_else(|| {
            unreachable!("schema {name} missing from spec — did the server drop it?")
        })
}

/// A value satisfying `schema`'s declared type, for building a synthetic
/// response.  Only the shape matters; the contents are never inspected.
fn sample(spec: &Value, schema: &Value) -> Value {
    let schema = resolve(spec, schema);
    // Nullable unions (`type: ["string", "null"]`) and `anyOf` wrappers:
    // take the first concrete arm.
    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array) {
        if let Some(first) = any_of
            .iter()
            .find(|s| s.get("type").and_then(Value::as_str) != Some("null"))
        {
            return sample(spec, first);
        }
    }
    let ty = match schema.get("type") {
        Some(Value::String(s)) => s.as_str(),
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(Value::as_str)
            .find(|s| *s != "null")
            .unwrap_or("object"),
        _ => "object",
    };
    match ty {
        "string" => json!("x"),
        "integer" => json!(1),
        "number" => json!(1.0),
        "boolean" => json!(true),
        "array" => {
            // Fixed-length tuples (LayerBands' `[usize; 2]`) declare
            // minItems; honour it or serde rejects the arity.
            let n = schema.get("minItems").and_then(Value::as_u64).unwrap_or(1) as usize;
            let item = schema
                .get("items")
                .map(|i| sample(spec, i))
                .unwrap_or(json!(1));
            Value::Array(std::iter::repeat_n(item, n.max(1)).collect())
        }
        _ => required_only(spec, schema),
    }
}

/// Build the *minimal* response the server promises: required fields only.
///
/// Deliberately omits optional fields.  A DTO that only parses when an
/// optional field happens to be present is broken against a server that
/// legitimately omits it, and this is what surfaces that.
fn required_only(spec: &Value, schema: &Value) -> Value {
    let schema = resolve(spec, schema);
    let props = schema.get("properties").and_then(Value::as_object);
    let required: Vec<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut out = Map::new();
    if let Some(props) = props {
        for name in required {
            if let Some(ps) = props.get(name) {
                out.insert(name.to_string(), sample(spec, ps));
            }
        }
    }
    Value::Object(out)
}

/// Does `T` actually *read* `field`?  Determined by giving `field` a value
/// of the wrong type: if `T` binds the field, serde reports a type error;
/// if `T` ignores it, deserialization succeeds.
fn reads_field<T: serde::de::DeserializeOwned>(base: &Value, field: &str) -> bool {
    let mut probe = base.clone();
    if let Some(obj) = probe.as_object_mut() {
        // An object is the wrong type for every scalar field, and a
        // string is the wrong type for object/array fields; try both.
        obj.insert(field.to_string(), json!({ "__wrong_type": true }));
    }
    if serde_json::from_value::<T>(probe).is_err() {
        return true;
    }
    let mut probe = base.clone();
    if let Some(obj) = probe.as_object_mut() {
        obj.insert(field.to_string(), json!("__wrong_type"));
    }
    serde_json::from_value::<T>(probe).is_err()
}

/// The core assertion: the minimal server-promised response for `schema`
/// must deserialize into `T`.
fn assert_parses<T: serde::de::DeserializeOwned>(schema_name: &str) {
    let spec = spec();
    let body = required_only(&spec, schema_of(&spec, schema_name));
    let err = match serde_json::from_value::<T>(body.clone()) {
        Ok(_) => return,
        Err(e) => e,
    };
    // `assert!(cond, msg)` rather than `panic!(msg)`: this crate denies
    // `clippy::panic`, and a bare panic in a test file trips it.
    assert!(
        err.to_string().is_empty(),
        "{} does not deserialize the minimal response larql-server \
         promises for `{schema_name}`.\n  serde error: {err}\n  \
         server-required body: {}\n\nEither the server changed or the \
         DTO is wrong. Fix the DTO; do not relax this test.",
        std::any::type_name::<T>(),
        serde_json::to_string_pretty(&body).unwrap_or_default(),
    );
}

// ── The endpoints pg_infer actually calls ────────────────────────────────

#[test]
fn stats_matches_server_schema() {
    assert_parses::<StatsResponse>("StatsResponse");
}

#[test]
fn describe_matches_server_schema() {
    assert_parses::<DescribeResponse>("DescribeResponse");
}

#[test]
fn infer_matches_server_schema() {
    assert_parses::<InferResponse>("InferResponse");
}

#[test]
fn relations_matches_server_schema() {
    // Regression: `RelationSummary` required `token`; the server emits
    // `name`.  This is the check that was missing.
    assert_parses::<RelationsResponse>("RelationsResponse");
}

#[test]
fn warmup_matches_server_schema() {
    assert_parses::<WarmupResponse>("WarmupResponse");
}

/// `/v1/warmup` — parsing is not the same as reading.
///
/// The original `WarmupResponse` was `{warmed, already_cached,
/// latency_ms}`, all `#[serde(default)]` and none of them fields the
/// server sends: the schema-parse test above passed while every value
/// came out zero. Feed a fully-populated *real* response and assert the
/// counters actually arrive — that is what separates "we accepted the
/// bytes" from "we understood them".
#[test]
fn warmup_response_is_not_silently_zero() {
    let spec = spec();
    let schema = schema_of(&spec, "WarmupResponse");
    // Populate *every* property the server declares (not only the
    // required ones) with a distinctive non-zero value.
    let mut body = Map::new();
    if let Some(props) = schema.get("properties").and_then(Value::as_object) {
        for (name, ps) in props {
            let v = match resolve(&spec, ps).get("type").and_then(Value::as_str) {
                Some("integer") => json!(42),
                Some("number") => json!(42.0),
                Some("boolean") => json!(true),
                Some("string") => json!("m"),
                _ => sample(&spec, ps),
            };
            body.insert(name.clone(), v);
        }
    }
    let body = Value::Object(body);
    let resp: WarmupResponse = serde_json::from_value(body.clone())
        .expect("a fully-populated server response must parse");
    let read_anything = resp.layers_prefetched != 0
        || resp.experts_prefetched != 0
        || resp.total_ms != 0
        || resp.weights_load_ms != 0
        || resp.prefetch_ms != 0;
    assert!(
        read_anything,
        "WarmupResponse read a fully-populated larql-server response as all \
         zeros, so infer_warmup() reports no work regardless of what the \
         server did. Every field must be one the server actually sends.\n  \
         server body: {}",
        serde_json::to_string_pretty(&body).unwrap_or_default(),
    );
    assert_eq!(
        resp.layers_prefetched, 42,
        "`layers_prefetched` must come from the server's field of that name"
    );
    assert!(
        resp.weights_loaded,
        "`weights_loaded` must be read, not defaulted to false"
    );
}

/// `/v1/relations` field-level contract.
///
/// Parsing the minimal body proves we do not *reject* the response;
/// it does not prove we read the data, because every field on
/// `RelationSummary` could be `#[serde(default)]` and yield an empty
/// summary from a populated response.  So assert the entry fields are
/// actually bound.
#[test]
fn relation_entry_fields_are_read_not_defaulted() {
    let spec = spec();
    let entry = required_only(&spec, schema_of(&spec, "RelationEntry"));
    let base = json!({ "relations": [entry], "total": 1, "latency_ms": 1.0 });
    // Sanity: the shape parses at all.
    serde_json::from_value::<RelationsResponse>(base.clone())
        .expect("RelationsResponse must parse a server-shaped body");

    let resp: RelationsResponse = serde_json::from_value(base).expect("parsed above");
    let first = resp
        .relations
        .first()
        .expect("server sent one relation, so we must surface one");
    assert_eq!(
        first.token, "x",
        "the relation name from the server must reach `RelationSummary::token` \
         (the server calls this field `name`; a serde rename/alias is required)"
    );
    assert_eq!(first.count, 1, "count must be read");
    assert!(
        !first.layers.is_empty(),
        "the server reports the layer span as `min_layer`/`max_layer`; \
         `layers` must be derived from them rather than silently empty"
    );
}

/// `/v1/models` — grid discovery.
///
/// Kept as a raw-JSON check because `grid.rs`'s `ModelsResponse` is
/// private to the extension crate; this asserts the envelope the server
/// actually emits, which is the OpenAI list shape, not `{"models": [...]}`.
#[test]
fn models_envelope_is_openai_shaped() {
    let spec = spec();
    let list = schema_of(&spec, "ModelsListResponse");
    let props: Vec<&str> = list
        .get("properties")
        .and_then(Value::as_object)
        .map(|o| o.keys().map(String::as_str).collect())
        .unwrap_or_default();
    assert!(
        props.contains(&"data"),
        "/v1/models is an OpenAI-shaped list: {{object, data}}. \
         pg_infer's grid discovery reads `models`, which the server never \
         sends — with #[serde(default)] that yields an empty server list \
         and the misleading error \"no servers for this model\". \
         Server properties: {props:?}"
    );
    assert!(
        !props.contains(&"models"),
        "server unexpectedly grew a `models` key; re-check grid.rs"
    );

    // And the per-entry shape carries no URL: the router is a proxy, not a
    // shard directory, so client-side round-robin has nothing to route on.
    let entry = schema_of(&spec, "ModelEntry");
    let eprops: Vec<&str> = entry
        .get("properties")
        .and_then(Value::as_object)
        .map(|o| o.keys().map(String::as_str).collect())
        .unwrap_or_default();
    assert!(
        !eprops.contains(&"url") && !eprops.contains(&"server"),
        "a per-model URL appeared in /v1/models ({eprops:?}); grid.rs could \
         resolve shards from discovery again"
    );
}

/// `/v1/warmup` — the request body pg_infer sends must be a body the
/// server accepts.  The old `{entities: [...]}` shape is not in the
/// schema at all.
#[test]
fn warmup_request_shape_is_layers_not_entities() {
    let spec = spec();
    let req = schema_of(&spec, "WarmupRequest");
    let props: Vec<&str> = req
        .get("properties")
        .and_then(Value::as_object)
        .map(|o| o.keys().map(String::as_str).collect())
        .unwrap_or_default();
    assert!(
        !props.contains(&"entities"),
        "warmup still takes `entities`? Server properties: {props:?}"
    );
    assert!(
        props.contains(&"layers"),
        "warmup takes `layers` + `skip_weights`; pg_infer posts \
         `{{entities: [...]}}`, which the server ignores entirely. \
         Server properties: {props:?}"
    );
}

/// Endpoints pg_infer calls that larql-server has never served.
///
/// Not a bug in itself — each is behind a 404 fallback — but the
/// compatibility docs described them as part of the `/v1` contract, and
/// they are not.  This pins that they are pg_infer-local so the claim
/// cannot quietly become false in either direction.
#[test]
fn pg_infer_local_endpoints_are_absent_upstream() {
    let spec = spec();
    let paths = spec
        .get("paths")
        .and_then(Value::as_object)
        .expect("spec has paths");
    for p in ["/v1/cache/stats", "/v1/rank", "/v1/features", "/v1/layers"] {
        assert!(
            !paths.contains_key(p),
            "{p} now exists upstream — drop pg_infer's 404 fallback and use it"
        );
    }
    // The replacement for /v1/cache/stats: server counters on /v1/stats.
    let stats = schema_of(&spec, "StatsResponse");
    let has_server_block = stats
        .get("properties")
        .and_then(Value::as_object)
        .map(|o| o.contains_key("server"))
        .unwrap_or(false);
    assert!(
        has_server_block,
        "/v1/stats lost its `server` block; infer_cache_stats() has no source"
    );
}

/// `/v1/capabilities` — lets pg_infer ask what a server supports instead
/// of probing for 404s.  Pinned so the handshake work has a fixed target.
#[test]
fn capabilities_endpoint_is_available() {
    let spec = spec();
    let paths = spec
        .get("paths")
        .and_then(Value::as_object)
        .expect("spec has paths");
    assert!(
        paths.contains_key("/v1/capabilities"),
        "no /v1/capabilities: pg_infer must keep probing endpoints to \
         discover support"
    );
}

#[test]
fn fixture_is_the_shape_we_think_it_is() {
    // Guard against a truncated or hand-edited fixture silently making
    // every assertion above vacuous.
    let spec = spec();
    assert_eq!(
        spec.get("openapi").and_then(Value::as_str),
        Some("3.1.0"),
        "fixture is not an OpenAPI 3.1 document"
    );
    let paths = spec
        .get("paths")
        .and_then(Value::as_object)
        .map(|o| o.len())
        .unwrap_or(0);
    assert!(
        paths > 40,
        "fixture has only {paths} paths — truncated? Re-dump it."
    );
    for required in ["/v1/describe", "/v1/walk", "/v1/stats", "/v1/infer"] {
        assert!(
            spec.pointer(&format!(
                "/paths/{}",
                required.replace('~', "~0").replace('/', "~1")
            ))
            .is_some(),
            "fixture is missing {required}, which pg_infer depends on"
        );
    }
}

// Keep the unused helpers honest: they document intent for the next
// person even though the current assertions do not need them.
#[test]
fn probe_helpers_behave() {
    let base = json!({ "entity": "France", "edges": [], "latency_ms": 1.0 });
    assert!(
        reads_field::<DescribeResponse>(&base, "entity"),
        "`entity` is bound by DescribeResponse, so a wrong-typed value must fail"
    );
    assert!(
        !reads_field::<DescribeResponse>(&base, "nonexistent_field_xyz"),
        "an unbound field must be ignored (no deny_unknown_fields)"
    );
}
