#![allow(clippy::type_complexity)]

use pgrx::prelude::*;

pgrx::pg_module_magic!();

mod am;
mod am_build;
mod am_cost;
mod am_options;
mod am_scan;
mod backend;
mod error;
mod fn_cache;
mod fn_describe;
mod fn_diff;
mod fn_implies;
mod fn_infer;
mod fn_nearest;
mod fn_show;
mod fn_similar;
mod fn_walk;
mod gucs;
mod helpers;
mod interrupt;
mod model_mgmt;
mod registry;
mod relation_classify;
mod tracing_layer;

// Bootstrap the infer schema and model registry table during CREATE EXTENSION.
extension_sql!(
    r#"
    CREATE SCHEMA IF NOT EXISTS infer;

    -- Model registry.  One row per registered model; the row points at
    -- either a local vindex directory (backend = 'local') or a remote
    -- larql-server endpoint (backend = 'remote').
    CREATE TABLE IF NOT EXISTS infer.models (
        model_name    TEXT PRIMARY KEY,
        vindex_path   TEXT NOT NULL,
        source        TEXT NOT NULL,
        extract_level TEXT NOT NULL DEFAULT 'browse',
        num_layers    INT,
        hidden_size   INT,
        vocab_size    INT,
        backend       TEXT NOT NULL DEFAULT 'local',
        server_url    TEXT,
        registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
"#,
    name = "bootstrap_schema",
);

/// Extension initialization — called once per backend when the shared library
/// is loaded.  Registers GUC parameters so they are visible before
/// `CREATE EXTENSION` runs.
#[pg_guard]
pub extern "C-unwind" fn _PG_init() {
    // SAFETY: called exactly once per backend by PostgreSQL.
    unsafe {
        gucs::init();
    }

    // Initialize tracing subscriber that routes to PostgreSQL elog().
    // The filter respects RUST_LOG env var with a default of "pg_infer=info".
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let filter = tracing_subscriber::filter::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            let level = gucs::log_level();
            tracing_subscriber::filter::EnvFilter::new(format!("pg_infer={}", level))
        });

    // Ignore errors from re-initialization (happens during pgrx test harness).
    let _ = tracing_subscriber::registry()
        .with(tracing_layer::PgLogLayer)
        .with(filter)
        .try_init();

    // Log auto-detected UDS socket (informational).
    if let Some(sock) = backend::remote::detect_local_socket() {
        pgrx::log!("pg_infer: auto-detected local larql-server at {}", sock);
    }
}

// ---------------------------------------------------------------------------
// pgrx test framework support
// ---------------------------------------------------------------------------

/// Required by pgrx's `#[pg_test]` macro — provides test setup hooks.
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        // No special setup needed.
    }

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}

// ---------------------------------------------------------------------------
// pgrx integration tests
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_extension_loads() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("CREATE EXTENSION failed");
    }

    #[pg_test]
    fn test_gucs_exist() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("CREATE EXTENSION failed");
        Spi::run("SHOW infer.default_model").expect("SHOW default_model failed");
        Spi::run("SHOW infer.data_directory").expect("SHOW data_directory failed");
        Spi::run("SHOW infer.max_memory").expect("SHOW max_memory failed");
        Spi::run("SHOW infer.auto_download").expect("SHOW auto_download failed");
        Spi::run("SHOW infer.gate_threshold").expect("SHOW gate_threshold failed");
        Spi::run("SHOW infer.grid_url").expect("SHOW grid_url failed");
        Spi::run("SHOW infer.grid_poll_interval").expect("SHOW grid_poll_interval failed");
    }

    #[pg_test]
    fn test_guc_default_values() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("CREATE EXTENSION failed");

        let max_mem = Spi::get_one::<String>("SHOW infer.max_memory")
            .expect("SHOW failed");
        assert_eq!(max_mem, Some("8192".to_string()));

        let auto_dl = Spi::get_one::<String>("SHOW infer.auto_download")
            .expect("SHOW failed");
        assert_eq!(auto_dl, Some("on".to_string()));

        let data_dir = Spi::get_one::<String>("SHOW infer.data_directory")
            .expect("SHOW failed");
        assert_eq!(data_dir, Some("infer".to_string()));

        let gate_thresh = Spi::get_one::<String>("SHOW infer.gate_threshold")
            .expect("SHOW failed");
        assert_eq!(gate_thresh, Some("5".to_string()));
    }

    #[pg_test]
    fn test_set_default_model() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("CREATE EXTENSION failed");
        Spi::run("SET infer.default_model = 'test_model'").expect("SET failed");
        let val = Spi::get_one::<String>("SHOW infer.default_model")
            .expect("SHOW failed");
        assert_eq!(val, Some("test_model".to_string()));
    }

    #[pg_test]
    fn test_models_table_exists() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("CREATE EXTENSION failed");
        let count = Spi::get_one::<i64>("SELECT count(*) FROM infer.models")
            .expect("query failed");
        assert_eq!(count, Some(0));
    }

    #[pg_test]
    fn test_infer_models_function() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("CREATE EXTENSION failed");
        let count = Spi::get_one::<i64>("SELECT count(*) FROM infer_models()")
            .expect("query failed");
        assert_eq!(count, Some(0));
    }

    #[pg_test]
    fn test_drop_nonexistent_model() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("CREATE EXTENSION failed");
        let result = Spi::get_one::<String>("SELECT infer_drop_model('nonexistent')")
            .expect("query failed");
        assert!(result.is_some());
    }

    #[pg_test]
    fn test_function_signatures_exist() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("CREATE EXTENSION failed");

        for func_name in &[
            "walk",
            "describe",
            "infer",
            "similar_to",
            "similar_to_many",
            "implies",
            "infer_create_model",
            "infer_create_model_remote",
            "infer_create_model_grid",
            "infer_create_models_remote",
            "infer_drop_model",
            "infer_models",
            "infer_distance",
            "nearest_to",
            "infer_show_layers",
            "infer_show_features",
            "infer_show_relations",
            "infer_explain_walk",
            "infer_diff",
            "infer_detect_server",
            "infer_warmup",
            "infer_server_stats",
            "infer_stats",
        ] {
            let exists = Spi::get_one::<bool>(&format!(
                "SELECT EXISTS(SELECT 1 FROM pg_proc WHERE proname = '{}')",
                func_name
            ))
            .expect("query failed");
            assert_eq!(
                exists,
                Some(true),
                "function '{}' not found in pg_proc",
                func_name
            );
        }
    }

    #[pg_test]
    fn test_distance_operator_exists() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("CREATE EXTENSION failed");

        let exists = Spi::get_one::<bool>(
            "SELECT EXISTS(SELECT 1 FROM pg_operator WHERE oprname = '<~>')",
        )
        .expect("query failed");
        assert_eq!(exists, Some(true));
    }

    #[pg_test]
    fn test_create_model_nonexistent_path() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("CREATE EXTENSION failed");
        // Creating a model with a path that doesn't exist should fail.
        let result = std::panic::catch_unwind(|| {
            Spi::run("SELECT infer_create_model('bad', '/nonexistent/path/to/vindex')")
                .expect("should not succeed");
        });
        assert!(result.is_err(), "expected error for nonexistent path");
    }

    #[pg_test]
    fn test_new_gucs_exist() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("ext");
        Spi::run("SHOW infer.describe_top_k").expect("describe_top_k");
        Spi::run("SHOW infer.walk_embed_mode").expect("walk_embed_mode");
    }

    #[pg_test]
    fn test_describe_top_k_settable() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("ext");
        Spi::run("SET infer.describe_top_k = 50").expect("SET");
        let val = Spi::get_one::<String>("SHOW infer.describe_top_k").expect("SHOW");
        assert_eq!(val, Some("50".to_string()));
    }

    #[pg_test]
    fn test_access_method_exists() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("CREATE EXTENSION failed");
        let exists = Spi::get_one::<bool>(
            "SELECT EXISTS(SELECT 1 FROM pg_am WHERE amname = 'infer')",
        )
        .expect("query failed");
        assert_eq!(exists, Some(true));
    }

    #[pg_test]
    fn test_operator_class_exists() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("CREATE EXTENSION failed");
        let exists = Spi::get_one::<bool>(
            "SELECT EXISTS(SELECT 1 FROM pg_opclass WHERE opcname = 'infer_text_ops')",
        )
        .expect("query failed");
        assert_eq!(exists, Some(true));
    }

    #[pg_test]
    fn test_similarity_operator_exists() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("CREATE EXTENSION failed");
        let exists = Spi::get_one::<bool>(
            "SELECT EXISTS(SELECT 1 FROM pg_operator WHERE oprname = '<~')",
        )
        .expect("query failed");
        assert_eq!(exists, Some(true));
    }

    #[pg_test]
    fn test_implies_operator_exists() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("CREATE EXTENSION failed");
        let exists = Spi::get_one::<bool>(
            "SELECT EXISTS(SELECT 1 FROM pg_operator WHERE oprname = '@>')",
        )
        .expect("query failed");
        assert_eq!(exists, Some(true));
    }

    #[pg_test]
    fn test_create_index_using_infer_syntax() {
        Spi::run("CREATE EXTENSION IF NOT EXISTS pg_infer").expect("CREATE EXTENSION failed");
        Spi::run("CREATE TABLE test_docs (id int, title text)").expect("CREATE TABLE");
        // Verify the parser accepts "USING infer" syntax by confirming:
        // 1. The AM exists (parser can resolve the name)
        let am_ok = Spi::get_one::<bool>(
            "SELECT EXISTS(SELECT 1 FROM pg_am WHERE amname = 'infer')",
        )
        .expect("query failed");
        assert_eq!(am_ok, Some(true));
        // 2. The default opclass is registered for the AM
        let opc_ok = Spi::get_one::<bool>(
            "SELECT EXISTS(SELECT 1 FROM pg_opclass WHERE opcname = 'infer_text_ops')",
        )
        .expect("query failed");
        assert_eq!(opc_ok, Some(true));
        // Note: Actually creating the index would require a model registered
        // in infer.models.  The ambuild callback errors with ModelNotFound,
        // but pgrx's error mechanism (longjmp) cannot be caught by
        // std::panic::catch_unwind, so we don't attempt it here.
    }
}
