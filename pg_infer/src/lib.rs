use pgrx::prelude::*;

pgrx::pg_module_magic!();

mod am;
mod build;
mod error;
mod fn_describe;
mod fn_implies;
mod fn_infer;
mod fn_similar;
mod fn_walk;
mod gucs;
mod model_mgmt;
mod options;
mod page_reader;
mod pages;
mod registry;
mod scan;

// Bootstrap the infer schema and model registry table during CREATE EXTENSION.
extension_sql!(
    r#"
    CREATE SCHEMA IF NOT EXISTS infer;

    CREATE TABLE IF NOT EXISTS infer.models (
        model_name  TEXT PRIMARY KEY,
        vindex_path TEXT NOT NULL,
        source      TEXT NOT NULL,
        extract_level TEXT NOT NULL DEFAULT 'browse',
        num_layers    INT,
        hidden_size   INT,
        vocab_size    INT,
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
        assert_eq!(gate_thresh, Some("0".to_string()));
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
            "implies",
            "infer_create_model",
            "infer_drop_model",
            "infer_models",
            "infer_distance",
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
}
