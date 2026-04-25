-- pg_infer regression tests
-- Extension lifecycle
CREATE EXTENSION pg_infer;

-- GUC parameters
SHOW infer.default_model;
SHOW infer.max_memory;
SHOW infer.auto_download;
SHOW infer.data_directory;

-- Model registry table
SELECT count(*) FROM infer.models;

-- Model management functions
SELECT * FROM infer_models();

-- Function signatures (verify they exist, even without a loaded model)
SELECT proname, pg_get_function_arguments(oid) AS args,
       pg_get_function_result(oid) AS rettype
FROM pg_proc WHERE proname = 'describe' AND prokind = 'f'
ORDER BY proname;

SELECT proname, pg_get_function_arguments(oid) AS args,
       pg_get_function_result(oid) AS rettype
FROM pg_proc WHERE proname = 'walk' AND prokind = 'f'
ORDER BY proname;

SELECT proname, pg_get_function_arguments(oid) AS args,
       pg_get_function_result(oid) AS rettype
FROM pg_proc WHERE proname = 'infer' AND prokind = 'f'
ORDER BY proname;

SELECT proname, pg_get_function_arguments(oid) AS args,
       pg_get_function_result(oid) AS rettype
FROM pg_proc WHERE proname = 'similar_to' AND prokind = 'f'
ORDER BY proname;

SELECT proname, pg_get_function_arguments(oid) AS args,
       pg_get_function_result(oid) AS rettype
FROM pg_proc WHERE proname = 'implies' AND prokind = 'f'
ORDER BY proname;

SELECT proname, pg_get_function_arguments(oid) AS args,
       pg_get_function_result(oid) AS rettype
FROM pg_proc WHERE proname = 'infer_create_model' AND prokind = 'f'
ORDER BY proname;

SELECT proname, pg_get_function_arguments(oid) AS args,
       pg_get_function_result(oid) AS rettype
FROM pg_proc WHERE proname = 'infer_drop_model' AND prokind = 'f'
ORDER BY proname;

SELECT proname, pg_get_function_arguments(oid) AS args,
       pg_get_function_result(oid) AS rettype
FROM pg_proc WHERE proname = 'infer_models' AND prokind = 'f'
ORDER BY proname;

SELECT proname, pg_get_function_arguments(oid) AS args,
       pg_get_function_result(oid) AS rettype
FROM pg_proc WHERE proname = 'infer_distance' AND prokind = 'f'
ORDER BY proname;

-- Operator
SELECT oprname, oprleft::regtype, oprright::regtype
FROM pg_operator WHERE oprname = '<~>';

-- Cleanup
DROP EXTENSION pg_infer;
