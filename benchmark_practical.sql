-- Practical benchmark: 10,000 rows, testing if gate cache cap prevents OOM
\timing on

DROP EXTENSION IF EXISTS pg_infer CASCADE;
CREATE EXTENSION pg_infer;
SET infer.data_directory TO '/home/gburd/ws/larql/vindexes';

-- Safety settings
SET infer.gate_cache_max_layers TO 4;   -- Cap decode cache (~450 MB)
SET infer.similarity_max_layers TO 6;   -- Sample 6 of 36 layers
SET infer.use_hnsw TO false;
SET infer.warmup_on_load TO false;

SELECT infer_create_model('qwen25_3b', 'qwen2.5-3b.vindex');
SET infer.default_model TO 'qwen25_3b';

\echo ''
\echo '========================================================================='
\echo ' Benchmark: Gate Cache Cap + Layer Sampling'
\echo '========================================================================='

-- Create test data
CREATE TEMP TABLE categories (id INT, name TEXT, description TEXT);
INSERT INTO categories VALUES
    (1, 'Electronics', 'Consumer electronics and gadgets'),
    (2, 'Computers', 'Desktop computers, laptops, and accessories'),
    (3, 'Software', 'Software applications and tools'),
    (4, 'Books', 'Physical and digital books'),
    (5, 'Music', 'Music albums and instruments'),
    (6, 'Movies', 'Films and video content'),
    (7, 'Sports', 'Sports equipment and apparel'),
    (8, 'Furniture', 'Home and office furniture'),
    (9, 'Clothing', 'Apparel and fashion items'),
    (10, 'Food', 'Groceries and food items');

CREATE TEMP TABLE products AS
SELECT
    generate_series AS id,
    'Product ' || generate_series AS name,
    'This is a product description for item number ' || generate_series ||
    ' with various features and specifications for testing' AS description
FROM generate_series(1, 10000);

\echo ''
\echo '--- Test 1: Single query (baseline, warm cache) ---'
SELECT round(similar_to('laptop computer', 'desktop computer')::numeric, 3) AS score;
SELECT round(similar_to('laptop computer', 'desktop computer')::numeric, 3) AS score;

\echo ''
\echo '--- Test 2: 10 category comparisons ---'
SELECT
    c.name,
    round(similar_to(c.description, 'portable computer device')::numeric, 3) AS sim
FROM categories c
ORDER BY sim DESC
LIMIT 5;

\echo ''
\echo '--- Test 3: 50 product comparisons (was crashing before) ---'
SELECT
    p.id,
    round(similar_to(p.description, 'electronic gadget')::numeric, 3) AS sim
FROM products p
WHERE p.id <= 50
ORDER BY sim DESC
LIMIT 10;

\echo ''
\echo '--- Test 4: 100 product comparisons ---'
SELECT
    p.id,
    round(similar_to(p.description, 'software tool')::numeric, 3) AS sim
FROM products p
WHERE p.id <= 100
ORDER BY sim DESC
LIMIT 10;

\echo ''
\echo '--- Test 5: 200 product comparisons ---'
SELECT
    p.id,
    round(similar_to(p.description, 'music instrument')::numeric, 3) AS sim
FROM products p
WHERE p.id <= 200
ORDER BY sim DESC
LIMIT 10;

\echo ''
\echo '--- Test 6: similar_to_many() batch (50 products) ---'
SELECT unnest(similar_to_many(
    (SELECT array_agg(description) FROM products WHERE id <= 50),
    'technology gadget'
)) AS score
LIMIT 10;

\echo ''
\echo 'COMPLETE: All tests passed without crashes!'
\echo ''

SELECT infer_drop_model('qwen25_3b');
