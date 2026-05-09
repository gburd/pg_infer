-- Schema + seed data for pg_infer's pgbench suite.
-- Creates a small product catalog and the phrase table the bench
-- scripts draw from.  Idempotent — rerunnable.

CREATE EXTENSION IF NOT EXISTS pg_infer;

-- ── Products ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS products (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    category_id INT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS products_cat_idx
    ON products (category_id, created_at DESC);

-- Seed 10k rows across 200 categories.
INSERT INTO products (name, category_id)
SELECT
    -- Noun-phrase generator: combination of three vocab buckets.
    (ARRAY[
        'fast', 'robust', 'distributed', 'scalable', 'native', 'modular',
        'embedded', 'open-source', 'compact', 'federated', 'lightweight'
    ])[1 + (floor(random() * 11))::int]
    || ' '
    || (ARRAY[
        'vector', 'graph', 'semantic', 'analytic', 'inference',
        'query', 'storage', 'retrieval', 'streaming', 'training',
        'evaluation', 'transformer', 'embedding', 'knowledge', 'search'
    ])[1 + (floor(random() * 15))::int]
    || ' '
    || (ARRAY[
        'engine', 'database', 'toolkit', 'pipeline', 'platform',
        'runtime', 'service', 'system', 'framework', 'library'
    ])[1 + (floor(random() * 10))::int]
    AS name,
    1 + (floor(random() * 200))::int AS category_id
FROM generate_series(1, 10000)
ON CONFLICT DO NOTHING;

-- ── Bench phrases ──────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS bench_phrases (
    id     INT PRIMARY KEY,
    phrase TEXT NOT NULL
);

INSERT INTO bench_phrases (id, phrase) VALUES
    (1, 'high performance computing'),
    (2, 'machine learning framework'),
    (3, 'natural language processing'),
    (4, 'distributed systems'),
    (5, 'artificial intelligence'),
    (6, 'large language model'),
    (7, 'neural network training'),
    (8, 'computer vision')
ON CONFLICT (id) DO NOTHING;

ANALYZE products;
ANALYZE bench_phrases;

SELECT
    (SELECT count(*) FROM products)      AS products_rows,
    (SELECT count(*) FROM bench_phrases) AS phrases;
