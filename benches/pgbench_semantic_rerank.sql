-- pgbench script: semantic re-ranking under concurrent load.
--
-- Models the "normal query latency under load" target from the
-- remote-backend roadmap.  Each transaction issues one relational
-- filter (category_id), then re-ranks the top 20 candidates by
-- similar_to_many() against a query phrase drawn at random from
-- bench_phrases.
--
-- Run:
--   pgbench -n -c 32 -T 60 -f benches/pgbench_semantic_rerank.sql
--
-- Requires:
--   - pg_infer loaded and infer.default_model set
--   - benches/schema.sql applied (creates products, bench_phrases, seeds)
--   - a remote backend registered at infer.default_model

\set cid    random(1, 200)
\set phrase random(1, 8)

WITH candidates AS (
    SELECT id, name
    FROM products
    WHERE category_id = :cid
    ORDER BY created_at DESC
    LIMIT 20
),
q AS (
    SELECT phrase FROM bench_phrases WHERE id = :phrase
),
ranked AS (
    SELECT
        c.id,
        c.name,
        unnest(similar_to_many(
            (SELECT array_agg(name ORDER BY created_at DESC) FROM candidates),
            (SELECT phrase FROM q)
        )) AS score
    FROM candidates c
)
SELECT id, name, score
  FROM ranked
 WHERE score IS NOT NULL
 ORDER BY score DESC
 LIMIT 5;
