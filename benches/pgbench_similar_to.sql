-- pgbench script: single similar_to() call per transaction.
--
-- This is the A path from "is the remote backend actually faster than
-- the in-process path".  Each transaction picks two random product
-- names and asks for a similarity score.  Meant to be run at varying
-- concurrency so you can watch throughput scale with server-side
-- activation cache and connection pooling.
--
-- Run (32-client, 60s sweep):
--   pgbench -n -c 32 -T 60 -f benches/pgbench_similar_to.sql
--
-- Compare with the mmap path by setting infer.default_model to a
-- 'local' model between runs.

\set a random(1, 10000)
\set b random(1, 10000)

SELECT similar_to(
    (SELECT name FROM products WHERE id = :a),
    (SELECT name FROM products WHERE id = :b)
);
