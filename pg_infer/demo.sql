-- pg_infer demo script
-- Load with:  \i demo.sql   (from a psql session with pg_infer loaded)
--
-- Requires the Qwen 0.5B vindex at ../vindexes/qwen05b.vindex
-- Start with:  cargo pgrx run pg18

\echo '============================================'
\echo '  pg_infer demo — Qwen 0.5B (24 layers)'
\echo '============================================'
\echo ''

-- (Re)load the extension so new function definitions are always applied.
\echo 'DROP EXTENSION IF EXISTS pg_infer CASCADE;'
DROP EXTENSION IF EXISTS pg_infer CASCADE;
\echo 'CREATE EXTENSION pg_infer;'
CREATE EXTENSION pg_infer;

-- Point data_directory at the vindexes folder so relative paths resolve.
\echo 'SET infer.data_directory TO ''/home/gburd/ws/larql/vindexes'';'
SET infer.data_directory TO '/home/gburd/ws/larql/vindexes';

-- Qwen 0.5B has very weak gate activations (~0.04-0.07 max).
-- The default threshold of 5.0 is designed for larger models (8B+).
-- Use adaptive mode (threshold=0) so describe() works on this tiny model.
\echo 'SET infer.gate_threshold = 0;  -- adaptive mode for tiny models'
SET infer.gate_threshold = 0;

------------------------------------------------------------
-- 1. Register the model (clean up stale data from previous runs first)
------------------------------------------------------------
\echo ''
\echo '--- 1. Register model ---'
\echo 'SELECT infer_drop_model(''qwen05b'');'
SELECT infer_drop_model('qwen05b');

\echo ''
\echo 'SELECT infer_create_model(''qwen05b'', ''qwen05b.vindex'');'
SELECT infer_create_model('qwen05b', 'qwen05b.vindex');

-- Set as default so we don't need model => 'qwen05b' everywhere.
\echo ''
\echo 'SET infer.default_model TO ''qwen05b'';'
SET infer.default_model TO 'qwen05b';

-- Verify it shows up
\echo ''
\echo 'SELECT * FROM infer_models();'
SELECT * FROM infer_models();

------------------------------------------------------------
-- 2. infer_show_layers() — Layer metadata
------------------------------------------------------------
\echo ''
\echo '--- 2. infer_show_layers() — layer metadata with band info ---'
\echo 'SELECT * FROM infer_show_layers() LIMIT 10;'
SELECT * FROM infer_show_layers() LIMIT 10;

------------------------------------------------------------
-- 3. describe() — What does the model know about a concept?
------------------------------------------------------------
\echo ''
\echo '--- 3. describe() — knowledge edges (adaptive threshold) ---'

\echo ''
\echo 'What does the model know about "France"?'
\echo 'SELECT relation, target, round(confidence::numeric, 3) AS confidence, layer'
\echo '  FROM describe(''France'') ORDER BY confidence DESC LIMIT 15;'
SELECT relation, target, round(confidence::numeric, 3) AS confidence, layer
  FROM describe('France')
 ORDER BY confidence DESC
 LIMIT 15;

\echo ''
\echo 'What does the model know about "Einstein"?'
\echo 'SELECT relation, target, round(confidence::numeric, 3) AS confidence, layer'
\echo '  FROM describe(''Einstein'') ORDER BY confidence DESC LIMIT 15;'
SELECT relation, target, round(confidence::numeric, 3) AS confidence, layer
  FROM describe('Einstein')
 ORDER BY confidence DESC
 LIMIT 15;

\echo ''
\echo 'What does the model know about "Python"?'
\echo 'SELECT relation, target, round(confidence::numeric, 3) AS confidence, layer'
\echo '  FROM describe(''Python'') ORDER BY confidence DESC LIMIT 15;'
SELECT relation, target, round(confidence::numeric, 3) AS confidence, layer
  FROM describe('Python')
 ORDER BY confidence DESC
 LIMIT 15;

------------------------------------------------------------
-- 4. nearest_to() — Single-layer gate KNN probe
------------------------------------------------------------
\echo ''
\echo '--- 4. nearest_to() — probe a single layer ---'
\echo 'SELECT * FROM nearest_to(''France'', layer => 14, top => 10);'
SELECT * FROM nearest_to('France', layer => 14, top => 10);

\echo ''
\echo 'SELECT * FROM nearest_to(''Python'', layer => 10, top => 10);'
SELECT * FROM nearest_to('Python', layer => 10, top => 10);

------------------------------------------------------------
-- 5. walk() — Trace raw feature activations
------------------------------------------------------------
\echo ''
\echo '--- 5. walk() — raw feature activations ---'

\echo ''
\echo 'Top activations for "The capital of France is":'
\echo 'SELECT layer, feature, round(activation::numeric, 3) AS activation, concept'
\echo '  FROM walk(''The capital of France is'', top => 5)'
\echo '  ORDER BY activation DESC LIMIT 20;'
SELECT layer, feature, round(activation::numeric, 3) AS activation, concept
  FROM walk('The capital of France is', top => 5)
 ORDER BY activation DESC
 LIMIT 20;

------------------------------------------------------------
-- 6. infer_explain_walk() — Annotated walk with bands
------------------------------------------------------------
\echo ''
\echo '--- 6. infer_explain_walk() — walk with band + secondary tokens ---'
\echo 'SELECT * FROM infer_explain_walk(''The capital of France is'', top => 3) LIMIT 15;'
SELECT * FROM infer_explain_walk('The capital of France is', top => 3) LIMIT 15;

------------------------------------------------------------
-- 7. infer_show_features() — Enumerate features at a layer
------------------------------------------------------------
\echo ''
\echo '--- 7. infer_show_features() — features at layer 14 ---'
\echo 'SELECT * FROM infer_show_features(14, top => 15);'
SELECT * FROM infer_show_features(14, top => 15);

------------------------------------------------------------
-- 8. infer_show_relations() — Discovered relation tokens
------------------------------------------------------------
\echo ''
\echo '--- 8. infer_show_relations() — aggregated content tokens ---'
\echo 'SELECT * FROM infer_show_relations() LIMIT 15;'
SELECT * FROM infer_show_relations() LIMIT 15;

------------------------------------------------------------
-- 9. similar_to() — Semantic similarity scores
------------------------------------------------------------
\echo ''
\echo '--- 9. similar_to() — semantic similarity ---'
\echo 'SELECT ... similar_to(a, b) ... for various concept pairs;'

SELECT 'France / Paris'     AS pair, round(similar_to('France', 'Paris')::numeric, 3)     AS score
 UNION ALL
SELECT 'France / Berlin',           round(similar_to('France', 'Berlin')::numeric, 3)
 UNION ALL
SELECT 'France / Europe',           round(similar_to('France', 'Europe')::numeric, 3)
 UNION ALL
SELECT 'France / banana',           round(similar_to('France', 'banana')::numeric, 3)
 UNION ALL
SELECT 'cat / dog',                 round(similar_to('cat', 'dog')::numeric, 3)
 UNION ALL
SELECT 'cat / algebra',             round(similar_to('cat', 'algebra')::numeric, 3)
 UNION ALL
SELECT 'Python / programming',      round(similar_to('Python', 'programming')::numeric, 3)
 UNION ALL
SELECT 'Python / snake',            round(similar_to('Python', 'snake')::numeric, 3);

------------------------------------------------------------
-- 10. implies() — Directional relationship test
------------------------------------------------------------
\echo ''
\echo '--- 10. implies() — does the model support this relationship? ---'
\echo 'SELECT subject, object, implies(subject, object) AS implied FROM ...;'

SELECT subject, object, implies(subject, object) AS implied
  FROM (VALUES
    ('France',   'Paris'),
    ('France',   'French'),
    ('France',   'banana'),
    ('Einstein', 'physics'),
    ('Einstein', 'cooking'),
    ('water',    'liquid'),
    ('Python',   'programming'),
    ('cat',      'animal')
  ) AS t(subject, object);

------------------------------------------------------------
-- 11. Cross-concept comparison with describe()
------------------------------------------------------------
\echo ''
\echo '--- 11. Cross-concept comparison ---'
\echo 'Shared knowledge edges between "Japan" and "China":'

WITH japan  AS (SELECT target, confidence FROM describe('Japan')),
     china  AS (SELECT target, confidence FROM describe('China'))
SELECT j.target,
       round(j.confidence::numeric, 3) AS japan_score,
       round(c.confidence::numeric, 3) AS china_score
  FROM japan j
  JOIN china c USING (target)
 ORDER BY j.confidence + c.confidence DESC
 LIMIT 10;

------------------------------------------------------------
-- 12. Column index and <~> operator
------------------------------------------------------------
\echo ''
\echo '--- 12. Column index + ORDER BY <~> ---'

DROP TABLE IF EXISTS topics;
CREATE TEMP TABLE topics (id serial, title text);
INSERT INTO topics (title) VALUES
    ('Machine learning fundamentals'),
    ('French cooking recipes'),
    ('Quantum physics experiments'),
    ('The history of Paris'),
    ('Database query optimization'),
    ('Neural network architectures'),
    ('European geography'),
    ('Cat behavior and training'),
    ('Python programming tutorial'),
    ('Space exploration missions');

\echo ''
\echo 'CREATE INDEX ON topics USING infer (title) WITH (model = ''qwen05b'');'
CREATE INDEX ON topics USING infer (title) WITH (model = 'qwen05b');

\echo ''
\echo 'Topics closest to "artificial intelligence":'
SELECT id, title, round((title <~> 'artificial intelligence')::numeric, 4) AS distance
  FROM topics
 ORDER BY title <~> 'artificial intelligence'
 LIMIT 5;

\echo ''
\echo 'Topics closest to "France":'
SELECT id, title, round((title <~> 'France')::numeric, 4) AS distance
  FROM topics
 ORDER BY title <~> 'France'
 LIMIT 5;

------------------------------------------------------------
-- 13. Layer band analysis — syntax vs knowledge vs output
------------------------------------------------------------
\echo ''
\echo '--- 13. Layer band analysis ---'
\echo 'Where does "Shakespeare" knowledge concentrate?'
\echo '  Layers 0-8: syntax | 9-18: knowledge | 19-23: output'

SELECT CASE
         WHEN layer BETWEEN 0  AND 8  THEN 'syntax (0-8)'
         WHEN layer BETWEEN 9  AND 18 THEN 'knowledge (9-18)'
         WHEN layer BETWEEN 19 AND 23 THEN 'output (19-23)'
       END AS band,
       count(*) AS features,
       round(avg(activation)::numeric, 3) AS avg_activation,
       round(max(activation)::numeric, 3) AS max_activation
  FROM walk('Shakespeare', top => 20)
 GROUP BY 1
 ORDER BY 1;

------------------------------------------------------------
-- Cleanup
------------------------------------------------------------
\echo ''
\echo '--- Cleanup ---'
SELECT infer_drop_model('qwen05b');
\echo 'Done.'
