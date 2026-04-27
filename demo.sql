-- pg_infer demo script
-- Load with:  \i demo.sql   (from a psql session with pg_infer loaded)
--
-- Requires the Qwen 0.5B vindex at ../vindexes/qwen05b.vindex
-- Start with:  cargo pgrx run pg18

\echo '============================================'
\echo '  pg_infer demo — Qwen 0.5B (24 layers)'
\echo '============================================'
\echo ''

-- Load the extension (idempotent).
\echo 'CREATE EXTENSION IF NOT EXISTS pg_infer CASCADE;'
CREATE EXTENSION IF NOT EXISTS pg_infer CASCADE;

-- Point data_directory at the vindexes folder so relative paths resolve.
\echo 'SET infer.data_directory TO ''/home/gburd/ws/larql/vindexes'';'
SET infer.data_directory TO '/home/gburd/ws/larql/vindexes';

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
-- 2. describe() — What does the model know about a concept?
------------------------------------------------------------
\echo ''
\echo '--- 2. describe() — knowledge edges ---'

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
-- 3. walk() — Trace raw feature activations
------------------------------------------------------------
\echo ''
\echo '--- 3. walk() — raw feature activations ---'

\echo ''
\echo 'Top activations for "The capital of France is":'
\echo 'SELECT layer, feature, round(activation::numeric, 3) AS activation, concept'
\echo '  FROM walk(''The capital of France is'', top => 10)'
\echo '  ORDER BY activation DESC LIMIT 30;'
SELECT layer, feature, round(activation::numeric, 3) AS activation, concept
  FROM walk('The capital of France is', top => 10)
 ORDER BY activation DESC
 LIMIT 30;

\echo ''
\echo 'Top activations for "water boils at":'
\echo 'SELECT layer, feature, round(activation::numeric, 3) AS activation, concept'
\echo '  FROM walk(''water boils at'', top => 10)'
\echo '  ORDER BY activation DESC LIMIT 30;'
SELECT layer, feature, round(activation::numeric, 3) AS activation, concept
  FROM walk('water boils at', top => 10)
 ORDER BY activation DESC
 LIMIT 30;

------------------------------------------------------------
-- 4. similar_to() — Semantic similarity scores
------------------------------------------------------------
\echo ''
\echo '--- 4. similar_to() — semantic similarity ---'
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
-- 5. implies() — Directional relationship test
------------------------------------------------------------
\echo ''
\echo '--- 5. implies() — does the model support this relationship? ---'
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
-- 6. Cross-concept comparison with describe()
------------------------------------------------------------
\echo ''
\echo '--- 6. Cross-concept comparison ---'
\echo 'Shared knowledge edges between "Japan" and "China":'
\echo 'WITH japan AS (SELECT ... FROM describe(''Japan'')),'
\echo '     china AS (SELECT ... FROM describe(''China''))'
\echo 'SELECT ... FROM japan JOIN china USING (target) ...;'

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
-- 7. Column index and <~> operator
------------------------------------------------------------
\echo ''
\echo '--- 7. Column index + ORDER BY <~> ---'

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
\echo 'SELECT id, title, round((title <~> ''artificial intelligence'')::numeric, 4) AS distance'
\echo '  FROM topics ORDER BY title <~> ''artificial intelligence'' LIMIT 5;'
SELECT id, title, round((title <~> 'artificial intelligence')::numeric, 4) AS distance
  FROM topics
 ORDER BY title <~> 'artificial intelligence'
 LIMIT 5;

\echo ''
\echo 'Topics closest to "France":'
\echo 'SELECT id, title, round((title <~> ''France'')::numeric, 4) AS distance'
\echo '  FROM topics ORDER BY title <~> ''France'' LIMIT 5;'
SELECT id, title, round((title <~> 'France')::numeric, 4) AS distance
  FROM topics
 ORDER BY title <~> 'France'
 LIMIT 5;

------------------------------------------------------------
-- 8. Layer band analysis — syntax vs knowledge vs output
------------------------------------------------------------
\echo ''
\echo '--- 8. Layer band analysis ---'
\echo 'Where does "Shakespeare" knowledge concentrate?'
\echo '  Layers 0-8: syntax | 9-18: knowledge | 19-23: output'
\echo 'SELECT CASE ... END AS band, count(*), avg(activation), max(activation)'
\echo '  FROM walk(''Shakespeare'', top => 20) GROUP BY 1;'

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
