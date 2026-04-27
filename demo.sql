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
CREATE EXTENSION IF NOT EXISTS pg_infer CASCADE;

-- Point data_directory at the vindexes folder so relative paths resolve.
SET infer.data_directory TO '/home/gburd/ws/larql/vindexes';

------------------------------------------------------------
-- 1. Register the model
------------------------------------------------------------
\echo '--- 1. Register model ---'
SELECT infer_create_model('qwen05b', 'qwen05b.vindex');

-- Set as default so we don't need model => 'qwen05b' everywhere.
SET infer.default_model TO 'qwen05b';

-- Verify it shows up
SELECT * FROM infer_models();

------------------------------------------------------------
-- 2. describe() — What does the model know about a concept?
------------------------------------------------------------
\echo ''
\echo '--- 2. describe() — knowledge edges ---'

\echo ''
\echo 'What does the model know about "France"?'
SELECT relation, target, round(confidence::numeric, 1) AS confidence, layer
  FROM describe('France')
 ORDER BY confidence DESC
 LIMIT 10;

\echo ''
\echo 'What does the model know about "Einstein"?'
SELECT relation, target, round(confidence::numeric, 1) AS confidence, layer
  FROM describe('Einstein')
 ORDER BY confidence DESC
 LIMIT 10;

\echo ''
\echo 'What does the model know about "Python"?'
SELECT relation, target, round(confidence::numeric, 1) AS confidence, layer
  FROM describe('Python')
 ORDER BY confidence DESC
 LIMIT 10;

------------------------------------------------------------
-- 3. walk() — Trace raw feature activations
------------------------------------------------------------
\echo ''
\echo '--- 3. walk() — raw feature activations ---'

\echo ''
\echo 'Top 15 activations for "The capital of France is":'
SELECT layer, feature, round(activation::numeric, 2) AS activation, concept
  FROM walk('The capital of France is', top => 15)
 ORDER BY activation DESC;

\echo ''
\echo 'Top 10 activations for "water boils at":'
SELECT layer, feature, round(activation::numeric, 2) AS activation, concept
  FROM walk('water boils at', top => 10)
 ORDER BY activation DESC;

------------------------------------------------------------
-- 4. similar_to() — Semantic similarity scores
------------------------------------------------------------
\echo ''
\echo '--- 4. similar_to() — semantic similarity ---'

SELECT 'France / Paris'     AS pair, round(similar_to('France', 'Paris')::numeric, 2)     AS score
 UNION ALL
SELECT 'France / Berlin',           round(similar_to('France', 'Berlin')::numeric, 2)
 UNION ALL
SELECT 'France / Europe',           round(similar_to('France', 'Europe')::numeric, 2)
 UNION ALL
SELECT 'France / banana',           round(similar_to('France', 'banana')::numeric, 2)
 UNION ALL
SELECT 'cat / dog',                 round(similar_to('cat', 'dog')::numeric, 2)
 UNION ALL
SELECT 'cat / algebra',             round(similar_to('cat', 'algebra')::numeric, 2)
 UNION ALL
SELECT 'Python / programming',      round(similar_to('Python', 'programming')::numeric, 2)
 UNION ALL
SELECT 'Python / snake',            round(similar_to('Python', 'snake')::numeric, 2);

------------------------------------------------------------
-- 5. implies() — Directional relationship test
------------------------------------------------------------
\echo ''
\echo '--- 5. implies() — does the model support this relationship? ---'

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

WITH japan  AS (SELECT target, confidence FROM describe('Japan')),
     china  AS (SELECT target, confidence FROM describe('China'))
SELECT j.target,
       round(j.confidence::numeric, 1) AS japan_score,
       round(c.confidence::numeric, 1) AS china_score
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
-- 8. Layer band analysis — syntax vs knowledge vs output
------------------------------------------------------------
\echo ''
\echo '--- 8. Layer band analysis ---'
\echo 'Where does "Shakespeare" knowledge concentrate?'
\echo '  Layers 0-8: syntax | 9-18: knowledge | 19-23: output'

SELECT CASE
         WHEN layer BETWEEN 0  AND 8  THEN 'syntax (0-8)'
         WHEN layer BETWEEN 9  AND 18 THEN 'knowledge (9-18)'
         WHEN layer BETWEEN 19 AND 23 THEN 'output (19-23)'
       END AS band,
       count(*) AS features,
       round(avg(activation)::numeric, 2) AS avg_activation,
       round(max(activation)::numeric, 2) AS max_activation
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
