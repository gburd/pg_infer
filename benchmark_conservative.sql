-- Conservative benchmark: Only test what's practical
\timing on

DROP EXTENSION IF EXISTS pg_infer CASCADE;
CREATE EXTENSION pg_infer;
SET infer.data_directory TO '/home/gburd/ws/larql/vindexes';

SELECT infer_create_model('qwen25_3b', 'qwen2.5-3b.vindex');
SET infer.default_model TO 'qwen25_3b';
SET infer.similarity_max_layers TO 12;

\echo ''
\echo '========================================================================='
\echo ' Test 1: Single Similarity Computations (Baseline)'
\echo '========================================================================='

SELECT 'laptop vs desktop' AS comparison, round(similar_to('laptop computer', 'desktop computer')::numeric, 3) AS similarity;
SELECT 'apple vs orange' AS comparison, round(similar_to('apple fruit', 'orange fruit')::numeric, 3) AS similarity;
SELECT 'car vs airplane' AS comparison, round(similar_to('car', 'airplane')::numeric, 3) AS similarity;

\echo ''
\echo '========================================================================='
\echo ' Test 2: Category Matching (20 comparisons)'
\echo '========================================================================='

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
    (10, 'Food', 'Groceries and food items'),
    (11, 'Automotive', 'Car parts and accessories'),
    (12, 'Tools', 'Hand tools and power tools'),
    (13, 'Garden', 'Gardening supplies and equipment'),
    (14, 'Toys', 'Toys and games for children'),
    (15, 'Health', 'Health and wellness products'),
    (16, 'Beauty', 'Beauty and personal care items'),
    (17, 'Pet', 'Pet supplies and accessories'),
    (18, 'Office', 'Office supplies and equipment'),
    (19, 'Art', 'Art supplies and crafts'),
    (20, 'Travel', 'Travel accessories and luggage');

\echo '--- Find best category for "portable computer device" (20 comparisons) ---'
SELECT
    c.name,
    round(similar_to(c.description, 'portable computer device')::numeric, 3) AS similarity
FROM categories c
ORDER BY similarity DESC
LIMIT 5;

\echo ''
\echo '========================================================================='
\echo ' Test 3: Small Batch (50 comparisons)'
\echo '========================================================================='

CREATE TEMP TABLE products AS
SELECT
    generate_series AS id,
    'Product ' || generate_series AS name,
    'This is a product description for item number ' || generate_series || ' with various features' AS description
FROM generate_series(1, 50);

\echo '--- Find products similar to "computer equipment" (50 comparisons) ---'
SELECT
    p.id,
    p.name,
    round(similar_to(p.description, 'computer equipment')::numeric, 3) AS similarity
FROM products p
ORDER BY similarity DESC
LIMIT 10;

\echo ''
\echo '========================================================================='
\echo ' Test 4: Moderate Batch (100 comparisons)'
\echo '========================================================================='

INSERT INTO products
SELECT
    generate_series AS id,
    'Product ' || generate_series AS name,
    'This is a product description for item number ' || generate_series || ' with various features' AS description
FROM generate_series(51, 100);

\echo '--- Find products similar to "electronic device" (100 comparisons) ---'
SELECT
    p.id,
    p.name,
    round(similar_to(p.description, 'electronic device')::numeric, 3) AS similarity
FROM products p
ORDER BY similarity DESC
LIMIT 10;

\echo ''
\echo '========================================================================='
\echo ' Performance Summary'
\echo '========================================================================='
\echo ''
\echo 'Based on observed times:'
\echo '- Single computation: ~2.6s (cached) to ~55s (cold/all layers)'
\echo '- 20 comparisons: ~52s (20 × 2.6s)'
\echo '- 50 comparisons: ~130s = 2.2 minutes'
\echo '- 100 comparisons: ~260s = 4.3 minutes'
\echo ''
\echo 'Scaling estimates:'
\echo '- 500 rows: ~21 minutes'
\echo '- 1,000 rows: ~43 minutes'
\echo '- 10,000 rows: ~7.2 hours'
\echo ''
\echo 'VERDICT: NOT suitable for "index speeds"'
\echo ''
\echo 'Traditional B-tree index: <1ms lookup'
\echo 'pg_infer similar_to(): ~2,600ms computation'
\echo 'Ratio: 2,600× slower than index lookup'
\echo ''
\echo 'Practical use cases:'
\echo '1. Small result sets (<100 rows) after traditional filtering'
\echo '2. Semantic re-ranking of pre-filtered results'
\echo '3. Category/tag assignment (small taxonomy)'
\echo '4. Deduplication on small batches'
\echo '5. Interactive search with LIMIT 20-50'
\echo ''
\echo 'NOT suitable for:'
\echo '1. Full table scans'
\echo '2. Large JOIN operations'
\echo '3. Real-time filtering (use traditional indexes first)'
\echo '4. High-throughput OLTP queries'
\echo ''

SELECT infer_drop_model('qwen25_3b');
