 
-- Note: Syntax is PostgreSQL-style (DATE_TRUNC, DATE_PART, CURRENT_DATE, NTILE).
-- For MySQL / SQL Server, equivalent date/window functions can be substituted.

```sql
-- =====================================================================
-- E-COMMERCE ANALYTICS – SAMPLE SQL QUERIES
-- These queries assume a transactional table named `transactions`
-- with columns similar to the synthetic data in this project:
--   transaction_id, date, customer_id, product_id,
--   quantity, unit_price, discount_percent, net_amount,
--   category, region
-- =====================================================================

-- 1. Monthly revenue trend
SELECT
    DATE_TRUNC('month', date) AS month,
    SUM(net_amount)           AS total_revenue,
    SUM(quantity)             AS total_units,
    COUNT(DISTINCT customer_id) AS unique_customers
FROM transactions
GROUP BY month
ORDER BY month;


-- 2. Top 10 products by revenue
SELECT
    product_id,
    SUM(net_amount) AS total_revenue,
    SUM(quantity)   AS total_units_sold,
    COUNT(*)        AS order_count
FROM transactions
GROUP BY product_id
ORDER BY total_revenue DESC
LIMIT 10;


-- 3. Category-level performance
SELECT
    category,
    SUM(net_amount) AS total_revenue,
    SUM(quantity)   AS total_units_sold,
    COUNT(*)        AS order_count
FROM transactions
GROUP BY category
ORDER BY total_revenue DESC;


-- 4. Region-level performance
SELECT
    region,
    SUM(net_amount) AS total_revenue,
    COUNT(DISTINCT customer_id) AS unique_customers,
    COUNT(*)        AS order_count
FROM transactions
GROUP BY region
ORDER BY total_revenue DESC;


-- 5. RFM-style customer summary
-- Recency: days since last purchase (requires a reference date)
-- Frequency: number of orders
-- Monetary: total spend

-- Example reference date: current_date (can be replaced as needed)
WITH customer_agg AS (
    SELECT
        customer_id,
        MAX(date)            AS last_purchase_date,
        COUNT(*)             AS frequency,
        SUM(net_amount)      AS monetary
    FROM transactions
    GROUP BY customer_id
)
SELECT
    customer_id,
    last_purchase_date,
    DATE_PART('day', CURRENT_DATE - last_purchase_date) AS recency_days,
    frequency,
    monetary
FROM customer_agg
ORDER BY monetary DESC;


-- 6. Identify high-value customers (top 20% by monetary)
WITH customer_agg AS (
    SELECT
        customer_id,
        SUM(net_amount) AS monetary
    FROM transactions
    GROUP BY customer_id
),
ranked AS (
    SELECT
        customer_id,
        monetary,
        NTILE(5) OVER (ORDER BY monetary DESC) AS value_bucket
    FROM customer_agg
)
SELECT
    customer_id,
    monetary
FROM ranked
WHERE value_bucket = 1    -- top 20%
ORDER BY monetary DESC;


-- 7. Simple churn-style inactivity flag (60+ days inactive)
WITH last_order AS (
    SELECT
        customer_id,
        MAX(date) AS last_purchase_date
    FROM transactions
    GROUP BY customer_id
)
SELECT
    customer_id,
    last_purchase_date,
    DATE_PART('day', CURRENT_DATE - last_purchase_date) AS days_since_last_purchase,
    CASE
        WHEN DATE_PART('day', CURRENT_DATE - last_purchase_date) >= 60
            THEN TRUE
        ELSE FALSE
    END AS churn_risk_flag
FROM last_order
ORDER BY days_since_last_purchase DESC;


-- 8. Customers with high value at risk (churn flag + high spend)
WITH customer_value AS (
    SELECT
        customer_id,
        SUM(net_amount) AS total_spent
    FROM transactions
    GROUP BY customer_id
),
last_order AS (
    SELECT
        customer_id,
        MAX(date) AS last_purchase_date
    FROM transactions
    GROUP BY customer_id
),
combined AS (
    SELECT
        v.customer_id,
        v.total_spent,
        l.last_purchase_date,
        DATE_PART('day', CURRENT_DATE - l.last_purchase_date) AS days_since_last_purchase
    FROM customer_value v
    JOIN last_order l USING (customer_id)
)
SELECT
    customer_id,
    total_spent,
    last_purchase_date,
    days_since_last_purchase
FROM combined
WHERE days_since_last_purchase >= 60
ORDER BY total_spent DESC;
