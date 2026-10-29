
-- Supply Chain Decision Engine
-- Analytics Layer: Initial Business KPIs

-- 1. Overall delivery performance
SELECT
    COUNT(*) AS total_orders,
    SUM(is_late) AS late_orders,
    COUNT(*) - SUM(is_late) AS non_late_orders,
    ROUND(100.0 * AVG(is_late), 2) AS late_rate_pct
FROM public.orders;

-- 2. Delivery performance by shipping mode
SELECT
    shipping_mode,
    COUNT(*) AS total_orders,
    SUM(is_late) AS late_orders,
    ROUND(100.0 * AVG(is_late), 2) AS late_rate_pct,
    ROUND(AVG(order_value)::numeric, 2) AS avg_order_value
FROM public.orders
GROUP BY shipping_mode
ORDER BY late_rate_pct DESC;

-- 3. Delivery performance by market
SELECT
    market,
    COUNT(*) AS total_orders,
    SUM(is_late) AS late_orders,
    ROUND(100.0 * AVG(is_late), 2) AS late_rate_pct,
    ROUND(AVG(order_value)::numeric, 2) AS avg_order_value
FROM public.orders
GROUP BY market
ORDER BY late_rate_pct DESC;

-- 4. Delivery performance by customer segment
SELECT
    customer_segment,
    COUNT(*) AS total_orders,
    SUM(is_late) AS late_orders,
    ROUND(100.0 * AVG(is_late), 2) AS late_rate_pct
FROM public.orders
GROUP BY customer_segment
ORDER BY late_rate_pct DESC;

-- 5. Monthly delivery performance
SELECT
    DATE_TRUNC('month', order_date::timestamp)::date AS order_month,
    COUNT(*) AS total_orders,
    SUM(is_late) AS late_orders,
    ROUND(100.0 * AVG(is_late), 2) AS late_rate_pct,
    ROUND(SUM(order_value)::numeric, 2) AS total_order_value
FROM public.orders
GROUP BY DATE_TRUNC('month', order_date::timestamp)
ORDER BY order_month;
