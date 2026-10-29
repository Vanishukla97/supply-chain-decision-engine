
-- Investigate whether First Class's 100% late rate
-- is consistent across the historical dataset.

SELECT
    shipping_mode,
    DATE_TRUNC('year', order_date::timestamp)::date AS order_year,
    COUNT(*) AS total_orders,
    SUM(is_late) AS late_orders,
    ROUND(100.0 * AVG(is_late), 2) AS late_rate_pct,
    ROUND(AVG(scheduled_shipping_days)::numeric, 2)
        AS avg_scheduled_days
FROM public.orders
GROUP BY
    shipping_mode,
    DATE_TRUNC('year', order_date::timestamp)::date
ORDER BY
    shipping_mode,
    order_year;
