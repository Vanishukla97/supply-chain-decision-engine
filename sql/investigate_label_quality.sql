
-- 1. Compare delivery outcomes across shipping modes
SELECT
    shipping_mode,
    delivery_status,
    COUNT(*) AS orders,
    ROUND(100.0 * AVG(is_late), 2) AS late_rate_pct,
    ROUND(AVG(scheduled_shipping_days), 2) AS avg_scheduled_days
FROM public.orders
GROUP BY shipping_mode, delivery_status
ORDER BY shipping_mode, delivery_status;


-- 2. Check whether the First Class pattern changes by year
SELECT
    shipping_mode,
    EXTRACT(YEAR FROM order_date::timestamp) AS order_year,
    COUNT(*) AS total_orders,
    SUM(is_late) AS late_orders,
    ROUND(100.0 * AVG(is_late), 2) AS late_rate_pct
FROM public.orders
GROUP BY
    shipping_mode,
    EXTRACT(YEAR FROM order_date::timestamp)
ORDER BY order_year, shipping_mode;


-- 3. Verify which delivery statuses remain after filtering
SELECT
    delivery_status,
    COUNT(*) AS orders,
    SUM(is_late) AS late_orders
FROM public.orders
GROUP BY delivery_status
ORDER BY orders DESC;
