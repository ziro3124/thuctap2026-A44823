--------------------------------------------------------------------------------
-- TUẦN 3: SQL NÂNG CAO, WINDOW FUNCTION, COHORT & NGHIỆP VỤ
-- Đã cập nhật sửa lỗi cú pháp Boolean, lỗi ép kiểu ngày tháng và xử lý quyền View
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- CÂU 1: Doanh thu tháng này tăng hay giảm bao nhiêu % so với tháng trước (dùng LAG)
--------------------------------------------------------------------------------
WITH monthly_revenue AS (
    SELECT 
        TO_CHAR(invoice_date::timestamp, 'YYYY-MM') AS month,
        SUM(CASE WHEN is_revenue_valid::boolean = true AND is_cancelled::boolean = false THEN actual_amount::numeric ELSE 0 END) AS revenue
    FROM masked_data.invoice
    GROUP BY TO_CHAR(invoice_date::timestamp, 'YYYY-MM')
)
SELECT 
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_month_revenue,
    ROUND((revenue - LAG(revenue) OVER (ORDER BY month)) * 100.0 / NULLIF(LAG(revenue) OVER (ORDER BY month), 0), 2) AS mom_growth_pct
FROM monthly_revenue
ORDER BY month;

--------------------------------------------------------------------------------
-- CÂU 2: Xếp hạng cửa hàng Retail theo doanh thu trong mỗi area bằng RANK()
--------------------------------------------------------------------------------
WITH store_revenue AS (
    SELECT 
        b.area,
        b.location_name,
        SUM(CASE WHEN i.is_revenue_valid::boolean = true AND i.is_cancelled::boolean = false THEN i.actual_amount::numeric ELSE 0 END) AS revenue
    FROM masked_data.invoice i
    JOIN masked_data.branch b ON i.branch_code = b.location_code
    WHERE b.channel = 'Retail'
    GROUP BY b.area, b.location_name
)
SELECT 
    area,
    location_name,
    revenue,
    RANK() OVER (PARTITION BY area ORDER BY revenue DESC) AS rank_in_area
FROM store_revenue
ORDER BY area, rank_in_area;

--------------------------------------------------------------------------------
-- CÂU 3: Top 3 sản phẩm (item) theo doanh thu trong mỗi group_cat bằng ROW_NUMBER()
--------------------------------------------------------------------------------
WITH item_revenue AS (
    SELECT 
        it.group_cat,
        it.item,
        it.product_name,
        SUM(d.total_amount::numeric) AS revenue
    FROM masked_data.invoice_detail d
    JOIN masked_data.sku s ON d.item_code = s.sku
    JOIN masked_data.item it ON s.item_no = it.item
    GROUP BY it.group_cat, it.item, it.product_name
),
ranked_items AS (
    SELECT 
        group_cat,
        item,
        product_name,
        revenue,
        ROW_NUMBER() OVER (PARTITION BY group_cat ORDER BY revenue DESC) AS rn
    FROM item_revenue
)
SELECT group_cat, item, product_name, revenue
FROM ranked_items
WHERE rn <= 3
ORDER BY group_cat, rn;

--------------------------------------------------------------------------------
-- CÂU 4: Doanh thu lũy kế theo ngày trong năm (Running Total)
--------------------------------------------------------------------------------
WITH daily_rev AS (
    SELECT 
        invoice_date::date AS dt,
        SUM(CASE WHEN is_revenue_valid::boolean = true AND is_cancelled::boolean = false THEN actual_amount::numeric ELSE 0 END) AS daily_revenue
    FROM masked_data.invoice
    GROUP BY invoice_date::date
)
SELECT 
    dt,
    daily_revenue,
    SUM(daily_revenue) OVER (ORDER BY dt ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_revenue
FROM daily_rev
ORDER BY dt;

--------------------------------------------------------------------------------
-- CÂU 5: Trung bình động 7 ngày của doanh thu toàn chuỗi (Moving Average 7 days)
--------------------------------------------------------------------------------
WITH daily_rev AS (
    SELECT 
        invoice_date::date AS dt,
        SUM(CASE WHEN is_revenue_valid::boolean = true AND is_cancelled::boolean = false THEN actual_amount::numeric ELSE 0 END) AS daily_revenue
    FROM masked_data.invoice
    GROUP BY invoice_date::date
)
SELECT 
    dt,
    daily_revenue,
    ROUND(AVG(daily_revenue) OVER (ORDER BY dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS ma_7d
FROM daily_rev
ORDER BY dt;

--------------------------------------------------------------------------------
-- CÂU 6: Truy vấn tóm tắt thông tin khách hàng (Dùng để chạy SELECT thay vì CREATE VIEW nếu thiếu quyền)
--------------------------------------------------------------------------------
SELECT 
    c.customer_id,
    c.member_level_name,
    cp.total_point,
    cp.total_amount AS lifetime_spending_recorded,
    COUNT(i.invoice_id) AS total_invoices,
    SUM(CASE WHEN i.is_revenue_valid::boolean = true AND i.is_cancelled::boolean = false THEN i.actual_amount::numeric ELSE 0 END) AS total_spent,
    MIN(i.invoice_date::timestamp) AS first_purchase_date,
    MAX(i.invoice_date::timestamp) AS last_purchase_date
FROM masked_data.customer c
LEFT JOIN masked_data.customer_points cp ON c.customer_id = cp.customer_id
LEFT JOIN masked_data.invoice i ON c.customer_id = i.customer_id
WHERE c.customer_id IS NOT NULL AND c.customer_id <> ''
GROUP BY c.customer_id, c.member_level_name, cp.total_point, cp.total_amount;

--------------------------------------------------------------------------------
-- CÂU 7: Tỷ lệ khách quay lại mua lần 2
--------------------------------------------------------------------------------
WITH customer_orders AS (
    SELECT 
        customer_id,
        COUNT(DISTINCT invoice_id) AS order_count
    FROM masked_data.invoice
    WHERE customer_id IS NOT NULL AND customer_id <> ''
      AND is_revenue_valid::boolean = true AND is_cancelled::boolean = false
    GROUP BY customer_id
),
order_distribution AS (
    SELECT 
        order_count,
        COUNT(customer_id) AS num_customers
    FROM customer_orders
    GROUP BY order_count
)
SELECT 
    order_count,
    num_customers,
    ROUND(num_customers * 100.0 / SUM(num_customers) OVER (), 2) AS percentage
FROM order_distribution
ORDER BY order_count;

--------------------------------------------------------------------------------
-- CÂU 8: Khoảng cách trung bình (ngày) giữa hai lần mua liên tiếp theo hạng thành viên
--------------------------------------------------------------------------------
WITH ordered_purchases AS (
    SELECT 
        i.customer_id,
        c.member_level_name,
        i.invoice_date::timestamp AS purchase_time,
        LAG(i.invoice_date::timestamp) OVER (PARTITION BY i.customer_id ORDER BY i.invoice_date::timestamp) AS prev_purchase_time
    FROM masked_data.invoice i
    JOIN masked_data.customer c ON i.customer_id = c.customer_id
    WHERE i.customer_id IS NOT NULL AND i.customer_id <> ''
      AND i.is_revenue_valid::boolean = true AND i.is_cancelled::boolean = false
),
diffs AS (
    SELECT 
        member_level_name,
        EXTRACT(EPOCH FROM (purchase_time - prev_purchase_time)) / 86400 AS days_between
    FROM ordered_purchases
    WHERE prev_purchase_time IS NOT NULL
)
SELECT 
    member_level_name,
    ROUND(AVG(days_between)::numeric, 2) AS avg_days_between_purchases
FROM diffs
GROUP BY member_level_name;

--------------------------------------------------------------------------------
-- CÂU 9: Cohort Analysis (Đã sửa lỗi định dạng ngày tháng)
--------------------------------------------------------------------------------
WITH first_purchase AS (
    SELECT 
        customer_id,
        MIN(TO_CHAR(invoice_date::timestamp, 'YYYY-MM')) AS cohort_month
    FROM masked_data.invoice
    WHERE customer_id IS NOT NULL AND customer_id <> ''
    GROUP BY customer_id
),
customer_activity AS (
    SELECT DISTINCT
        fp.customer_id,
        fp.cohort_month,
        TO_CHAR(i.invoice_date::timestamp, 'YYYY-MM') AS activity_month
    FROM first_purchase fp
    JOIN masked_data.invoice i ON fp.customer_id = i.customer_id
    WHERE i.is_revenue_valid::boolean = true AND i.is_cancelled::boolean = false
),
cohort_indexed AS (
    SELECT 
        cohort_month,
        customer_id,
        (EXTRACT(YEAR FROM (activity_month || '-01')::date) - EXTRACT(YEAR FROM (cohort_month || '-01')::date)) * 12 +
        (EXTRACT(MONTH FROM (activity_month || '-01')::date) - EXTRACT(MONTH FROM (cohort_month || '-01')::date)) AS month_number
    FROM customer_activity
)
SELECT 
    cohort_month,
    COUNT(DISTINCT CASE WHEN month_number = 0 THEN customer_id END) AS m0_size,
    COUNT(DISTINCT CASE WHEN month_number = 1 THEN customer_id END) AS m1_active,
    COUNT(DISTINCT CASE WHEN month_number = 2 THEN customer_id END) AS m2_active,
    COUNT(DISTINCT CASE WHEN month_number = 3 THEN customer_id END) AS m3_active
FROM cohort_indexed
GROUP BY cohort_month
ORDER BY cohort_month;

--------------------------------------------------------------------------------
-- CÂU 10: Chấm điểm RFM (Dùng NTILE(5))
--------------------------------------------------------------------------------
WITH rfm_base AS (
    SELECT 
        customer_id,
        EXTRACT(DAY FROM ('2025-12-31'::timestamp - MAX(invoice_date::timestamp))) AS recency,
        COUNT(DISTINCT invoice_id) AS frequency,
        SUM(actual_amount::numeric) AS monetary
    FROM masked_data.invoice
    WHERE customer_id IS NOT NULL AND customer_id <> ''
      AND is_revenue_valid::boolean = true AND is_cancelled::boolean = false
      AND invoice_date::timestamp <= '2025-12-31 23:59:59'
    GROUP BY customer_id
),
rfm_scores AS (
    SELECT 
        customer_id,
        NTILE(5) OVER (ORDER BY recency DESC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,
        NTILE(5) OVER (ORDER BY monetary ASC) AS m_score
    FROM rfm_base
)
SELECT 
    r_score,
    f_score,
    m_score,
    COUNT(customer_id) AS customer_count
FROM rfm_scores
GROUP BY r_score, f_score, m_score
ORDER BY r_score DESC, f_score DESC, m_score DESC;

--------------------------------------------------------------------------------
-- CÂU 11: Tỷ lệ trả hàng theo group_cat và size
--------------------------------------------------------------------------------
SELECT 
    it.group_cat,
    s.size_name,
    SUM(CASE WHEN d.quantity::numeric < 0 THEN ABS(d.quantity::numeric) ELSE 0 END) AS returned_qty,
    SUM(CASE WHEN d.quantity::numeric > 0 THEN d.quantity::numeric ELSE 0 END) AS sold_qty,
    ROUND(
        SUM(CASE WHEN d.quantity::numeric < 0 THEN ABS(d.quantity::numeric) ELSE 0 END) * 100.0 / 
        NULLIF(SUM(CASE WHEN d.quantity::numeric > 0 THEN d.quantity::numeric ELSE 0 END), 0), 2
    ) AS return_rate_pct
FROM masked_data.invoice_detail d
JOIN masked_data.sku s ON d.item_code = s.sku
JOIN masked_data.item it ON s.item_no = it.item
GROUP BY it.group_cat, s.size_name
ORDER BY return_rate_pct DESC;

--------------------------------------------------------------------------------
-- CÂU 12: Chỉ số UPT (Units Per Transaction)
--------------------------------------------------------------------------------
SELECT 
    ROUND(SUM(CASE WHEN quantity::numeric > 0 THEN quantity::numeric ELSE 0 END) * 1.0 / 
    NULLIF(COUNT(DISTINCT invoice_id), 0), 2) AS upt
FROM masked_data.invoice_detail;

--------------------------------------------------------------------------------
-- CÂU 13: So sánh đơn online với đơn tại cửa hàng
--------------------------------------------------------------------------------
SELECT 
    i.sale_channel,
    ROUND(AVG(i.actual_amount::numeric), 2) AS aov,
    ROUND(AVG(i.discount_amount::numeric), 2) AS avg_discount,
    ROUND(SUM(CASE WHEN i.is_cod::boolean = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS cod_pct
FROM masked_data.invoice i
GROUP BY i.sale_channel
ORDER BY aov DESC;

--------------------------------------------------------------------------------
-- CÂU 14: Tỷ lệ doanh thu từ sản phẩm có bản quyền nhân vật (licenses)
--------------------------------------------------------------------------------
SELECT 
    CASE 
        WHEN it.licenses IS NOT NULL AND it.licenses <> '' THEN 'Licensed'
        ELSE 'Normal'
    END AS product_type,
    SUM(d.total_amount::numeric) AS revenue,
    ROUND(SUM(d.total_amount::numeric) * 100.0 / SUM(SUM(d.total_amount::numeric)) OVER (), 2) AS revenue_pct
FROM masked_data.invoice_detail d
JOIN masked_data.sku s ON d.item_code = s.sku
JOIN masked_data.item it ON s.item_no = it.item
GROUP BY CASE 
        WHEN it.licenses IS NOT NULL AND it.licenses <> '' THEN 'Licensed'
        ELSE 'Normal'
    END;

--------------------------------------------------------------------------------
-- CÂU 15: EXPLAIN ANALYZE tối ưu truy vấn doanh thu theo cửa hàng
--------------------------------------------------------------------------------
EXPLAIN ANALYZE
SELECT 
    b.location_name,
    SUM(i.actual_amount::numeric) AS total_revenue
FROM masked_data.invoice i
JOIN masked_data.branch b ON i.branch_code = b.location_code
WHERE i.is_revenue_valid::boolean = true AND i.is_cancelled::boolean = false
GROUP BY b.location_name;
