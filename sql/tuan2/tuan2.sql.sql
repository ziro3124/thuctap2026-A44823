--------------------------------------------------------------------------------
-- TUẦN 2: 
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- CÂU 1: Liệt kê danh sách 100 cửa hàng (branch) đầu tiên thuộc kênh 'Retail'
--------------------------------------------------------------------------------
SELECT 
    location_code,
    location_name,
    area,
    channel,
    city
FROM masked_data.branch
WHERE channel = 'Retail'
LIMIT 100;

--------------------------------------------------------------------------------
-- CÂU 2: Thống kê tổng số lượng hóa đơn và doanh thu theo từng kênh bán hàng (sale_channel)
--------------------------------------------------------------------------------
SELECT 
    sale_channel,
    COUNT(invoice_id) AS total_invoices,
    SUM(CASE WHEN is_revenue_valid::boolean = true AND is_cancelled::boolean = false THEN actual_amount::numeric ELSE 0 END) AS total_revenue
FROM masked_data.invoice
GROUP BY sale_channel
ORDER BY total_revenue DESC;

--------------------------------------------------------------------------------
-- CÂU 3: Thống kê top 5 khu vực (area) có doanh thu cao nhất từ các cửa hàng kênh Retail
--------------------------------------------------------------------------------
SELECT 
    b.area,
    SUM(CASE WHEN i.is_revenue_valid::boolean = true AND i.is_cancelled::boolean = false THEN i.actual_amount::numeric ELSE 0 END) AS retail_revenue
FROM masked_data.invoice i
JOIN masked_data.branch b ON i.branch_code = b.location_code
WHERE b.channel = 'Retail'
GROUP BY b.area
ORDER BY retail_revenue DESC
LIMIT 5;

--------------------------------------------------------------------------------
-- CÂU 4: Thống kê số lượng khách hàng theo từng phân hạng thành viên (member_level_name)
--------------------------------------------------------------------------------
SELECT 
    member_level_name,
    COUNT(customer_id) AS customer_count
FROM masked_data.customer
WHERE customer_id IS NOT NULL AND customer_id <> ''
GROUP BY member_level_name
ORDER BY customer_count DESC;

--------------------------------------------------------------------------------
-- CÂU 5: Thống kê doanh thu và số đơn hàng theo từng tháng của toàn hệ thống
--------------------------------------------------------------------------------
SELECT 
    TO_CHAR(invoice_date::timestamp, 'YYYY-MM') AS sales_month,
    COUNT(invoice_id) AS total_orders,
    SUM(CASE WHEN is_revenue_valid::boolean = true AND is_cancelled::boolean = false THEN actual_amount::numeric ELSE 0 END) AS monthly_revenue,
    ROUND(AVG(CASE WHEN is_revenue_valid::boolean = true AND is_cancelled::boolean = false THEN actual_amount::numeric ELSE 0 END), 2) AS average_order_value
FROM masked_data.invoice
GROUP BY TO_CHAR(invoice_date::timestamp, 'YYYY-MM')
ORDER BY sales_month;

--------------------------------------------------------------------------------
-- CÂU 6: Top 10 sản phẩm (item) có tổng số lượng bán ra lớn nhất
--------------------------------------------------------------------------------
SELECT 
    it.item,
    it.product_name,
    it.group_cat,
    SUM(d.quantity::numeric) AS total_quantity_sold
FROM masked_data.invoice_detail d
JOIN masked_data.sku s ON d.item_code = s.sku
JOIN masked_data.item it ON s.item_no = it.item
WHERE d.quantity::numeric > 0
GROUP BY it.item, it.product_name, it.group_cat
ORDER BY total_quantity_sold DESC
LIMIT 10;

--------------------------------------------------------------------------------
-- CÂU 7: Thống kê tỷ lệ đơn hàng có áp dụng mã giảm giá (discount_amount > 0) theo kênh bán
--------------------------------------------------------------------------------
SELECT 
    sale_channel,
    COUNT(*) AS total_orders,
    SUM(CASE WHEN discount_amount::numeric > 0 THEN 1 ELSE 0 END) AS discounted_orders,
    ROUND(SUM(CASE WHEN discount_amount::numeric > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS discount_order_pct
FROM masked_data.invoice
GROUP BY sale_channel;

--------------------------------------------------------------------------------
-- CÂU 8: Tìm kiếm các khách hàng có tổng chi tiêu tích lũy lớn nhất (dựa trên bảng invoice)
--------------------------------------------------------------------------------
SELECT 
    i.customer_id,
    c.member_level_name,
    COUNT(DISTINCT i.invoice_id) AS total_orders,
    SUM(CASE WHEN i.is_revenue_valid::boolean = true AND i.is_cancelled::boolean = false THEN i.actual_amount::numeric ELSE 0 END) AS total_spent
FROM masked_data.invoice i
LEFT JOIN masked_data.customer c ON i.customer_id = c.customer_id
WHERE i.customer_id IS NOT NULL AND i.customer_id <> ''
GROUP BY i.customer_id, c.member_level_name
ORDER BY total_spent DESC
LIMIT 10;

--------------------------------------------------------------------------------
-- CÂU 9: Thống kê doanh thu theo từng nhóm ngành hàng chính (group_cat)
--------------------------------------------------------------------------------
SELECT 
    it.group_cat,
    SUM(d.total_amount::numeric) AS total_group_revenue,
    ROUND(SUM(d.total_amount::numeric) * 100.0 / SUM(SUM(d.total_amount::numeric)) OVER (), 2) AS revenue_share_pct
FROM masked_data.invoice_detail d
JOIN masked_data.sku s ON d.item_code = s.sku
JOIN masked_data.item it ON s.item_no = it.item
GROUP BY it.group_cat
ORDER BY total_group_revenue DESC;

--------------------------------------------------------------------------------
-- CÂU 10: Thống kê số lượng cửa hàng theo từng Tỉnh/Thành phố (city)
--------------------------------------------------------------------------------
SELECT 
    city,
    COUNT(location_code) AS branch_count
FROM masked_data.branch
WHERE city IS NOT NULL AND city <> ''
GROUP BY city
ORDER BY branch_count DESC;

--------------------------------------------------------------------------------
-- CÂU 11: Liệt kê các hóa đơn có giá trị thực tế (actual_amount) lớn hơn 5,000,000
--------------------------------------------------------------------------------
SELECT 
    invoice_id,
    customer_id,
    branch_code,
    actual_amount::numeric AS actual_amount,
    invoice_date
FROM masked_data.invoice
WHERE actual_amount::numeric > 5000000
  AND is_revenue_valid::boolean = true 
  AND is_cancelled::boolean = false
ORDER BY actual_amount::numeric DESC
LIMIT 50;

--------------------------------------------------------------------------------
-- CÂU 12: Thống kê số lượng hóa đơn bị hủy (is_cancelled = true) theo từng kênh bán hàng
--------------------------------------------------------------------------------
SELECT 
    sale_channel,
    COUNT(*) AS total_invoices,
    SUM(CASE WHEN is_cancelled::boolean = true THEN 1 ELSE 0 END) AS cancelled_invoices,
    ROUND(SUM(CASE WHEN is_cancelled::boolean = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS cancellation_rate_pct
FROM masked_data.invoice
GROUP BY sale_channel;

--------------------------------------------------------------------------------
-- CÂU 13: Thống kê tổng số điểm tích lũy (total_point) của khách hàng theo từng phân hạng
--------------------------------------------------------------------------------
SELECT 
    c.member_level_name,
    COUNT(c.customer_id) AS total_customers,
    SUM(cp.total_point::numeric) AS total_points_accumulated,
    ROUND(AVG(cp.total_point::numeric), 2) AS avg_points_per_customer
FROM masked_data.customer c
JOIN masked_data.customer_points cp ON c.customer_id = cp.customer_id
WHERE c.customer_id IS NOT NULL AND c.customer_id <> ''
GROUP BY c.member_level_name
ORDER BY total_points_accumulated DESC;

--------------------------------------------------------------------------------
-- CÂU 14: Top 10 SKU có doanh thu cao nhất trong chi tiết hóa đơn
--------------------------------------------------------------------------------
SELECT 
    d.item_code AS sku,
    SUM(d.quantity::numeric) AS total_qty,
    SUM(d.total_amount::numeric) AS total_revenue
FROM masked_data.invoice_detail d
GROUP BY d.item_code
ORDER BY total_revenue DESC
LIMIT 10;

--------------------------------------------------------------------------------
-- CÂU 15: Thống kê số lượng hóa đơn theo từng hình thức thanh toán COD (is_cod)
--------------------------------------------------------------------------------
SELECT 
    CASE WHEN is_cod::boolean = true THEN 'COD (Thanh toán khi nhận hàng)' ELSE 'Non-COD (Thanh toán trước)' ENDxis_cod_status,
    COUNT(*) AS total_orders,
    SUM(actual_amount::numeric) AS total_amount
FROM masked_data.invoice
WHERE is_revenue_valid::boolean = true AND is_cancelled::boolean = false
GROUP BY is_cod::boolean;

--------------------------------------------------------------------------------
-- CÂU 16: Thống kê doanh thu theo từng năm dựa vào invoice_date
--------------------------------------------------------------------------------
SELECT 
    TO_CHAR(invoice_date::timestamp, 'YYYY') AS sales_year,
    COUNT(invoice_id) AS total_orders,
    SUM(actual_amount::numeric) AS yearly_revenue
FROM masked_data.invoice
WHERE is_revenue_valid::boolean = true AND is_cancelled::boolean = false
GROUP BY TO_CHAR(invoice_date::timestamp, 'YYYY')
ORDER BY sales_year;

--------------------------------------------------------------------------------
-- CÂU 17: Tìm danh sách các mặt hàng (item) có giá bán trung bình (unit_price) cao nhất
--------------------------------------------------------------------------------
SELECT 
    it.item,
    it.product_name,
    it.group_cat,
    ROUND(AVG(d.unit_price::numeric), 2) AS avg_unit_price
FROM masked_data.invoice_detail d
JOIN masked_data.sku s ON d.item_code = s.sku
JOIN masked_data.item it ON s.item_no = it.item
WHERE d.unit_price::numeric > 0
GROUP BY it.item, it.product_name, it.group_cat
ORDER BY avg_unit_price DESC
LIMIT 10;

--------------------------------------------------------------------------------
-- CÂU 18: Thống kê số lượng cửa hàng theo từng kênh (channel) và khu vực (area)
--------------------------------------------------------------------------------
SELECT 
    channel,
    area,
    COUNT(location_code) AS branch_count
FROM masked_data.branch
GROUP BY channel, area
ORDER BY channel, branch_count DESC;

--------------------------------------------------------------------------------
-- CÂU 19: Thống kê tổng số lượng sản phẩm bị trả lại (quantity < 0) theo từng nhóm ngành hàng
--------------------------------------------------------------------------------
SELECT 
    it.group_cat,
    SUM(ABS(d.quantity::numeric)) AS total_returned_qty
FROM masked_data.invoice_detail d
JOIN masked_data.sku s ON d.item_code = s.sku
JOIN masked_data.item it ON s.item_no = it.item
WHERE d.quantity::numeric < 0
GROUP BY it.group_cat
ORDER BY total_returned_qty DESC;

--------------------------------------------------------------------------------
-- CÂU 20: Thống kê khách hàng có số lượng hóa đơn mua sắm nhiều nhất trong hệ thống
--------------------------------------------------------------------------------
SELECT 
    customer_id,
    COUNT(invoice_id) AS total_purchases,
    SUM(actual_amount::numeric) AS total_spent
FROM masked_data.invoice
WHERE customer_id IS NOT NULL AND customer_id <> ''
  AND is_revenue_valid::boolean = true AND is_cancelled::boolean = false
GROUP BY customer_id
ORDER BY total_purchases DESC
LIMIT 10;
