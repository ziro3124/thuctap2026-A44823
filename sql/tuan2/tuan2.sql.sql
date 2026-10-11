--------------------------------------------------------------------------------
-- TUẦN 2: CƠ BẢN VỀ TRUY VẤN DỮ LIỆU, LỌC VÀ TỔNG HỢP (GROUP BY)
-- Đảm bảo xử lý đúng kiểu dữ liệu (text/boolean) nếu cần thiết trên schema masked_data
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- CÂU 1: Thống kê tổng số lượng hóa đơn và doanh thu theo từng kênh bán hàng (sale_channel)
--------------------------------------------------------------------------------
SELECT 
    sale_channel,
    COUNT(invoice_id) AS total_invoices,
    SUM(CASE WHEN is_revenue_valid::boolean = true AND is_cancelled::boolean = false THEN actual_amount::numeric ELSE 0 END) AS total_revenue
FROM masked_data.invoice
GROUP BY sale_channel
ORDER BY total_revenue DESC;

--------------------------------------------------------------------------------
-- CÂU 2: Thống kê top 5 khu vực (area) có doanh thu cao nhất từ các cửa hàng Retail
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
-- CÂU 3: Thống kê số lượng khách hàng theo từng phân hạng thành viên (member_level_name)
--------------------------------------------------------------------------------
SELECT 
    member_level_name,
    COUNT(customer_id) AS customer_count
FROM masked_data.customer
WHERE customer_id IS NOT NULL AND customer_id <> ''
GROUP BY member_level_name
ORDER BY customer_count DESC;

--------------------------------------------------------------------------------
-- CÂU 4: Doanh thu theo từng tháng của toàn bộ hệ thống
--------------------------------------------------------------------------------
SELECT 
    TO_CHAR(invoice_date::timestamp, 'YYYY-MM') AS sales_month,
    COUNT(invoice_id) AS total_orders,
    SUM(CASE WHEN is_revenue_valid::boolean = true AND is_cancelled::boolean = false THEN actual_amount::numeric ELSE 0 END) AS monthly_revenue
FROM masked_data.invoice
GROUP BY TO_CHAR(invoice_date::timestamp, 'YYYY-MM')
ORDER BY sales_month;