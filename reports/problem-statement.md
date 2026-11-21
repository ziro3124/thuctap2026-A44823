# Problem statement: Gợi ý sản phẩm cho khách thành viên (Đề D, tuần 6)

## 1. Bài toán và giá trị kinh doanh

Với mỗi khách thành viên, hệ thống đề xuất **10 mẫu sản phẩm (item)** khách có khả năng mua ở lần mua kế tiếp. Kết quả dùng để gửi gợi ý cá nhân hóa (tin nhắn, ứng dụng, tư vấn tại cửa hàng) nhằm **tăng số sản phẩm mỗi đơn**: UPT hiện chỉ 1,63 ở kênh online và 3,58 ở cửa hàng.

Khách thành viên tạo 53,2% doanh thu bán (EDA tuần 5), nhưng chỉ 20,9% khách mua từ hai ngày trở lên và tỷ lệ quay lại ở tháng kế tiếp khoảng 4,3%. Vì vậy gợi ý phải làm tốt với dữ liệu rất thưa (độ thưa 99,82%, trung bình 4,97 mẫu mỗi khách).

## 2. Dữ liệu

| Thành phần | Nguồn và xử lý |
| --- | --- |
| Tương tác | `invoice_detail` nối `invoice` và `sku`; gộp SKU về mẫu qua `sku.item_no`; mỗi dòng là (khách, mẫu, hóa đơn) với số lượng và giá trị |
| Lọc | Hóa đơn hợp lệ và không bị hủy; số lượng > 0; chỉ khách có mã; năm 2025 |
| Loại mã không phải hàng hóa | Túi đựng hàng, túi giấy, quà tặng, voucher (doanh thu ≤ 0, giá trung bình < 5.000đ, hoặc không phải dòng `item`) |
| Thuộc tính sản phẩm | `item`: ngành hàng, giới tính, mùa, bản quyền, giá; `sku`: size |
| Phản hồi ngầm | Có mua / số lượng đã mua (log), trọng số giảm dần theo thời gian |

Phát hiện quan trọng: nếu không loại các mã túi/quà (có mặt trong 5–25% hóa đơn), baseline "bán chạy" đạt Hit Rate@10 ≈ 0,88 một cách giả tạo; sau khi loại còn khoảng 0,10.

## 3. Nhãn và cách chia dữ liệu

- **Nhãn:** các mẫu trong **hóa đơn cuối cùng** của khách. Chỉ đánh giá khách mua từ 2 lần trở lên (1 lần trước ngày chốt làm lịch sử, 1 lần là nhãn).
- **Ngày chốt chung** cho mọi khách để không dùng hành vi tương lai của người khác. Huấn luyện chỉ dùng dữ liệu trước ngày chốt.
- **Validation (chọn mô hình, CV theo thời gian):** hai fold, ngày chốt 01/07/2025 và 01/09/2025, mỗi fold dài hai tháng.
- **Test (chạy một lần):** ngày chốt 01/11/2025, kỳ kiểm tra tháng 11–12/2025.
- **Mẫu được gợi ý:** có giao dịch trong 60 ngày trước ngày chốt (proxy cho hàng còn bán, vì `first_date`/`lated_date` của bảng `item` trống 100%); loại mẫu khách đã mua (chỉ 2,7% cặp là mua lại).

## 4. Metric

Precision@10, Recall@10, NDCG@10 (chính), Hit Rate@10, và **độ phủ danh mục** (tỷ lệ mẫu từng được gợi ý) để biết mô hình có chỉ lặp lại vài mẫu bán chạy không. Recall@10 bị chặn trên bởi **trần Recall**: phần nhãn thuộc các mẫu còn gợi ý được.

## 5. Baseline

1. Top bán chạy toàn chuỗi trong 30 ngày (số người mua khác nhau).
2. Top bán chạy theo giới tính của bé (suy ra từ lịch sử mua) và theo mùa (mẫu trái mùa bị giảm điểm).

## 6. Mô hình sẽ so sánh

Item-kNN, ALS (implicit), mô hình xếp hạng LightGBM (LambdaRank) dùng thuộc tính sản phẩm và khách, và mô hình deep learning two-tower (PyTorch). Chọn trên validation, đánh giá một lần trên test, so sánh bằng bootstrap ghép cặp.

## 7. Ràng buộc nghiệp vụ

- **Không gợi ý sai size:** tuổi của bé được suy ra từ size đã mua gần đây (`customer.birthday` thiếu 96%); mẫu có khoảng size lệch quá xa tuổi ước tính bị loại. Quy đổi size sang tuổi là xấp xỉ.
- **Lọc hàng ngừng bán:** dùng giao dịch trong 60 ngày gần nhất làm dấu hiệu còn bán.

## 8. Rủi ro

Khách mới chưa có lịch sử (chiếm đa số khách mua trong mỗi tháng) chỉ có thể nhận gợi ý theo độ phổ biến; dữ liệu một năm nên mùa vụ chỉ quan sát được một lần; kết quả offline chưa chứng minh tác động lên doanh số, cần thử nghiệm A/B.
