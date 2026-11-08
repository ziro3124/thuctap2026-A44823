# Thực tập Lập trình AI 2026: Gợi ý sản phẩm cho chuỗi bán lẻ thời trang trẻ em

Repository của chương trình thực tập năm 2026, nhánh **Lập trình AI**, trên dữ liệu giao dịch bán hàng năm 2025 của một chuỗi bán lẻ thời trang trẻ em (PostgreSQL).

**Đề tài:** xây dựng mô hình học máy để gợi ý 10 mẫu sản phẩm (item) cho mỗi khách thành viên có khả năng mua trong tháng tiếp theo, chỉ dựa trên hành vi mua.

## 1. Kết quả chính

Đánh giá trên tập test (tháng 12/2025), huấn luyện bằng dữ liệu đến 30/11, chọn mô hình trên tháng 11.

| Mô hình | Recall@10 khách có lịch sử | Recall@10 khách mới | Recall@10 tất cả |
| --- | --- | --- | --- |
| Baseline: bán chạy 30 ngày | 0,0439 | 0,0612 | 0,0548 |
| Bán chạy theo giới tính của bé | 0,0482 | 0,0582 | 0,0545 |
| **Blend: 30% ItemKNN + 70% bán chạy theo giới tính** | **0,0554** | 0,0582 | 0,0572 |

- Với khách đã có lịch sử, mô hình tốt nhất hơn baseline có ý nghĩa thống kê (Recall@10 tăng khoảng 26%, NDCG@10 khoảng 32%, bootstrap khoảng tin cậy 95%).
- Với khách mới, mô hình không hơn baseline. Khách mới chiếm 63% khách trong kỳ test nên tác động gộp toàn bộ còn nhỏ.
- Độ phủ danh mục tăng từ 1,3% lên 12,8% số mẫu.
- Trần Recall của bài toán là 0,805 (khoảng 19,5% lượt mua thuộc mẫu không thể gợi ý, như mẫu mới ra mắt).
- Phát hiện về dữ liệu: baseline ban đầu có Hit Rate@10 = 0,88 là giả tạo do các mã túi đựng hàng, túi giấy, quà tặng và voucher. Sau khi loại các mã này, Hit Rate@10 còn 0,105.

Chi tiết và hạn chế xem trong báo cáo tuần 6–9 tại thư mục `reports/`.

## 2. Công nghệ

- Python, PostgreSQL (schema `masked_data`)
- pandas, NumPy, SciPy, Matplotlib
- scikit-learn, `implicit` (ALS; nếu không cài được thì tự dùng SVD của scikit-learn)
- SQLAlchemy, psycopg2, python-dotenv, pyarrow

## 3. Cấu trúc repository

```text
sql/                      Câu truy vấn SQL và các view (tuần 2–3)
sql/tuan/                 Ảnh đề bài hoặc ảnh tham khảo câu truy vấn
notebooks/
  01_profiling_cleaning.ipynb          Kiểm tra và làm sạch dữ liệu (tuần 4)
  02_eda.ipynb                         Phân tích khám phá dữ liệu (tuần 5)
  03_1_du_lieu_va_loc_hang.ipynb       Lấy dữ liệu tương tác, loại mã không phải hàng hóa
  03_2_khung_danh_gia_baseline.ipynb   Chia dữ liệu theo thời gian, đo baseline
  03_3_chon_mo_hinh_validation.ipynb   So sánh mô hình, chọn tham số trên validation
  03_4_test_phan_tich_luu_mo_hinh.ipynb  Đánh giá cuối trên test, phân tích, lưu mô hình
  recsys_lib.py                        Lớp mô hình và hàm đánh giá dùng chung cho 03_x
reports/                  Báo cáo, bảng kết quả (CSV) và hình ảnh
models/                   Mô hình đã huấn luyện (không đưa lên repository)
data/                     Cache dữ liệu cục bộ (không đưa lên repository)
```

Các notebook ghi kết quả vào `reports/`, `reports/figs/`, `data/` và `models/` tính từ thư mục đang chạy notebook.

## 4. Cách chạy

1. Tạo môi trường và cài thư viện:

   ```bash
   pip install pandas numpy scipy scikit-learn matplotlib sqlalchemy psycopg2-binary python-dotenv pyarrow implicit jupyter
   ```

   Trên Windows, nếu thư viện biên dịch sẵn (như `implicit`, `pandas`) bị chặn bởi Smart App Control, hãy chạy trong WSL2.

2. Tạo file `.env` ở thư mục gốc (không commit):

   ```text
   PGHOST=...
   PGPORT=5432
   PGDATABASE=...
   PGUSER=...
   PGPASSWORD=...
   PGSCHEMA=masked_data
   ```

3. Chạy lần lượt các notebook: `01` → `02` → `03_1` → `03_2` → `03_3` → `03_4`. Phần 03_1 cần kết nối database, các phần sau đọc dữ liệu đã cache trong `data/`. Phần 03_3 chạy lâu nhất (vài phút đến vài chục phút).

4. Sau 03_1, kiểm tra bảng top mã bán nhiều nhất. Mã nào không phải hàng thật mà quy tắc tự động chưa bắt được thì thêm vào `NON_MERCH_MANUAL` rồi chạy lại.

Lưu ý: các cột trong database đều là kiểu TEXT, các truy vấn đã ép kiểu (`::numeric`, `::timestamp`, `::text::boolean`).

## 5. Phương pháp đánh giá

- **Chia theo thời gian:** huấn luyện đến 31/10, validation tháng 11, test tháng 12/2025 (chạy một lần).
- **Mẫu được gợi ý:** chỉ các mẫu có giao dịch trong 60 ngày cuối của tập huấn luyện; loại các mẫu khách đã mua.
- **Chỉ số:** Recall, Precision, NDCG, MAP, Hit Rate ở K=10; độ phủ danh mục; độ đa dạng ngành hàng.
- **Nhóm khách:** có lịch sử và khách mới (khách mới dùng mô hình riêng dựa trên mẫu khách mới hay mua).
- **Kiểm định:** bootstrap ghép cặp, khoảng tin cậy 95% cho chênh lệch với baseline.

Mô hình đã thử: bán chạy 30 ngày, bán chạy theo giới tính, ItemKNN (cosine, trọng số giảm dần theo thời gian), ALS/SVD, Hybrid với độ phổ biến, Blend với bán chạy theo giới tính.

## 6. Tiến độ

- [x] Tuần 1: thiết lập môi trường, kết nối database và repository.
- [x] Tuần 2–3: hoàn thành bài tập SQL và các view.
- [x] Tuần 4–5: kiểm tra dữ liệu và phân tích khám phá (EDA).
- [x] Tuần 6–9: xây dựng và đánh giá mô hình gợi ý.
- [ ] Tuần 10: hoàn thiện báo cáo và tổng kết.

## 7. Bảo mật

Không đưa lên repository: mật khẩu, file `.env`, dữ liệu bán hàng thô, thông tin truy cập database, thư mục `data/` (chứa dữ liệu mức khách hàng) và file mô hình trong `models/` (chứa danh sách mã khách). Chỉ đưa lên các bảng tổng hợp và hình ảnh trong `reports/`.

Gợi ý `.gitignore`:

```gitignore
.env
data/
models/*.pkl
*.parquet
__pycache__/
.ipynb_checkpoints/
```

## 8. Nhật ký thực tập

Nhật ký từng tuần được lưu tại `sql/tuan-XX.md`.
