# Gợi ý sản phẩm cho khách thành viên (Đề D): học máy và học sâu
 
Pipeline đọc dữ liệu từ PostgreSQL, huấn luyện và so sánh nhiều mô hình gợi ý, đánh giá theo thời gian, và phục vụ gợi ý qua API.
 
- Bài toán, nhãn, cách chia dữ liệu: `reports/problem-statement.md`
- Kết quả đánh giá chi tiết: `reports/danh-gia-mo-hinh.md` (sinh bằng `python -m src.report`)
## Mô hình được so sánh (cùng tập kiểm tra, cùng metric)
 
| Nhóm | Mô hình | Ghi chú |
| --- | --- | --- |
| Baseline | `PopGlobal`, `PopSegment` | top bán chạy toàn chuỗi; theo giới tính của bé và mùa |
| Lọc cộng tác | `ItemKNN`, `ALS` (implicit; SVD nếu không cài được) | giảm trọng số theo thời gian |
| Boosting | `LGBMRanker` (LightGBM LambdaRank) | xếp hạng lại ứng viên bằng thuộc tính sản phẩm, khách, giới tính, size, mùa; tối ưu bằng Optuna |
| Deep learning | `TwoTower` (PyTorch) | tháp khách và tháp sản phẩm có thuộc tính; softmax mẫu âm trong batch |
 
## Kết quả chính (tập test, 5.263 khách, không lọc size)
 
| Mô hình | Precision@10 | Recall@10 | NDCG@10 | Hit rate@10 |
| --- | --- | --- | --- | --- |
| PopGlobal | 0,0073 | 0,0335 | 0,0201 | 6,9% |
| PopSegment | 0,0113 | 0,0564 | 0,0332 | 10,7% |
| ItemKNN (k=50, tau=120) | 0,0073 | 0,0279 | 0,0183 | 6,8% |
| ALS (64 yếu tố) | 0,0083 | 0,0340 | 0,0211 | 7,8% |
| TwoTower | 0,0064 | 0,0278 | 0,0165 | 6,0% |
| **LGBMRanker** | **0,0150** | **0,0681** | **0,0415** | **13,7%** |
 
LGBMRanker hơn mọi mô hình khác có ý nghĩa thống kê (bootstrap 95%), nhưng chỉ hơn `PopSegment` khoảng 0,012 Recall. Two-Tower thua cả hai baseline. Xem `reports/danh-gia-mo-hinh.md` và `reports/ket_qua_*.csv`.
 
## Cấu trúc
 
```text
models/
  README.md  requirements.txt
  app.py               giao diện demo Streamlit (kết quả, phân tích lỗi, demo gợi ý trực tiếp)
  src/
    config.py          tham số thí nghiệm (ngày chốt, K, lọc size, ...)
    db.py              kết nối PostgreSQL (đọc .env)
    features.py        SQL đọc dữ liệu, loại mã không phải hàng hóa, size -> tuổi, feature mẫu/khách
    split.py           chia theo thời gian, dựng ngữ cảnh cho một ngày chốt, kiểm tra rò rỉ
    metrics.py         Precision/Recall/NDCG/Hit Rate/coverage, bootstrap
    models.py          các mô hình
    train.py           tune -> final -> deploy, log MLflow
    evaluate.py        bootstrap, phân tích lỗi, SHAP, ví dụ cho 3 khách
    report.py          sinh reports/danh-gia-mo-hinh.md
    recommend.py       nạp bundle và gợi ý top-K
    predict_batch.py   top-10 cho mọi khách mua >= 2 lần -> CSV
  api/main.py          FastAPI
  tests/test_leakage.py
```
 
## Cài đặt và chạy lại toàn bộ pipeline
 
```bash
cd models
pip install -r requirements.txt
# Tạo file .env (không commit): PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD, PGSCHEMA=masked_data
 
pytest -q                                   # 1. kiểm tra chống rò rỉ dữ liệu
python -m src.train --stage all             # 2. đọc DB, chọn mô hình, đánh giá test, đóng gói  (--quick để chạy thử nhanh)
python -m src.evaluate                      # 3. khoảng tin cậy, phân tích lỗi, SHAP, ví dụ 3 khách
python -m src.report                        # 4. sinh reports/danh-gia-mo-hinh.md
python -m src.predict_batch                 # 5. reports/goi_y_top10.csv
uvicorn api.main:app --port 8000            # 6. API
python -m streamlit run app.py              # 7. giao diện demo (cần artifacts/bundle.pkl và reports/)
```
 
Lần đầu `src.train` đọc PostgreSQL và lưu cache vào `data/`; thêm `--no-cache` để đọc lại. Xem thí nghiệm bằng MLflow:
 
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
 
Kết quả tái lập được: cố định seed (`config.SEED = 42`), ngày chốt và lưới tham số nằm trong `src/config.py` và `src/train.py`.
 
Nếu Windows báo `An Application Control policy has blocked this file` khi chạy `streamlit.exe`, dùng `python -m streamlit run app.py` như trên (hoặc chạy trong WSL2).
 
## API
 
```text
GET /recommend?customer_id=C_000123&k=10
GET /health
```
 
Khách chưa có lịch sử nhận top bán chạy toàn chuỗi (`source = fallback_popularity`). Mặc định có lọc size; thêm `&size_filter=false` để tắt.
 
## Chống rò rỉ dữ liệu: cách làm và cách kiểm tra
 
**Cách làm**
 
1. Ngày chốt chung cho mọi khách. Huấn luyện, feature, ma trận tương tác, độ bán chạy, tuổi ước tính của bé đều chỉ dùng giao dịch trước ngày chốt.
2. Nhãn là hóa đơn cuối cùng của khách trong kỳ kiểm tra (sau ngày chốt).
3. Mô hình xếp hạng LightGBM học từ nhãn của các fold validation (đều trước ngày chốt test); feature tại ngày test được tính lại chỉ từ dữ liệu trước ngày test.
4. Chọn mô hình và tham số trên validation; test chỉ chạy một lần.
5. Mã không phải hàng hóa và danh mục mẫu "còn bán" cũng chỉ xác định bằng dữ liệu trước ngày chốt.
**Cách kiểm tra**
 
- `split.assert_no_leakage`: chạy tự động mỗi lần dựng tập dữ liệu (train hoàn toàn trước ngày chốt; khách kiểm tra phải có lịch sử trước ngày chốt).
- `features.item_features` và `features.user_features` có `assert` từ chối đầu vào chứa dòng từ ngày chốt trở đi.
- `tests/test_leakage.py`: thêm dữ liệu tương lai cực đoan (mọi khách mua nhiều mẫu sau ngày chốt) hoặc đảo toàn bộ giao dịch sau ngày chốt, rồi xác nhận feature không đổi.
Hạn chế còn lại: danh sách mã không phải hàng hóa được rà bằng thống kê trên toàn bộ năm (chỉ dùng để loại mã, không ảnh hưởng nhãn), và các thuộc tính danh mục sản phẩm (giá, ngành hàng) là thuộc tính tĩnh của bảng `item`.
 
## Lọc size và hàng ngừng bán
 
- `customer.birthday` thiếu 96% nên **tuổi của bé được suy ra từ size đã mua gần đây** (2 hóa đơn gần nhất, cộng thời gian đã trôi qua). Mẫu bị loại nếu khoảng tuổi theo size của mẫu lệch quá xa tuổi ước tính (`SIZE_FILTER_LO/HI` trong `config.py`). Thiếu thông tin thì không loại.
- Quy đổi tên size sang tuổi nằm ở `features.SIZE_RULES`. Các size chưa nhận ra được liệt kê trong `reports/size_chua_quy_doi.csv`; bổ sung quy tắc rồi chạy lại.
- `item.first_date` và `item.lated_date` trống 100% nên "còn bán" được xác định bằng giao dịch trong 60 ngày trước ngày chốt.
## Giới hạn đã biết
 
- Bộ lọc size gần như không đổi kết quả vì hơn 8.000 SKU chưa quy đổi được về tuổi (`(blank)`, `Free size`, size em bé như `12-18M-10-11kg`, size giày `K3`–`K6`); xem `reports/size_chua_quy_doi.csv` và bổ sung `features.SIZE_RULES`.
- Tham số `reg` của ALS cho kết quả giống hệt ở hai mức 0,05 và 0,5: kiểm tra xem tham số có được truyền đúng hoặc pipeline đang dùng SVD dự phòng.
- Mã `97101124` (giá trung bình khoảng 9.900đ, bán 1.508 cái) có thể là túi hoặc quà tặng; nếu đúng, thêm vào `NON_MERCH_MANUAL` trong `config.py` rồi chạy lại.
- Hai đặc trưng `is_boy_item`, `is_girl_item` có gain bằng 0, có thể bỏ.
## Docker (không bắt buộc)
 
Chưa đóng gói Docker. Có thể thêm `Dockerfile` chạy `uvicorn api.main:app` sau khi có `artifacts/bundle.pkl`.
 
