# Đánh giá mô hình gợi ý sản phẩm (Đề D)

_Bảng số liệu dưới đây được sinh tự động từ `reports/*.csv` bằng `python -m src.report`. Các mục **Phân tích của bạn** để trống cho bạn viết._

## 1. Tóm tắt kết quả

Trên tập test, mô hình tốt nhất là **LGBMRanker** (NDCG@10 = 0.0415, Recall@10 = 0.0681), so với baseline top bán chạy toàn chuỗi (NDCG@10 = 0.0201, Recall@10 = 0.0335): NDCG tăng 106%, Recall tăng 103%. Trần Recall của bài toán là 0.750 (phần nhãn thuộc các mẫu còn có thể gợi ý).

## 2. Thiết kế thí nghiệm

- Chia theo thời gian: validation gồm 2 fold (ngày chốt 2025-07-01 và 2025-09-01), test ngày chốt 2025-11-01 (kỳ 2025-11-01 đến 2026-01-01). Mỗi khách kiểm tra có ≥ 2 lần mua: 1 lần trước ngày chốt, nhãn là hóa đơn cuối cùng trong kỳ.
- Mẫu được gợi ý: có giao dịch trong 60 ngày trước ngày chốt, loại mẫu khách đã mua, loại mã không phải hàng hóa.
- Metric: Precision@10, Recall@10, NDCG@10, Hit Rate@10, coverage danh mục (tỷ lệ mẫu từng được gợi ý).
- Cấu hình chọn trên validation: {"ItemKNN": {"k": 50, "tau": 120}, "ALS": {"factors": 64, "reg": 0.05, "alpha": 40}, "TwoTower": {"dim": 64, "epochs": 30, "tau": 0.1}, "LGBMRanker": {"num_leaves": 17, "learning_rate": 0.06723992631737642, "min_child_samples": 59, "colsample_bytree": 0.9371404720544085, "reg_lambda": 2.0238972113476317, "n_estimators": 54}}

## 3. So sánh mô hình trên validation (fold 2)

| mo_hinh | precision@10 | recall@10 | ndcg@10 | hit_rate@10 | coverage |
| --- | --- | --- | --- | --- | --- |
| LGBMRanker | 0.0220 | 0.0969 | 0.0671 | 0.1964 | 0.2014 |
| PopSegment | 0.0155 | 0.0715 | 0.0491 | 0.1451 | 0.0196 |
| PopGlobal | 0.0114 | 0.0518 | 0.0399 | 0.1088 | 0.0143 |
| TwoTower dim=64 ep=30 | 0.0122 | 0.0542 | 0.0389 | 0.1138 | 0.3560 |
| TwoTower dim=32 ep=30 | 0.0114 | 0.0503 | 0.0382 | 0.1050 | 0.3590 |
| TwoTower dim=32 ep=15 | 0.0118 | 0.0522 | 0.0376 | 0.1109 | 0.2843 |
| ALS f=64 reg=0.05 | 0.0117 | 0.0480 | 0.0311 | 0.1058 | 0.2074 |
| ALS f=64 reg=0.5 | 0.0117 | 0.0480 | 0.0311 | 0.1058 | 0.2074 |
| ItemKNN k=50 tau=120 | 0.0116 | 0.0456 | 0.0301 | 0.1050 | 0.9427 |
| ItemKNN k=20 tau=120 | 0.0099 | 0.0404 | 0.0268 | 0.0909 | 0.9465 |
| ALS f=32 reg=0.05 | 0.0106 | 0.0420 | 0.0267 | 0.0967 | 0.1569 |
| ALS f=32 reg=0.5 | 0.0106 | 0.0420 | 0.0267 | 0.0967 | 0.1569 |
| ItemKNN k=50 tau=60 | 0.0091 | 0.0365 | 0.0222 | 0.0831 | 0.9389 |
| ItemKNN k=20 tau=60 | 0.0073 | 0.0298 | 0.0183 | 0.0673 | 0.9434 |

## 4. Kết quả trên tập test (không lọc size)

| mo_hinh | precision@10 | recall@10 | ndcg@10 | hit_rate@10 | coverage |
| --- | --- | --- | --- | --- | --- |
| LGBMRanker | 0.0150 | 0.0681 | 0.0415 | 0.1368 | 0.1996 |
| PopSegment | 0.0113 | 0.0564 | 0.0332 | 0.1070 | 0.0196 |
| ALS f=64 reg=0.05 | 0.0083 | 0.0340 | 0.0211 | 0.0781 | 0.2045 |
| PopGlobal | 0.0073 | 0.0335 | 0.0201 | 0.0694 | 0.0112 |
| ItemKNN k=50 tau=120 | 0.0073 | 0.0279 | 0.0183 | 0.0682 | 0.9629 |
| TwoTower | 0.0064 | 0.0278 | 0.0165 | 0.0599 | 0.4006 |

### Có lọc size theo tuổi của bé

| mo_hinh | precision@10 | recall@10 | ndcg@10 | hit_rate@10 | coverage |
| --- | --- | --- | --- | --- | --- |
| LGBMRanker + lọc size | 0.0151 | 0.0675 | 0.0420 | 0.1378 | 0.2157 |
| PopSegment + lọc size | 0.0120 | 0.0593 | 0.0356 | 0.1113 | 0.0504 |
| PopGlobal + lọc size | 0.0083 | 0.0370 | 0.0221 | 0.0781 | 0.0322 |
| ALS f=64 reg=0.05 + lọc size | 0.0085 | 0.0349 | 0.0218 | 0.0790 | 0.2528 |
| ItemKNN k=50 tau=120 + lọc size | 0.0074 | 0.0282 | 0.0186 | 0.0694 | 0.9552 |
| TwoTower + lọc size | 0.0065 | 0.0280 | 0.0169 | 0.0610 | 0.4307 |

Lọc size là quy tắc nghiệp vụ (không gợi ý mẫu sai size so với tuổi ước tính của bé); bảng trên cho biết cái giá của quy tắc này đối với metric.

## 5. Mô hình tốt nhất có hơn các mô hình khác không? (bootstrap ghép cặp, khoảng tin cậy 95%)

| mo_hinh_tot_nhat | so_voi | chi_so | chenh_lech | ktc95_thap | ktc95_cao | co_y_nghia |
| --- | --- | --- | --- | --- | --- | --- |
| LGBMRanker | PopGlobal | ndcg@10 | 0.0214 | 0.0180 | 0.0247 | True |
| LGBMRanker | PopGlobal | recall@10 | 0.0346 | 0.0289 | 0.0401 | True |
| LGBMRanker | PopSegment | ndcg@10 | 0.0083 | 0.0051 | 0.0113 | True |
| LGBMRanker | PopSegment | recall@10 | 0.0117 | 0.0062 | 0.0167 | True |
| LGBMRanker | ItemKNN k=50 tau=120 | ndcg@10 | 0.0232 | 0.0197 | 0.0267 | True |
| LGBMRanker | ItemKNN k=50 tau=120 | recall@10 | 0.0402 | 0.0347 | 0.0460 | True |
| LGBMRanker | ALS f=64 reg=0.05 | ndcg@10 | 0.0204 | 0.0168 | 0.0239 | True |
| LGBMRanker | ALS f=64 reg=0.05 | recall@10 | 0.0341 | 0.0283 | 0.0398 | True |
| LGBMRanker | TwoTower | ndcg@10 | 0.0250 | 0.0216 | 0.0285 | True |
| LGBMRanker | TwoTower | recall@10 | 0.0403 | 0.0343 | 0.0463 | True |

## 6. Phân tích lỗi: nhóm nào dự đoán sai nhiều?

| chieu_phan_tich | nhom | so_khach | recall@10 | ndcg@10 | hit_rate@10 |
| --- | --- | --- | --- | --- | --- |
| so_lan_mua_truoc | 1 | 2597 | 0.0679 | 0.0408 | 0.1313 |
| so_lan_mua_truoc | 2 | 1179 | 0.0712 | 0.0422 | 0.1408 |
| so_lan_mua_truoc | 3+ | 1487 | 0.0660 | 0.0420 | 0.1432 |
| gioi_tinh_suy_ra | be trai | 1554 | 0.0674 | 0.0442 | 0.1467 |
| gioi_tinh_suy_ra | be gai | 3420 | 0.0694 | 0.0405 | 0.1333 |
| gioi_tinh_suy_ra | chua ro | 289 | 0.0566 | 0.0388 | 0.1246 |
| tuoi_uoc_tinh | (6, 8] | 1439 | 0.0686 | 0.0411 | 0.1348 |
| tuoi_uoc_tinh | (0, 2] | 54 | 0.0880 | 0.0440 | 0.1111 |
| tuoi_uoc_tinh | (4, 6] | 1461 | 0.0639 | 0.0403 | 0.1342 |
| tuoi_uoc_tinh | (2, 4] | 537 | 0.0728 | 0.0441 | 0.1397 |
| tuoi_uoc_tinh | (8, 20] | 1614 | 0.0711 | 0.0429 | 0.1450 |
| tuoi_uoc_tinh | nan | 158 | 0.0505 | 0.0321 | 0.0949 |
| thoi_gian_tu_lan_mua_cuoi | (30, 90] | 1520 | 0.0717 | 0.0436 | 0.1414 |
| thoi_gian_tu_lan_mua_cuoi | (-1, 30] | 1110 | 0.0683 | 0.0389 | 0.1279 |
| thoi_gian_tu_lan_mua_cuoi | (90, 180] | 1280 | 0.0671 | 0.0429 | 0.1398 |
| thoi_gian_tu_lan_mua_cuoi | (180, 400] | 1353 | 0.0649 | 0.0399 | 0.1360 |
| nganh_hang_cua_mau_nhan | 01. Top (áo) | 3220 | 0.0484 |  |  |
| nganh_hang_cua_mau_nhan | 02. Bottom (quần & chân váy) | 2326 | 0.0959 |  |  |
| nganh_hang_cua_mau_nhan | 03. Dress (đầm & yếm) | 2026 | 0.0380 |  |  |
| nganh_hang_cua_mau_nhan | 04. Set (bộ) | 1176 | 0.0910 |  |  |
| nganh_hang_cua_mau_nhan | 05. Footwear | 888 | 0.1824 |  |  |
| nganh_hang_cua_mau_nhan | 06. Others | 692 | 0.0954 |  |  |

**Phân tích của bạn:** nhóm nào có Recall thấp nhất? Vì sao (thiếu lịch sử, ngành hàng khó đoán như giày dép có size, bé trai ít dữ liệu hơn bé gái, ...)?

## 7. Mô hình dựa vào đâu? (feature importance của mô hình xếp hạng, theo gain)

| feature | gain |
| --- | --- |
| pop_norm | 7,687 |
| seg_norm | 5,172 |
| seg_rank | 3,014 |
| knn_rank | 2,363 |
| age_gap | 2,309 |
| trend | 2,184 |
| girl_share | 2,053 |
| buyers_7d | 1,892 |
| seg_score | 1,887 |
| group_aff | 1,310 |
| lic_code | 1,022 |
| buyers_90d | 1,021 |
| age_mid | 962 |
| pop_score | 944 |
| als_rank | 861 |

Biểu đồ SHAP: `reports/ranker_shap.png`. Ví dụ gợi ý cụ thể cho 3 khách: `reports/vi_du_goi_y.md`.

**Phân tích của bạn:** feature nào quan trọng nhất và có hợp lý về nghiệp vụ không (ví dụ khách hay mua ngành hàng nào, khớp giới tính, hợp size)?

## 8. Vì sao mô hình deep learning thắng/thua?

Xem `reports/twotower_loss.csv` (đường loss theo epoch) và bảng mục 3–4.

**Phân tích của bạn:** so sánh TwoTower với ItemKNN/ALS/LGBMRanker. Gợi ý lập luận: dữ liệu thưa (khoảng 5 mẫu mỗi khách), mô hình nhiều tham số dễ quá khớp; thuộc tính sản phẩm giúp tổng quát hóa cho mẫu ít dữ liệu; mô hình xếp hạng có thêm feature thời gian/mùa/bản quyền mà two-tower chưa dùng hết.

## 9. Hạn chế và hướng cải thiện

- (viết theo kết quả thật: nhóm khách mới, mẫu mới ra mắt, tuổi của bé chỉ suy ra từ size đã mua, chưa có A/B test)
