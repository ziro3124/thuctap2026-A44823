# Kết quả EDA - Tuần 5

_File sinh tự động bởi `eda.py`. Số liệu tính từ hóa đơn hợp lệ năm 2025; doanh thu = `actual_amount`; doanh thu thuần = bán trừ trả._

## Tổng quan dữ liệu

- Dùng 251,707/251,945 hóa đơn hợp lệ (99.9%), ngày 01/01/2025 – 30/12/2025.
- Doanh thu bán 135.1 tỷ đồng; trả hàng 2.62 tỷ (1.9% doanh thu bán); doanh thu thuần 132.5 tỷ.
- Hóa đơn trả hàng (số tiền âm): 8,199 (3.3% số hóa đơn).
- Ngoại lai: AOV trung vị 399,000đ, P99 2,643,729đ, lớn nhất 98,671,000đ. Không loại ngoại lai (có thể là đơn sỉ thật); dùng trung vị và kiểm định phi tham số.
- 39 hóa đơn có giảm giá lớn hơn tiền hàng -> khi tính tỷ lệ giảm giá bị chặn ở 100%.

## A. Mùa vụ và thời gian

- Tháng 1 cao nhất (19.5 tỷ, 14.7% cả năm); tháng 2 thấp nhất (4.3 tỷ). Quý 4 chiếm 27.8% doanh thu năm.
- Dịp mạnh nhất: Trước Tết (15–28/1) - doanh thu/ngày gấp 2.25 lần ngày thường; dịp yếu nhất: Nghỉ Tết (29/1–4/2) (0.38 lần).
- Cửa hàng bán lẻ: T7+CN chiếm 47.5% doanh thu (mức đều là 28,6%); 3 giờ cao điểm (19h, 20h, 21h) chiếm 33.6%.
- Online có giờ cao điểm quanh 11h; cửa hàng bán lẻ quanh 21h.

## B. Điểm bán

- 24/41 cửa hàng bán lẻ (59%) tạo 80% doanh thu cửa hàng; 10% cửa hàng hàng đầu (4 cửa hàng) chiếm 21.3%.
- Khu vực 'South' chiếm 59.0% doanh thu; mặt bằng 'Mall' chiếm 72.8%.
- Cửa hàng đã đóng ('closed') vẫn đóng góp 10.7 tỷ (8.0% doanh thu năm) -> cần tách khi lập kế hoạch 2026 để không ước lượng thừa.
- 1 mã điểm bán không có trong bảng branch, doanh thu 381 triệu (0.3%).
- Hiệu suất trung vị: 'Road' 10.8 triệu/ngày vs 'Mall' 6.9 triệu/ngày.

## C. Sản phẩm

- Ngành hàng '03. Dress (đầm & yếm)' chiếm 29.0% doanh thu, '01. Top (áo)' 22.6%; top 3 ngành chiếm 68.9%.
- Theo giới tính: 'Bé gái' dẫn đầu (62.8%). Theo mùa: 'Spring - Summer' chiếm 67.8%.
- Sản phẩm bản quyền nhân vật: 99.6% doanh thu từ 3,240 item (TB 42.6 triệu/item) vs thương hiệu riêng 59 item (TB 8.4 triệu/item).
- ABC: 655 item hạng A (20.0% số item) tạo 80% doanh thu; 2,069 item hạng C (63.2%) chỉ tạo 5.0%.

## D. Giá và khuyến mãi

- 44.3% đơn bán có giảm giá; trên đơn có giảm, mức giảm trung bình 24.2% tiền hàng; tính cả năm, giảm giá bằng 20.7% tổng tiền hàng.
- Cửa hàng bán lẻ: tháng giảm giá nhiều nhất là tháng 11 (63% đơn), ít nhất tháng 2 (29%).
- Online: tháng giảm giá nhiều nhất là tháng 11 (94% đơn), ít nhất tháng 1 (60%).
- Cửa hàng bán lẻ: AOV trung vị đơn có giảm giá 518,500đ vs không giảm 458,000đ (+13%).

## E. Khách hàng thành viên

- Khách có mã thành viên tạo 53.2% doanh thu bán (40.8% số đơn); cao nhất tháng 1 (64%), thấp nhất tháng 2 (47%).
- Tỷ lệ doanh thu thành viên theo kênh: Cửa hàng bán lẻ 65%; Khác (sỉ/sự kiện/không rõ) 2%; Online 0%.
- 67,198 khách thành viên mua trong năm; 20.9% mua từ 2 ngày trở lên (trung bình 1.38 ngày mua/khách). Khoảng cách giữa hai lần mua: trung vị 55 ngày, 36% trong vòng 30 ngày.
- Theo hạng thẻ: 'Thỏ Nhí' đóng góp nhiều doanh thu thành viên nhất (50.6%); AOV TB từ 640,564đ (Thỏ Nhí) đến 1,063,177đ (Thỏ Kim Cương).
- AOV trung vị cửa hàng bán lẻ: thành viên 538,000đ vs vãng lai 399,000đ (+35%).
- Cohort giữ chân: trung bình 4.3% khách quay lại ở tháng kế tiếp sau lần mua đầu (cohort tháng 1 gồm cả khách cũ nên bị cao ảo).

## F. Trả hàng

- Tổng cộng bán 765,481 sản phẩm, trả 18,100 (2.36% số lượng).
- Ngành hàng trả nhiều nhất: '05. Footwear' (6.75%); ít nhất: '06. Others' (0.65%) (chỉ xét ngành bán ≥ 1,000 sản phẩm).
- Size hay bị trả nhất: 'Size33-22cm' (8.68%), kế đến 'Size34-23cm'; thấp nhất '1Y-9-11kg' (0.34%).
- Theo giá trị: cửa hàng trả nhiều nhất 'PB - Vincom Trần Duy Hưng' (1.1% doanh thu bán), trung vị các cửa hàng 0.2%.
- Tỷ lệ trả theo giá trị cao nhất tháng 12 (5.2%), thấp nhất tháng 11 (0.8%).

## G. Online và offline

- Online chiếm 1.7% doanh thu bán (6,884 đơn); cửa hàng bán lẻ 81.3% (167,710 đơn).
- AOV trung vị: online 289,000đ vs cửa hàng 478,000đ; UPT: 1.63 vs 3.58 sản phẩm/đơn.
- Giảm giá bằng 18.5% tiền hàng ở online vs 10.5% ở cửa hàng; tỷ lệ COD online 94.2%; giá trị trả hàng 2.9% (online) vs 0.3% (cửa hàng).
- Tỷ lệ COD theo sàn: Adayroi 0%; Haravan 0%; Shopee 100%; Tiktok 0%.

## Kiểm định thống kê

### 1. [D] AOV đơn có giảm giá có khác đơn không giảm giá (cửa hàng bán lẻ)?

- **H0:** Phân phối giá trị đơn của hai nhóm như nhau
- **H1:** Phân phối giá trị đơn của hai nhóm khác nhau
- **Kiểm định:** Mann–Whitney U (hai phía)
- **Số liệu:** Có giảm giá: n=73,169, trung vị=518,500, TB=721,091 | Không giảm giá: n=94,477, trung vị=458,000, TB=604,142
- **Thống kê:** U=3,571,694,646; p < 0,001; rank-biserial r=+0.03
- **Kết luận:** bác bỏ H0 (mức 5%). r>0 nghĩa là Có giảm giá có xu hướng lớn hơn Không giảm giá.

### 2. [E] AOV của khách thành viên có khác khách vãng lai (cửa hàng bán lẻ)?

- **H0:** Phân phối giá trị đơn của khách thành viên và vãng lai như nhau
- **H1:** Phân phối giá trị đơn của hai nhóm khác nhau
- **Kiểm định:** Mann–Whitney U (hai phía)
- **Số liệu:** Thành viên: n=98,089, trung vị=538,000, TB=727,256 | Vãng lai: n=69,621, trung vị=399,000, TB=553,102
- **Thống kê:** U=4,171,597,822; p < 0,001; rank-biserial r=+0.22
- **Kết luận:** bác bỏ H0 (mức 5%). r>0 nghĩa là Thành viên có xu hướng lớn hơn Vãng lai.

### 3. [F] Tỷ lệ trả hàng (theo số lượng) có phụ thuộc ngành hàng? (6 ngành bán nhiều nhất)

- **H0:** Tỷ lệ trả hàng như nhau giữa các ngành hàng
- **H1:** Có ít nhất một ngành có tỷ lệ trả khác
- **Kiểm định:** Chi-square độc lập
- **Số liệu:** 06. Others: 0.65%; 01. Top (áo): 2.87%; 03. Dress (đầm & yếm): 3.21%; 02. Bottom (quần & chân váy): 2.69%; 04. Set (bộ): 2.54%; 05. Footwear: 6.75%
- **Thống kê:** chi2=6,675.8; dof=5; p < 0,001; Cramér's V=0.093
- **Kết luận:** bác bỏ H0 (mức 5%). Cỡ ảnh hưởng V=0.093 (nhỏ). Lưu ý: đơn vị là sản phẩm chứ không độc lập tuyệt đối, p-value có xu hướng quá nhỏ.

### 4. [G] AOV đơn online có khác đơn tại cửa hàng bán lẻ?

- **H0:** Phân phối giá trị đơn online và cửa hàng như nhau
- **H1:** Phân phối giá trị đơn hai kênh khác nhau
- **Kiểm định:** Mann–Whitney U (hai phía)
- **Số liệu:** Online: n=6,884, trung vị=289,000, TB=336,548 | Cửa hàng: n=167,710, trung vị=478,000, TB=654,960
- **Thống kê:** U=332,531,881; p < 0,001; rank-biserial r=-0.42
- **Kết luận:** bác bỏ H0 (mức 5%). r>0 nghĩa là Online có xu hướng lớn hơn Cửa hàng.

## Danh sách biểu đồ

- `reports/figures/A1_doanh_thu_theo_ngay.png`
- `reports/figures/A2_theo_thang_va_thu.png`
- `reports/figures/A3_hieu_ung_dip_le.png`
- `reports/figures/A4_nhiet_do_thu_gio.png`
- `reports/figures/B1_tap_trung_va_khu_vuc.png`
- `reports/figures/B2_top_cua_hang_va_hieu_suat.png`
- `reports/figures/C1_co_cau_nganh_gioi_mua.png`
- `reports/figures/C2_ban_quyen_vs_thuong_hieu_rieng.png`
- `reports/figures/C3_phan_tich_ABC.png`
- `reports/figures/D1_khuyen_mai_theo_thang_kenh.png`
- `reports/figures/D2_aov_va_muc_giam.png`
- `reports/figures/E1_ty_trong_va_hang_the.png`
- `reports/figures/E2_tan_suat_va_khoang_cach_mua.png`
- `reports/figures/E3_cohort_giu_chan.png`
- `reports/figures/F1_tra_hang_nganh_va_size.png`
- `reports/figures/F2_tra_hang_cua_hang_va_thang.png`
- `reports/figures/G1_kenh_ban_va_chi_so.png`
- `reports/figures/G2_online_theo_thang_va_cod.png`
