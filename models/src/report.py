"""Sinh reports/danh-gia-mo-hinh.md từ các CSV kết quả (bảng số liệu tự điền; phần phân tích bạn tự viết).

    python -m src.report
"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C

K = C.K


def md_table(df: pd.DataFrame, fmt: str = "{:.4f}") -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    head = "| " + " | ".join(map(str, d.columns)) + " |\n| " + " | ".join("---" for _ in d.columns) + " |\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.values)


def main():
    R = C.REPORTS
    val = pd.read_csv(R / "ket_qua_validation.csv")
    test = pd.read_csv(R / "ket_qua_test.csv")
    best = json.loads((R / "best_config.json").read_text(encoding="utf-8"))
    boot = pd.read_csv(R / "ket_qua_bootstrap.csv") if (R / "ket_qua_bootstrap.csv").exists() else None
    err = pd.read_csv(R / "phan_tich_loi.csv") if (R / "phan_tich_loi.csv").exists() else None
    imp = pd.read_csv(R / "ranker_feature_importance.csv") if (R / "ranker_feature_importance.csv").exists() else None
    cols = ["mo_hinh", f"precision@{K}", f"recall@{K}", f"ndcg@{K}", f"hit_rate@{K}", "coverage"]

    last = val[(val.nhom == "tat_ca") & (val.fold == val.fold.max())].sort_values(f"ndcg@{K}", ascending=False)
    t_all = test[(test.nhom == "tat_ca") & (~test.loc_size)].sort_values(f"ndcg@{K}", ascending=False)
    t_filt = test[(test.nhom == "tat_ca") & (test.loc_size)].sort_values(f"ndcg@{K}", ascending=False)
    base = t_all[t_all.mo_hinh == "PopGlobal"].iloc[0]
    top = t_all.iloc[0]
    gain_n = top[f"ndcg@{K}"] / base[f"ndcg@{K}"] - 1
    gain_r = top[f"recall@{K}"] / base[f"recall@{K}"] - 1
    ceil = err[err.chieu_phan_tich == "tran_recall"][f"recall@{K}"].iloc[0] if err is not None else float("nan")

    out = [f"# Đánh giá mô hình gợi ý sản phẩm (Đề D)\n",
           "_Bảng số liệu dưới đây được sinh tự động từ `reports/*.csv` bằng `python -m src.report`. Các mục **Phân tích của bạn** để trống cho bạn viết._\n",
           "## 1. Tóm tắt kết quả\n",
           f"Trên tập test, mô hình tốt nhất là **{top.mo_hinh}** (NDCG@{K} = {top[f'ndcg@{K}']:.4f}, Recall@{K} = {top[f'recall@{K}']:.4f}), "
           f"so với baseline top bán chạy toàn chuỗi (NDCG@{K} = {base[f'ndcg@{K}']:.4f}, Recall@{K} = {base[f'recall@{K}']:.4f}): "
           f"NDCG tăng {100 * gain_n:.0f}%, Recall tăng {100 * gain_r:.0f}%. "
           f"Trần Recall của bài toán là {ceil:.3f} (phần nhãn thuộc các mẫu còn có thể gợi ý).\n",
           "## 2. Thiết kế thí nghiệm\n",
           f"- Chia theo thời gian: validation gồm 2 fold (ngày chốt {C.VAL_FOLDS[0][0]} và {C.VAL_FOLDS[1][0]}), test ngày chốt {C.TEST[0]} (kỳ {C.TEST[0]} đến {C.TEST[1]}). "
           "Mỗi khách kiểm tra có ≥ 2 lần mua: 1 lần trước ngày chốt, nhãn là hóa đơn cuối cùng trong kỳ.\n"
           f"- Mẫu được gợi ý: có giao dịch trong {C.ACTIVE_DAYS} ngày trước ngày chốt, loại mẫu khách đã mua, loại mã không phải hàng hóa.\n"
           f"- Metric: Precision@{K}, Recall@{K}, NDCG@{K}, Hit Rate@{K}, coverage danh mục (tỷ lệ mẫu từng được gợi ý).\n"
           f"- Cấu hình chọn trên validation: {json.dumps({k: v for k, v in best.items() if k != 'val_ndcg'}, ensure_ascii=False)}\n",
           f"## 3. So sánh mô hình trên validation (fold {int(val.fold.max())})\n", md_table(last[cols]) + "\n",
           "## 4. Kết quả trên tập test (không lọc size)\n", md_table(t_all[cols]) + "\n",
           "### Có lọc size theo tuổi của bé\n", md_table(t_filt[cols]) + "\n",
           "Lọc size là quy tắc nghiệp vụ (không gợi ý mẫu sai size so với tuổi ước tính của bé); bảng trên cho biết cái giá của quy tắc này đối với metric.\n"]
    if boot is not None:
        out += [f"## 5. Mô hình tốt nhất có hơn các mô hình khác không? (bootstrap ghép cặp, khoảng tin cậy 95%)\n", md_table(boot) + "\n"]
    if err is not None:
        out += ["## 6. Phân tích lỗi: nhóm nào dự đoán sai nhiều?\n", md_table(err[err.chieu_phan_tich != "tran_recall"]) + "\n",
                "**Phân tích của bạn:** nhóm nào có Recall thấp nhất? Vì sao (thiếu lịch sử, ngành hàng khó đoán như giày dép có size, bé trai ít dữ liệu hơn bé gái, ...)?\n"]
    if imp is not None:
        out += ["## 7. Mô hình dựa vào đâu? (feature importance của mô hình xếp hạng, theo gain)\n", md_table(imp.head(15), "{:,.0f}") + "\n",
                "Biểu đồ SHAP: `reports/ranker_shap.png`. Ví dụ gợi ý cụ thể cho 3 khách: `reports/vi_du_goi_y.md`.\n",
                "**Phân tích của bạn:** feature nào quan trọng nhất và có hợp lý về nghiệp vụ không (ví dụ khách hay mua ngành hàng nào, khớp giới tính, hợp size)?\n"]
    out += ["## 8. Vì sao mô hình deep learning thắng/thua?\n",
            "Xem `reports/twotower_loss.csv` (đường loss theo epoch) và bảng mục 3–4.\n",
            "**Phân tích của bạn:** so sánh TwoTower với ItemKNN/ALS/LGBMRanker. Gợi ý lập luận: dữ liệu thưa (khoảng 5 mẫu mỗi khách), mô hình nhiều tham số dễ quá khớp; "
            "thuộc tính sản phẩm giúp tổng quát hóa cho mẫu ít dữ liệu; mô hình xếp hạng có thêm feature thời gian/mùa/bản quyền mà two-tower chưa dùng hết.\n",
            "## 9. Hạn chế và hướng cải thiện\n",
            "- (viết theo kết quả thật: nhóm khách mới, mẫu mới ra mắt, tuổi của bé chỉ suy ra từ size đã mua, chưa có A/B test)\n"]
    (R / "danh-gia-mo-hinh.md").write_text("\n".join(out), encoding="utf-8")
    print("Đã ghi", R / "danh-gia-mo-hinh.md")


if __name__ == "__main__":
    main()
