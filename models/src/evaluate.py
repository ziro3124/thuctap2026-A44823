"""Đánh giá sâu trên tập test: khoảng tin cậy, phân tích lỗi theo nhóm, giải thích mô hình, ví dụ cụ thể cho 3 khách.

    python -m src.evaluate [--model LGBMRanker]

Đọc artifacts/test_run.pkl do `python -m src.train --stage final` tạo ra. Mọi kết quả ghi vào reports/.
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd

from . import config as C
from .metrics import evaluate, paired_bootstrap, size_mask, summarize

NDCG = f"ndcg@{C.K}"


def load_run():
    with open(C.ART / "test_run.pkl", "rb") as f:
        return pickle.load(f)


def bootstrap_table(ctx, models: dict, best_name: str) -> pd.DataFrame:
    """Chênh lệch của mô hình tốt nhất so với từng mô hình còn lại (khoảng tin cậy 95%, bootstrap ghép cặp theo khách)."""
    res = {n: evaluate(ctx, m) for n, m in models.items()}
    out = []
    for other in models:
        if other == best_name:
            continue
        for metric in ("ndcg", "recall"):
            d, lo, hi = paired_bootstrap(getattr(res[best_name], metric), getattr(res[other], metric))
            out.append({"mo_hinh_tot_nhat": best_name, "so_voi": other, "chi_so": f"{metric}@{C.K}",
                        "chenh_lech": d, "ktc95_thap": lo, "ktc95_cao": hi, "co_y_nghia": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(out)


def error_analysis(ctx, res) -> pd.DataFrame:
    """Nhóm nào dự đoán sai nhiều? Recall@K theo: số lần mua trước, giới tính suy ra, tuổi ước tính, ngành hàng của mẫu nhãn."""
    um = ctx.user_meta.iloc[ctx.rows].reset_index(drop=True)
    rows = []

    def add(kind, labels):
        for lab in pd.unique(labels):
            m = np.asarray(labels == lab)
            if m.sum() >= 20:
                rows.append({"chieu_phan_tich": kind, "nhom": str(lab), "so_khach": int(m.sum()),
                             f"recall@{C.K}": res.recall[m].mean(), f"ndcg@{C.K}": res.ndcg[m].mean(), f"hit_rate@{C.K}": res.hit[m].mean()})
    add("so_lan_mua_truoc", np.where(um.n_inv >= 3, "3+", um.n_inv.astype(int).astype(str)))
    add("gioi_tinh_suy_ra", um.seg.map({0: "be gai", 1: "be trai", 2: "chua ro"}).values)
    add("tuoi_uoc_tinh", pd.cut(um.age_est, [0, 2, 4, 6, 8, 20]).astype(str).values)
    add("thoi_gian_tu_lan_mua_cuoi", pd.cut(um.days_since_last, [-1, 30, 90, 180, 400]).astype(str).values)
    # theo ngành hàng của từng mẫu nhãn: mẫu nhãn có nằm trong top-K không
    r, c = ctx.target.nonzero()
    found = np.array([c[j] in set(res.rec[r[j]]) for j in range(len(r))])
    g = pd.DataFrame({"nhom_nganh": ctx.groups[c], "found": found})
    for lab, d in g.groupby("nhom_nganh"):
        if len(d) >= 20:
            rows.append({"chieu_phan_tich": "nganh_hang_cua_mau_nhan", "nhom": lab, "so_khach": len(d), f"recall@{C.K}": d.found.mean(),
                         f"ndcg@{C.K}": np.nan, f"hit_rate@{C.K}": np.nan})
    rows.append({"chieu_phan_tich": "tran_recall", "nhom": "ty le nhan thuoc mau goi y duoc", "so_khach": len(ctx.eval_users),
                 f"recall@{C.K}": ctx.ceiling, f"ndcg@{C.K}": np.nan, f"hit_rate@{C.K}": np.nan})
    return pd.DataFrame(rows)


def explain(models: dict, ctx) -> None:
    """Feature importance + SHAP của mô hình xếp hạng."""
    rk = models.get("LGBMRanker")
    if rk is None:
        return
    from .models import build_rank_frame
    X, _, _, _ = build_rank_frame(ctx, rk.l1, ctx.rows[:600], rk.topn)
    X = X[rk.booster.feature_name_]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        sv = shap.TreeExplainer(rk.booster).shap_values(X.sample(min(4000, len(X)), random_state=C.SEED))
        shap.summary_plot(sv, X.sample(min(4000, len(X)), random_state=C.SEED), show=False, max_display=15)
        plt.title("SHAP: feature nào đẩy một sản phẩm lên cao trong danh sách gợi ý")
        plt.tight_layout(); plt.savefig(C.REPORTS / "ranker_shap.png", dpi=150); plt.close()
    except Exception as e:                                    # SHAP không bắt buộc; đã có feature importance dạng gain
        print("Bỏ qua SHAP:", e)


def three_examples(ctx, model, names: dict, n: int = 3) -> str:
    """Ví dụ cụ thể: lịch sử mua, 10 gợi ý, nhãn thật cho 3 khách (một bé gái, một bé trai, một khách mà mô hình gợi ý trượt)."""
    res = evaluate(ctx, model, size_filter=True)
    um = ctx.user_meta
    seg = um.seg.values[ctx.rows]
    pick = []
    for want, cond in (("bé gái, gợi ý trúng", (seg == 0) & (res.hit > 0)), ("bé trai, gợi ý trúng", (seg == 1) & (res.hit > 0)),
                       ("gợi ý trượt", res.hit == 0)):
        idx = np.nonzero(cond & (ctx.target_size >= 2) & (um.n_inv.values[ctx.rows] >= 1))[0]
        if len(idx) == 0:
            idx = np.nonzero(cond)[0]
        if len(idx):
            pick.append((want, int(idx[len(idx) // 2])))
    items_meta = names["items"].drop_duplicates("item").set_index("item")
    lines = ["# Ví dụ gợi ý cụ thể cho 3 khách (tập test)\n"]
    for k, (why, i) in enumerate(pick, 1):
        u = ctx.eval_users[i]
        hist = ctx.train[ctx.train.customer_id == u].sort_values("ts")
        r = ctx.rows[i]
        lines.append(f"## Khách {k}: {u} ({why})\n")
        lines.append(f"Bé suy ra: {['gái', 'trai', 'chưa rõ'][int(um.seg.values[r])]}; tuổi ước tính {um.age_est.values[r]:.1f}; "
                     f"{int(um.n_inv.values[r])} lần mua trước ngày chốt.\n")
        lines.append("**Đã mua:** " + "; ".join(f"{items_meta.product_name.get(x, x)} (size {s})" for x, s in zip(hist.item, hist.size_name)) + "\n")
        lines.append("**Top 10 gợi ý** (✓ = khách thật sự mua):\n")
        tgt = set(np.nonzero(ctx.target[i].toarray().ravel())[0])
        for rank, j in enumerate(res.rec[i], 1):
            lines.append(f"{rank}. {'✓ ' if j in tgt else ''}{ctx.names[j]} ({ctx.groups[j]})")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="tên mô hình để phân tích sâu (mặc định: mô hình tốt nhất theo validation)")
    a = ap.parse_args()
    run = load_run()
    ctx, models, best = run["ctx"], run["models"], run["best"]
    if a.model:
        name = a.model
    else:
        cand = {k: v for k, v in best["val_ndcg"].items() if not k.startswith("Pop")}
        name = max(cand, key=cand.get)
    key = next(k for k in models if k.startswith(name.split(" ")[0]))
    print("Phân tích mô hình:", key)
    bt = bootstrap_table(ctx, models, key)
    bt.to_csv(C.REPORTS / "ket_qua_bootstrap.csv", index=False)
    print(bt.round(4).to_string(index=False))
    res = evaluate(ctx, models[key])
    err = error_analysis(ctx, res)
    err.to_csv(C.REPORTS / "phan_tich_loi.csv", index=False)
    print(err.round(3).to_string(index=False))
    explain(models, ctx)
    import pandas as _pd
    from .features import load_raw
    items = load_raw(use_cache=True)["items"]
    (C.REPORTS / "vi_du_goi_y.md").write_text(three_examples(ctx, models[key], {"items": items}), encoding="utf-8")
    print("Đã ghi reports/: ket_qua_bootstrap.csv, phan_tich_loi.csv, ranker_shap.png, vi_du_goi_y.md")


if __name__ == "__main__":
    main()
