"""Dự đoán hàng loạt: top-10 gợi ý cho mỗi khách đã mua >= 2 lần (tính đến ngày chốt 31/12/2025). Ghi CSV.

    python -m src.predict_batch [--out reports/goi_y_top10.csv]
"""
from __future__ import annotations

import argparse

from . import config as C
from .recommend import Recommender


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.REPORTS / "goi_y_top10.csv"))
    ap.add_argument("--k", type=int, default=C.K)
    ap.add_argument("--no-size-filter", action="store_true")
    a = ap.parse_args()
    rec = Recommender()
    um = rec.ctx.user_meta
    cust = um.index[um.n_inv >= 2].tolist()                  # n_inv = số hóa đơn đến ngày chốt
    df = rec.recommend_many(cust, a.k, size_filter=not a.no_size_filter)
    df.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"{len(cust):,} khách mua >= 2 lần, {len(df):,} dòng gợi ý -> {a.out} (mô hình: {rec.name}, dữ liệu đến {rec.trained_until})")


if __name__ == "__main__":
    main()
