"""Dùng chung cho dự đoán hàng loạt và API: nạp bundle và gợi ý top-K cho khách."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from . import config as C
from .metrics import size_mask


class Recommender:
    def __init__(self, path=None):
        with open(path or C.ART / "bundle.pkl", "rb") as f:
            b = pickle.load(f)
        self.name, self.model, self.ctx, self.fallback = b["model_name"], b["model"], b["ctx"], b["fallback"]
        self.trained_until = b["trained_until"]

    def known(self, customer_id: str) -> bool:
        return str(customer_id) in self.ctx.user_idx

    def recommend(self, customer_id: str, k: int = C.K, size_filter: bool = True) -> pd.DataFrame:
        """Top-k sản phẩm. Khách chưa có lịch sử: trả về top bán chạy toàn chuỗi (source = fallback_popularity)."""
        return self.recommend_many([customer_id], k, size_filter)

    def recommend_many(self, customer_ids, k: int = C.K, size_filter: bool = True) -> pd.DataFrame:
        ctx, out = self.ctx, []
        ids = [str(c) for c in customer_ids]
        rows = np.array([ctx.user_idx.get(c, -1) for c in ids])
        known = rows >= 0
        for a in range(0, len(ids), 1000):
            sl = slice(a, a + 1000)
            r, kn, cid = rows[sl], known[sl], ids[sl]
            S = np.empty((len(r), ctx.n_items), np.float32)
            if kn.any():
                S[kn] = np.asarray(self.model.score(ctx, r[kn]), np.float32)
                S[kn] = np.where(ctx.seen[r[kn]].toarray() > 0, -1e9, S[kn])        # không gợi ý lại mẫu đã mua
                if size_filter:
                    S[kn] = np.where(size_mask(ctx, r[kn]), S[kn], -1e9)
            if (~kn).any():
                S[~kn] = self.fallback.score(ctx, np.zeros((~kn).sum(), int))
            kk = min(k, ctx.n_items)
            top = np.argsort(-S, axis=1)[:, :kk]
            for n_, c in enumerate(cid):
                for rank, j in enumerate(top[n_], 1):
                    out.append((c, rank, ctx.items[j], ctx.names[j], ctx.groups[j], float(S[n_, j]),
                                self.name if kn[n_] else "fallback_popularity"))
        return pd.DataFrame(out, columns=["customer_id", "rank", "item", "product_name", "group_cat", "score", "source"])
