"""Metric đánh giá hệ gợi ý: Precision@K, Recall@K, NDCG@K, Hit Rate, độ phủ danh mục (coverage)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from .split import Ctx


@dataclass
class Result:
    ctx: Ctx
    rec: np.ndarray            # (n_eval, K) chỉ số mẫu được gợi ý
    hits: np.ndarray           # (n_eval, K) 1 nếu mẫu đúng
    precision: np.ndarray
    recall: np.ndarray
    ndcg: np.ndarray
    hit: np.ndarray


def rownorm(S: np.ndarray) -> np.ndarray:
    S = S - S.min(axis=1, keepdims=True)
    return S / (S.max(axis=1, keepdims=True) + 1e-12)


def size_mask(ctx: Ctx, rows: np.ndarray) -> np.ndarray:
    """Ma trận bool (len(rows), n_items): True nếu mẫu hợp size với tuổi ước tính của bé."""
    im, um = ctx.item_meta, ctx.user_meta
    return F.size_compatible(im.age_min.values[None, :], im.age_max.values[None, :], um.age_est.values[rows][:, None])


def evaluate(ctx: Ctx, model, k: int = C.K, size_filter: bool = False, chunk: int = 1500) -> Result:
    """Đánh giá `model` (có model.score(ctx, rows) -> ma trận điểm) trên khách kiểm tra của ctx."""
    n = len(ctx.eval_users)
    k = min(k, ctx.n_items - 1)
    rec = np.zeros((n, k), dtype=int)
    hits = np.zeros((n, k), dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        rows = ctx.rows[a:b]
        S = np.asarray(model.score(ctx, rows), dtype=np.float32).copy()
        S = np.where(ctx.seen[rows].toarray() > 0, -1e9, S)                      # không gợi ý lại mẫu đã mua
        if size_filter:
            S = np.where(size_mask(ctx, rows), S, -1e9)
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, idx, axis=1), axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        rec[a:b] = idx
        hits[a:b] = np.take_along_axis(ctx.target[a:b].toarray(), idx, axis=1)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    ts = ctx.target_size
    nh = hits.sum(1)
    ndcg = (hits * disc).sum(1) / np.cumsum(disc)[np.minimum(ts, k) - 1]
    return Result(ctx, rec, hits, nh / k, nh / ts, ndcg, (nh > 0).astype(float))


def summarize(res: Result, name: str, extra: dict | None = None) -> pd.DataFrame:
    """Bảng metric theo nhóm khách (tất cả, ít lần mua, nhiều lần mua)."""
    ctx = res.ctx
    n_inv = ctx.user_meta.n_inv.values[ctx.rows]
    groups = {"tat_ca": np.ones(len(ctx.rows), bool), "mua_2_lan": n_inv == 1, "mua_3_lan_tro_len": n_inv >= 2}
    cov = len(np.unique(res.rec)) / ctx.n_items
    div = np.array([len(set(ctx.groups[r])) for r in res.rec])
    out = []
    for g, m in groups.items():
        if m.sum() == 0:
            continue
        out.append({"mo_hinh": name, "nhom": g, "so_khach": int(m.sum()),
                    f"precision@{C.K}": res.precision[m].mean(), f"recall@{C.K}": res.recall[m].mean(),
                    f"ndcg@{C.K}": res.ndcg[m].mean(), f"hit_rate@{C.K}": res.hit[m].mean(),
                    "coverage": cov, "da_dang_nganh": div[m].mean(), **(extra or {})})
    return pd.DataFrame(out)


def paired_bootstrap(a, b, n: int = 2000, seed: int = C.SEED):
    """Chênh lệch trung bình (a - b) và khoảng tin cậy 95% bằng bootstrap ghép cặp theo khách."""
    r = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    means = [d[r.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)
