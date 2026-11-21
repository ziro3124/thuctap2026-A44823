"""Các mô hình gợi ý. Mọi mô hình có chung giao diện:

    model.fit(ctx)                       # huấn luyện chỉ bằng ctx.train (ts < ngày chốt)
    model.score(ctx, rows) -> ndarray    # điểm (len(rows), n_items) cho các khách có chỉ số `rows` trong ctx.users

Nhóm 1 (baseline): PopGlobal, PopSegment.   Nhóm 2 (lọc cộng tác): ItemKNN, ALS.
Nhóm 3 (boosting): RankerModel (LightGBM LambdaRank, xếp hạng lại ứng viên của nhóm 1+2 bằng thuộc tính sản phẩm/khách).
Nhóm 4 (deep learning): TwoTower (PyTorch, hai tháp khách và sản phẩm, có thuộc tính sản phẩm).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

from . import config as C
from . import features as F
from .metrics import rownorm
from .split import Ctx

try:
    from implicit.als import AlternatingLeastSquares
    HAS_IMPLICIT = True
except Exception:                      # thư viện biên dịch sẵn có thể bị chặn; dùng SVD thay thế
    HAS_IMPLICIT = False
    from sklearn.decomposition import TruncatedSVD


# =============================================================================== baseline
class PopGlobal:
    """Top sản phẩm bán chạy toàn chuỗi: số người mua khác nhau trong `days` ngày trước ngày chốt."""
    name = "PopGlobal"

    def __init__(self, days: int = 30): self.days = days

    def fit(self, ctx: Ctx):
        t = ctx.train[(ctx.train.ts >= ctx.cutoff - pd.Timedelta(days=self.days)) & ctx.train.item.isin(ctx.item_idx)]
        s = t.groupby("item").customer_id.nunique()
        v = np.zeros(ctx.n_items, np.float32)
        if len(s):
            v[s.index.map(ctx.item_idx).values.astype(int)] = s.values
        self.v = v
        return self

    def score(self, ctx, rows): return np.tile(self.v, (len(rows), 1))


class PopSegment:
    """Bán chạy theo giới tính của bé (suy ra từ lịch sử mua) và theo mùa: mẫu trái mùa bị giảm điểm."""
    name = "PopSegment"

    def __init__(self, days: int = 30, mix: float = 0.7, off_season: float = 0.6):
        self.days, self.mix, self.off_season = days, mix, off_season

    def fit(self, ctx: Ctx):
        seg = ctx.user_meta.seg.values
        t = ctx.train[(ctx.train.ts >= ctx.cutoff - pd.Timedelta(days=self.days)) & ctx.train.item.isin(ctx.item_idx)].copy()
        t["seg"] = t.customer_id.map(dict(zip(ctx.users, seg)))
        self.v = np.zeros((3, ctx.n_items), np.float32)
        for g in range(3):
            s = t[t.seg == g].groupby("item").customer_id.nunique()
            if len(s):
                self.v[g, s.index.map(ctx.item_idx).values.astype(int)] = s.values
        s = t.groupby("item").customer_id.nunique()
        self.glob = np.zeros(ctx.n_items, np.float32)
        if len(s):
            self.glob[s.index.map(ctx.item_idx).values.astype(int)] = s.values
        self.season = np.where(ctx.item_meta.season_match.values > 0, 1.0, self.off_season).astype(np.float32)
        self.seg = seg
        return self

    def score(self, ctx, rows):
        gnorm = self.glob / (self.glob.max() + 1e-12)
        out = np.empty((len(rows), ctx.n_items), np.float32)
        for i, r in enumerate(rows):
            g = self.seg[r]
            if g == 2:
                out[i] = gnorm
            else:
                v = self.v[g]
                out[i] = self.mix * v / (v.max() + 1e-12) + (1 - self.mix) * gnorm
        return out * self.season[None, :]


# =============================================================================== lọc cộng tác
class ItemKNN:
    """Item-kNN: độ tương đồng cosine giữa các mẫu, giữ k láng giềng gần nhất, trọng số giảm dần theo thời gian."""
    name = "ItemKNN"

    def __init__(self, k: int = 50, tau: float = 120): self.k, self.tau = k, tau

    def fit(self, ctx: Ctx):
        X = ctx.matrix(self.tau)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel()) + 1e-9
        Xn = X @ sp.diags(1.0 / norms)
        S = (Xn.T @ Xn).toarray().astype(np.float32)
        np.fill_diagonal(S, 0)
        kk = min(self.k, S.shape[0] - 1)
        thr = -np.partition(-S, kk - 1, axis=1)[:, kk - 1][:, None]
        S[S < thr] = 0
        self.S = S
        return self

    def score(self, ctx, rows): return np.asarray(ctx.matrix(self.tau)[rows] @ self.S)


class ALS:
    """ALS cho phản hồi ngầm (thư viện implicit). Không có implicit thì dùng SVD."""
    name = "ALS"

    def __init__(self, factors: int = 32, reg: float = 0.5, alpha: float = 40, iters: int = 15, tau: float = 120):
        self.factors, self.reg, self.alpha, self.iters, self.tau = factors, reg, alpha, iters, tau

    def fit(self, ctx: Ctx):
        X = ctx.matrix(self.tau).copy()
        if HAS_IMPLICIT:
            Cm = X.copy()
            Cm.data = 1.0 + self.alpha * Cm.data
            m = AlternatingLeastSquares(factors=self.factors, regularization=self.reg, iterations=self.iters,
                                        random_state=C.SEED, calculate_training_loss=False, use_gpu=False)
            m.fit(Cm.tocsr(), show_progress=False)
            self.U, self.V = np.asarray(m.user_factors), np.asarray(m.item_factors)
        else:
            svd = TruncatedSVD(n_components=min(self.factors, ctx.n_items - 1), random_state=C.SEED)
            self.U, self.V = svd.fit_transform(X), svd.components_.T
        return self

    def score(self, ctx, rows): return self.U[rows] @ self.V.T


# =============================================================================== boosting: xếp hạng hai tầng
ITEM_COLS = ["log_price", "season_match", "is_girl_item", "is_boy_item", "age_mid", "buyers_7d", "buyers_30d", "buyers_90d",
             "log_buyers_all", "days_since_first_sale", "trend", "group_code", "lic_code"]
USER_COLS = ["n_inv", "n_items", "days_since_last", "days_since_first", "avg_price", "girl_share", "age_est"]
SOURCES = ["pop", "seg", "knn", "als"]


def make_level1(ctx: Ctx, knn=(50, 120), als=(32, 0.5, 40)) -> dict:
    """Huấn luyện các mô hình tầng 1 (nguồn ứng viên) trên ctx."""
    return {"pop": PopGlobal(30).fit(ctx), "seg": PopSegment(30).fit(ctx),
            "knn": ItemKNN(*knn).fit(ctx), "als": ALS(*als).fit(ctx)}


def build_rank_frame(ctx: Ctx, l1: dict, rows: np.ndarray, topn: int = C.CANDIDATES_PER_SOURCE, target=None):
    """Sinh ứng viên (hợp top-N của các nguồn, bỏ mẫu đã mua) và feature cho từng cặp (khách, mẫu).

    Mọi feature dùng ctx (chỉ gồm dữ liệu < ngày chốt). `target` (nếu có) chỉ dùng để gán nhãn y."""
    n, I = len(rows), ctx.n_items
    seen = ctx.seen[rows].toarray() > 0
    S = {k: np.asarray(m.score(ctx, rows), np.float32) for k, m in l1.items()}
    topn = min(topn, I - 1)
    cand = np.zeros((n, I), bool)
    for s in S.values():
        top = np.argpartition(-np.where(seen, -1e9, s), topn, axis=1)[:, :topn]
        cand[np.arange(n)[:, None], top] = True
    cand &= ~seen
    u, i = np.nonzero(cand)
    cols = {}
    for k, s in S.items():
        masked = np.where(seen, -1e9, s)
        order = np.argsort(-masked, axis=1)
        rank = np.empty_like(order)
        rank[np.arange(n)[:, None], order] = np.arange(I)[None, :]
        cols[f"{k}_score"] = s[u, i]
        cols[f"{k}_rank"] = rank[u, i].astype(np.float32)
        cols[f"{k}_norm"] = rownorm(s)[u, i]
    im, um = ctx.item_meta, ctx.user_meta
    for c in ITEM_COLS:
        cols[c] = im[c].values[i].astype(np.float32)
    r = rows[u]
    for c in USER_COLS:
        cols[c] = um[c].values[r].astype(np.float32)
    cols["group_aff"] = ctx.ug[r, np.clip(im.group_code.values[i], 0, ctx.ug.shape[1] - 1)]       # khách hay mua ngành hàng này đến đâu
    cols["lic_aff"] = ctx.ul[r, np.clip(im.lic_code.values[i], 0, ctx.ul.shape[1] - 1)]
    seg = um.seg.values[r]
    cols["gender_match"] = np.where(seg == 2, 0.5, np.where(((seg == 0) & (im.is_girl_item.values[i] > 0)) | ((seg == 1) & (im.is_boy_item.values[i] > 0)), 1.0,
                                    np.where((im.is_girl_item.values[i] + im.is_boy_item.values[i]) == 0, 0.5, 0.0))).astype(np.float32)
    cols["age_gap"] = (im.age_mid.values[i] - um.age_est.values[r]).astype(np.float32)
    cols["size_ok"] = F.size_compatible(im.age_min.values[i], im.age_max.values[i], um.age_est.values[r]).astype(np.float32)
    cols["price_ratio"] = (np.expm1(im.log_price.values[i]) / np.maximum(um.avg_price.values[r], 1.0)).astype(np.float32)
    X = pd.DataFrame(cols)
    y = None
    if target is not None:
        y = np.asarray(target[u, i]).ravel().astype(np.float32)
    return X, y, u, i


def train_ranker(parts_train: list, parts_valid: list | None = None, params: dict | None = None):
    """Huấn luyện LightGBM LambdaRank. parts = danh sách (X, y, u, i) của các lần chốt khác nhau."""
    import lightgbm as lgb
    p = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=1.0)
    p.update(params or {})

    def stack(parts):
        Xs, ys, gs, off = [], [], [], 0
        for X, y, u, _ in parts:
            Xs.append(X); ys.append(y)
            gs.append(np.bincount(u))
        g = np.concatenate([x[x > 0] for x in gs])
        return pd.concat(Xs, ignore_index=True), np.concatenate(ys), g
    X, y, g = stack(parts_train)
    mdl = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", random_state=C.SEED, n_jobs=4, verbose=-1, **p)
    kw = {}
    if parts_valid:
        Xv, yv, gv = stack(parts_valid)
        kw = dict(eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[C.K], callbacks=[lgb.early_stopping(40, verbose=False)])
    mdl.fit(X, y, group=g, **kw)
    return mdl


class RankerModel:
    """Tầng 2: xếp hạng lại ứng viên bằng LightGBM. `l1` là các mô hình tầng 1 đã huấn luyện trên chính ctx dùng để chấm điểm."""
    name = "LGBMRanker"

    def __init__(self, l1: dict, booster, topn: int = C.CANDIDATES_PER_SOURCE):
        self.l1, self.booster, self.topn = l1, booster, topn

    def score(self, ctx, rows):
        out = np.full((len(rows), ctx.n_items), -1e9, np.float32)
        for a in range(0, len(rows), 1000):
            sub = rows[a:a + 1000]
            X, _, u, i = build_rank_frame(ctx, self.l1, sub, self.topn)
            if len(X):
                out[a + u, i] = self.booster.predict(X[self.booster.feature_name_])
        return out


# =============================================================================== deep learning: two-tower

class _Net(nn.Module):
    """Mạng hai tháp; `o` là đối tượng TwoTower giữ tensor thuộc tính của mẫu và siêu tham số."""

    def __init__(self, outer, n, ng, nl):
        super().__init__()
        self.o, self.n = outer, n
        self.id = nn.Embedding(n + 1, 16, padding_idx=n)
        self.g = nn.Embedding(ng + 2, 8)
        self.l = nn.Embedding(nl + 2, 8)
        self.gd = nn.Embedding(3, 4)
        self.se = nn.Embedding(2, 4)
        self.enc = nn.Sequential(nn.Linear(16 + 8 + 8 + 4 + 4 + 3, 64), nn.ReLU())
        self.it = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, outer.dim))
        self.ut = nn.Sequential(nn.Linear(64 + 2, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, outer.dim))

    def enc_items(self, idx):                         # idx: chỉ số mẫu (n = đệm)
        valid = idx < self.n
        safe = torch.where(valid, idx, torch.zeros_like(idx))
        at, nm = self.o.attrs, self.o.num
        z = torch.cat([self.id(idx), self.g(at["group_code"][safe]), self.l(at["lic_code"][safe]), self.gd(at["gender"][safe]),
                       self.se(at["season"][safe]), nm[safe]], -1)
        return self.enc(z) * valid.unsqueeze(-1)

    def item_vec(self, idx):
        return nn.functional.normalize(self.it(self.enc_items(idx)), dim=-1)

    def user_vec(self, hist, uf):
        h = self.enc_items(hist)
        m = (hist < self.n).float().unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return nn.functional.normalize(self.ut(torch.cat([pooled, uf], -1)), dim=-1)


class TwoTower:
    """Mô hình hai tháp (PyTorch). Tháp sản phẩm: embedding id + thuộc tính (ngành hàng, giới tính, mùa, bản quyền, giá, tuổi size).
    Tháp khách: trung bình các mẫu đã mua (qua cùng bộ mã hóa sản phẩm) + số lần mua + thời gian từ lần mua cuối.
    Huấn luyện: với mỗi khách, ẩn ngẫu nhiên 1 mẫu đã mua làm nhãn dương, các nhãn dương khác trong batch làm âm
    (softmax mẫu âm trong batch, hiệu chỉnh theo tần suất mẫu)."""
    name = "TwoTower"

    def __init__(self, dim: int = 32, epochs: int = 25, tau: float = 0.1, lr: float = 2e-3, batch: int = 512, hist_len: int = 40):
        self.dim, self.epochs, self.tau, self.lr, self.batch, self.hist_len = dim, epochs, tau, lr, batch, hist_len

    def _prep(self, ctx: Ctx):
        im = ctx.item_meta
        n = ctx.n_items
        age = im.age_mid.values
        num = np.stack([im.log_price.values / 14.0, np.nan_to_num(age, nan=0.0) / 10.0, (~np.isnan(age)).astype(float)], 1)
        self.attrs = {k: torch.tensor(np.clip(im[k].values, -1, None) + 1, dtype=torch.long)
                      for k in ("group_code", "lic_code")}
        self.attrs["gender"] = torch.tensor((im.is_girl_item.values * 1 + im.is_boy_item.values * 2).astype(int), dtype=torch.long)
        self.attrs["season"] = torch.tensor(im.season_match.values.astype(int), dtype=torch.long)
        self.num = torch.tensor(num, dtype=torch.float32)
        um = ctx.user_meta
        self.uf = torch.tensor(np.stack([np.log1p(um.n_inv.values) / 3.0, np.log1p(um.days_since_last.clip(upper=400).values) / 6.0], 1), dtype=torch.float32)
        # lịch sử: tối đa hist_len mẫu gần nhất của mỗi khách, đệm bằng chỉ số n
        a = ctx.agg.copy()
        a["u"] = a.customer_id.map(ctx.user_idx).values
        a["i"] = a.item.map(ctx.item_idx).values
        a = a.sort_values(["u", "last"], ascending=[True, False])
        a["pos"] = a.groupby("u").cumcount()
        a = a[a.pos < self.hist_len]
        H = np.full((len(ctx.users), self.hist_len), n, dtype=np.int64)
        H[a.u.values, a.pos.values] = a.i.values
        self.H = torch.tensor(H)
        self.n_pad = n
        cnt = np.bincount(ctx.agg.item.map(ctx.item_idx).values.astype(int), minlength=n).astype(np.float64) + 1
        self.logq = torch.tensor(np.log(cnt / cnt.sum()), dtype=torch.float32)

    def _build(self, ctx, vocab_sizes):
        return _Net(self, ctx.n_items, *vocab_sizes)

    def fit(self, ctx: Ctx):
        torch.manual_seed(C.SEED); np.random.seed(C.SEED)
        torch.set_num_threads(4)
        self._prep(ctx)
        vs = (int(ctx.item_meta.group_code.max()) + 1, int(ctx.item_meta.lic_code.max()) + 1)
        self.net = self._build(ctx, vs)
        valid_users = torch.nonzero((self.H < self.n_pad).sum(1) >= 2).squeeze(1)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-6)
        g = torch.Generator().manual_seed(C.SEED)
        self.loss_curve = []
        for ep in range(self.epochs):
            self.net.train()
            perm = valid_users[torch.randperm(len(valid_users), generator=g)]
            tot, nb = 0.0, 0
            for a in range(0, len(perm), self.batch):
                ub = perm[a:a + self.batch]
                if len(ub) < 8:
                    continue
                H = self.H[ub].clone()
                valid = H < self.n_pad
                r = torch.rand(H.shape, generator=g).masked_fill(~valid, -1.0)
                pp = r.argmax(1)
                pos = H.gather(1, pp[:, None]).squeeze(1)
                H.scatter_(1, pp[:, None], self.n_pad)                 # ẩn mẫu nhãn khỏi lịch sử
                u = self.net.user_vec(H, self.uf[ub])
                v = self.net.item_vec(pos)
                logits = (u @ v.T) / self.tau - self.logq[pos][None, :]
                same = (pos[:, None] == pos[None, :]) & ~torch.eye(len(pos), dtype=torch.bool)
                logits = logits.masked_fill(same, -1e9)
                loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(pos)))
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
            self.loss_curve.append(tot / max(nb, 1))
        self.net.eval()
        return self

    def score(self, ctx, rows):
        with torch.no_grad():
            iv = self.net.item_vec(torch.arange(ctx.n_items))
            rows_t = torch.tensor(rows)
            uv = self.net.user_vec(self.H[rows_t], self.uf[rows_t])
            return (uv @ iv.T).numpy()
