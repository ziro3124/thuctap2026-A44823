import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")      # implicit khuyến nghị 1 luồng BLAS
os.environ.setdefault("MKL_NUM_THREADS", "1")
import pickle, warnings, json, re, itertools, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()
pd.set_option("display.max_columns", 50); pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.autolayout": True})

# ---- Cấu hình bài toán ----
K = 10
VAL  = ("2025-11-01", "2025-12-01")      # train: < 01/11
TEST = ("2025-12-01", "2026-01-01")      # train: < 01/12
ACTIVE_DAYS = 60                         # chỉ gợi ý mẫu có giao dịch trong 60 ngày cuối của train
SEED = 42
GENDER_GIRL, GENDER_BOY = "Bé gái", "Bé trai"
rng = np.random.default_rng(SEED)

REPORTS = Path("reports"); FIGS = REPORTS / "figs"; DATA = Path("data"); MODELS = Path("models")
for p in (FIGS, DATA, MODELS): p.mkdir(parents=True, exist_ok=True)


STATE = DATA / "rs_state.json"

def load_cache():
    """Đọc dữ liệu đã lưu ở notebook 03_1."""
    inter, item_stats, catalog = (pd.read_parquet(DATA / f"rs_{n}.parquet") for n in ["inter", "item_stats", "catalog"])
    n_inv_total = int(pd.read_parquet(DATA / "rs_meta.parquet").n_inv_total[0])
    for d in (inter, item_stats, catalog): d["item"] = d["item"].astype(str)
    inter["ngay"] = pd.to_datetime(inter["ngay"])
    for c in ["qty", "rev"]: inter[c] = inter[c].astype(float)
    return inter, item_stats, catalog, n_inv_total

def load_state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}

def save_state(**kw):
    st = load_state(); st.update(kw)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

def rownorm(S):
    S = S - S.min(axis=1, keepdims=True)
    return S / (S.max(axis=1, keepdims=True) + 1e-12)

class Split:
    def __init__(self, inter, catalog, blacklist, start, stop, active_days=ACTIVE_DAYS):
        self.start, self.stop = pd.Timestamp(start), pd.Timestamp(stop)
        inter = inter[~inter.item.isin(blacklist)]
        self.train = inter[inter.ngay < self.start]
        win = inter[(inter.ngay >= self.start) & (inter.ngay < self.stop)]
        recent = self.train[self.train.ngay >= self.start - pd.Timedelta(days=active_days)]
        self.items = np.array(sorted(recent.item.unique()))
        self.item_idx = {it: i for i, it in enumerate(self.items)}
        self.users = np.array(sorted(self.train.customer_id.unique()))
        self.user_idx = {u: i for i, u in enumerate(self.users)}
        cat = catalog.drop_duplicates("item").set_index("item")
        self.gender_map = cat.gender.to_dict()
        self.name = cat.product_name.reindex(self.items).fillna("").values
        self.group = cat.group_cat.reindex(self.items).fillna("(không rõ)").values
        # khách trong kỳ đánh giá
        self.eval_users = np.array(sorted(win.customer_id.unique()))
        self.rows = np.array([self.user_idx.get(u, -1) for u in self.eval_users], dtype=int)
        self.truth_size = win.groupby("customer_id").item.nunique().reindex(self.eval_users).values.astype(int)
        pairs = win[win.item.isin(self.item_idx)][["customer_id", "item"]].drop_duplicates()
        r = pd.Index(self.eval_users).get_indexer(pairs.customer_id)
        c = pairs.item.map(self.item_idx).values.astype(int)
        self.T = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(self.eval_users), len(self.items)))
        self.ceiling = len(pairs) / max(1, win[["customer_id", "item"]].drop_duplicates().shape[0])
        # ma trận lịch sử
        tc = self.train[self.train.item.isin(self.item_idx)]
        self.agg = tc.groupby(["customer_id", "item"]).agg(qty=("qty", "sum"), last=("ngay", "max")).reset_index()
        self._ar = self.agg.customer_id.map(self.user_idx).values.astype(int)
        self._ac = self.agg.item.map(self.item_idx).values.astype(int)
        shape = (len(self.users), len(self.items))
        self.seen = sp.csr_matrix((np.ones(len(self.agg), np.float32), (self._ar, self._ac)), shape=shape)
        self.user_days = self.train.groupby("customer_id").ngay.nunique().reindex(self.users).values
        self.known = self.rows >= 0
        self.known2 = self.known & (np.where(self.rows >= 0, self.user_days[np.maximum(self.rows, 0)], 0) >= 2)
        self.n_items, self.n_users = len(self.items), len(self.users)
        self._mat = {}

    def matrix(self, tau=None):
        # trọng số = log(1 + số lượng), giảm dần theo thời gian nếu có tau (ngày)
        if tau not in self._mat:
            w = np.log1p(self.agg.qty.values)
            if tau:
                w = w * np.exp(-(self.start - self.agg["last"]).dt.days.values / tau)
            self._mat[tau] = sp.csr_matrix((w.astype(np.float32), (self._ar, self._ac)), shape=(self.n_users, self.n_items))
        return self._mat[tau]

    def summary(self):
        return (f"train < {self.start:%d/%m/%Y}: {len(self.train):,} dòng, {self.n_users:,} khách | mẫu được gợi ý: {self.n_items:,} | "
                f"khách kỳ đánh giá: {len(self.eval_users):,} (có lịch sử {self.known.sum():,}, mới {(~self.known).sum():,}) | "
                f"trần Recall: {self.ceiling:.3f}")

class Result:
    def __init__(self, split, rec, recall, precision, ndcg, ap, hit):
        self.split, self.rec = split, rec
        self.recall, self.precision, self.ndcg, self.ap, self.hit = recall, precision, ndcg, ap, hit

def evaluate(split, scorer, k=K, exclude_seen=True, chunk=2000):
    n = len(split.eval_users); k = min(k, split.n_items - 1)
    rec = np.zeros((n, k), dtype=int)
    hits_all = np.zeros((n, k), dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n); rows = split.rows[a:b]
        S = np.asarray(scorer(rows), dtype=np.float32).copy()
        if exclude_seen:
            kn = rows >= 0
            if kn.any():
                S[kn] = np.where(split.seen[rows[kn]].toarray() > 0, -1e9, S[kn])
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, idx, axis=1), axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        rec[a:b] = idx
        hits_all[a:b] = np.take_along_axis(split.T[a:b].toarray(), idx, axis=1)
    disc = 1.0 / np.log2(np.arange(2, k + 2)); ts = split.truth_size
    nh = hits_all.sum(1)
    ndcg = (hits_all * disc).sum(1) / np.cumsum(disc)[np.minimum(ts, k) - 1]
    ap = ((np.cumsum(hits_all, 1) / np.arange(1, k + 1)) * hits_all).sum(1) / np.minimum(ts, k)
    return Result(split, rec, nh / ts, nh / k, ndcg, ap, (nh > 0).astype(float))

def summarize(res, name, extra=None):
    sp_ = res.split; rows = []
    groups = {"tat_ca": np.ones(len(sp_.eval_users), bool), "co_lich_su": sp_.known,
              "khach_moi": ~sp_.known, "lich_su_2_ngay": sp_.known2}
    n_groups = len(set(sp_.group))
    div = np.array([len(set(sp_.group[r])) for r in res.rec])
    cov = len(np.unique(res.rec)) / sp_.n_items
    for g, m in groups.items():
        if m.sum() == 0: continue
        rows.append({"mo_hinh": name, "nhom": g, "so_khach": int(m.sum()),
                     f"recall@{K}": res.recall[m].mean(), f"precision@{K}": res.precision[m].mean(),
                     f"ndcg@{K}": res.ndcg[m].mean(), f"map@{K}": res.ap[m].mean(), f"hit_rate@{K}": res.hit[m].mean(),
                     "coverage": cov, "da_dang_nganh": div[m].mean(), **(extra or {})})
    return pd.DataFrame(rows)

def paired_bootstrap(a, b, n=2000, seed=SEED):
    # khoảng tin cậy 95% cho chênh lệch trung bình (a - b) trên cùng một nhóm khách
    r = np.random.default_rng(seed); d = np.asarray(a) - np.asarray(b)
    means = [d[r.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

def make_scorer(known_model, cold_model=None):
    cold_model = cold_model or known_model
    def f(rows):
        S = np.zeros((len(rows), known_model.n), np.float32); kn = rows >= 0
        if kn.any(): S[kn] = known_model.score(rows[kn])
        if (~kn).any(): S[~kn] = cold_model.score(rows[~kn])
        return S
    return f

class Popularity:
    def __init__(self, days=30, by="buyers"): self.days, self.by = days, by
    def fit(self, sp_):
        t = sp_.train[(sp_.train.ngay >= sp_.start - pd.Timedelta(days=self.days)) & sp_.train.item.isin(sp_.item_idx)]
        s = t.groupby("item").customer_id.nunique() if self.by == "buyers" else t.groupby("item").qty.sum()
        v = np.zeros(sp_.n_items, np.float32)
        if len(s): v[s.index.map(sp_.item_idx).values.astype(int)] = s.values
        self.v, self.n = v, sp_.n_items
        return self
    def score(self, rows): return np.tile(self.v, (len(rows), 1))

class FirstTimer(Popularity):
    # mẫu mà khách MỚI hay mua trong lần mua đầu tiên của họ (phân phối khác với khách cũ)
    def __init__(self, days=60): self.days = days
    def fit(self, sp_):
        t = sp_.train
        first = t.groupby("customer_id").ngay.transform("min")
        t = t[(t.ngay == first) & (t.ngay >= sp_.start - pd.Timedelta(days=self.days)) & t.item.isin(sp_.item_idx)]
        s = t.groupby("item").customer_id.nunique()
        v = np.zeros(sp_.n_items, np.float32)
        if len(s): v[s.index.map(sp_.item_idx).values.astype(int)] = s.values
        self.v, self.n = v, sp_.n_items
        return self

class SegmentPopularity:
    def __init__(self, days=30, mix=0.7): self.days, self.mix = days, mix
    def fit(self, sp_):
        t = sp_.train.copy(); g = t.item.map(sp_.gender_map)
        t["girl"] = np.where(g == GENDER_GIRL, t.qty, 0.0); t["boy"] = np.where(g == GENDER_BOY, t.qty, 0.0)
        c = t.groupby("customer_id")[["girl", "boy"]].sum().reindex(sp_.users).fillna(0)
        self.seg = np.where(c.girl > c.boy, 0, np.where(c.boy > c.girl, 1, 2))
        rc = t[(t.ngay >= sp_.start - pd.Timedelta(days=self.days)) & t.item.isin(sp_.item_idx)].copy()
        rc["seg"] = rc.customer_id.map(dict(zip(sp_.users, self.seg)))
        self.v = np.zeros((3, sp_.n_items), np.float32)
        for sgm in range(3):
            s = rc[rc.seg == sgm].groupby("item").customer_id.nunique()
            if len(s): self.v[sgm, s.index.map(sp_.item_idx).values.astype(int)] = s.values
        self.glob = np.zeros(sp_.n_items, np.float32)
        s = rc.groupby("item").customer_id.nunique(); self.glob[s.index.map(sp_.item_idx).values.astype(int)] = s.values
        self.n = sp_.n_items
        return self
    def score(self, rows):
        gnorm = self.glob / (self.glob.max() + 1e-12); out = np.empty((len(rows), self.n), np.float32)
        for i, r in enumerate(rows):
            sg = self.seg[r]
            if sg == 2: out[i] = gnorm
            else:
                v = self.v[sg]; out[i] = self.mix * v / (v.max() + 1e-12) + (1 - self.mix) * gnorm
        return out

class ItemKNN:
    def __init__(self, k=50, tau=120): self.k, self.tau = k, tau
    def fit(self, sp_):
        X = sp_.matrix(self.tau); self.X, self.n = X, sp_.n_items
        norms = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel()) + 1e-9
        Xn = X @ sp.diags(1.0 / norms)
        S = (Xn.T @ Xn).toarray().astype(np.float32); np.fill_diagonal(S, 0)
        kk = min(self.k, S.shape[0] - 1)
        thr = -np.partition(-S, kk - 1, axis=1)[:, kk - 1][:, None]
        S[S < thr] = 0
        self.S = S
        return self
    def score(self, rows): return np.asarray(self.X[rows] @ self.S)

try:
    from implicit.als import AlternatingLeastSquares
    HAS_IMPLICIT = True
except Exception:
    HAS_IMPLICIT = False
    from sklearn.decomposition import TruncatedSVD
print("Latent model:", "ALS (implicit)" if HAS_IMPLICIT else "SVD (scikit-learn)")

class Latent:
    def __init__(self, factors=64, reg=0.05, alpha=20, iters=15, tau=120):
        self.factors, self.reg, self.alpha, self.iters, self.tau = factors, reg, alpha, iters, tau
    def fit(self, sp_):
        X = sp_.matrix(self.tau).copy(); self.n = sp_.n_items
        if HAS_IMPLICIT:
            Cm = X.copy(); Cm.data = 1.0 + self.alpha * Cm.data
            m = AlternatingLeastSquares(factors=self.factors, regularization=self.reg, iterations=self.iters,
                                        random_state=SEED, calculate_training_loss=False)
            m.fit(Cm.tocsr(), show_progress=False)
            self.U, self.V = np.asarray(m.user_factors), np.asarray(m.item_factors)
        else:
            svd = TruncatedSVD(n_components=min(self.factors, self.n - 1), random_state=SEED)
            self.U = svd.fit_transform(X); self.V = svd.components_.T
        return self
    def score(self, rows): return self.U[rows] @ self.V.T

class Hybrid:
    # điểm = (1 - w) * điểm mô hình cá nhân hóa (chuẩn hóa theo hàng) + w * độ phổ biến gần đây (chuẩn hóa)
    def __init__(self, model, pop, w=0.2): self.model, self.pop, self.w, self.n = model, pop, w, model.n
    def score(self, rows):
        pv = self.pop.v / (self.pop.v.max() + 1e-12)
        return (1 - self.w) * rownorm(self.model.score(rows)) + self.w * pv[None, :]

def rebuild(name, sp_):
    """Dựng lại mô hình từ tên cấu hình và huấn luyện trên split sp_."""
    if name.startswith("Hybrid["):
        inner, w = re.match(r"Hybrid\[(.*)\] w=([0-9.]+)", name).groups()
        return Hybrid(rebuild(inner, sp_), Popularity(30).fit(sp_), float(w))
    if name.startswith("ItemKNN"):
        k_, t_ = re.match(r"ItemKNN k=(\d+) tau=(\d+)", name).groups(); return ItemKNN(int(k_), int(t_)).fit(sp_)
    if name.startswith("Latent"):
        f_, r_, a_ = re.match(r"Latent f=(\d+) reg=([0-9.]+) a=(\d+)", name).groups(); return Latent(int(f_), float(r_), int(a_)).fit(sp_)
    if name == "SegmentPop": return SegmentPopularity(30, 0.7).fit(sp_)
    return Popularity(30).fit(sp_)

def make_cold(kind, sp_):
    return FirstTimer(60).fit(sp_) if "FirstTimer" in kind else Popularity(30).fit(sp_)
