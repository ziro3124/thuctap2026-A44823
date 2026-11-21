"""Chia dữ liệu theo thời gian và dựng "ngữ cảnh" (Ctx) cho một ngày chốt.

Quy tắc (biến thể có ngày chốt của đề bài "giữ hóa đơn cuối cùng làm tập kiểm tra"):
  - train      = mọi giao dịch có ts < cutoff
  - kiểm tra   = khách có lịch sử trước cutoff VÀ có mua trong [cutoff, end); nhãn = sản phẩm của HÓA ĐƠN CUỐI CÙNG
                 của khách trong khoảng đó. Như vậy mỗi khách kiểm tra có ít nhất 2 lần mua (1 trước, 1 làm nhãn).
  - Ngày chốt chung cho tất cả khách để không dùng hành vi tương lai của người khác (xem README, mục chống rò rỉ).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config as C
from . import features as F


@dataclass
class Ctx:
    cutoff: pd.Timestamp
    end: pd.Timestamp | None
    train: pd.DataFrame                      # giao dịch < cutoff (đã bỏ mã không phải hàng hóa)
    items: np.ndarray                        # mẫu được phép gợi ý (còn bán trong ACTIVE_DAYS ngày cuối)
    item_idx: dict
    users: np.ndarray                        # mọi khách có lịch sử trước cutoff
    user_idx: dict
    seen: sp.csr_matrix                      # khách x mẫu đã mua (để loại khỏi gợi ý)
    agg: pd.DataFrame                        # (khách, mẫu): số lượng, lần mua cuối, số hóa đơn
    item_meta: pd.DataFrame                  # feature mẫu, hàng = items
    user_meta: pd.DataFrame                  # feature khách, hàng = users
    names: np.ndarray
    groups: np.ndarray
    ug: np.ndarray                           # khách x nhóm ngành hàng: tỷ lệ mẫu đã mua
    ul: np.ndarray                           # khách x bản quyền: tỷ lệ mẫu đã mua
    eval_users: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    rows: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))   # chỉ số trong users
    target: sp.csr_matrix | None = None      # khách kiểm tra x mẫu (nhãn nằm trong không gian mẫu gợi ý được)
    target_size: np.ndarray | None = None    # số mẫu thật sự mua ở hóa đơn cuối (kể cả mẫu không gợi ý được)
    ceiling: float = float("nan")
    _mat: dict = field(default_factory=dict)

    @property
    def n_items(self): return len(self.items)

    def matrix(self, tau: float | None = None) -> sp.csr_matrix:
        """Ma trận khách x mẫu: log(1 + số lượng), giảm dần theo thời gian (tau ngày) nếu có."""
        if tau not in self._mat:
            w = np.log1p(self.agg.qty.values)
            if tau:
                w = w * np.exp(-(self.cutoff - self.agg["last"]).dt.days.values / tau)
            r = self.agg.customer_id.map(self.user_idx).values.astype(int)
            c = self.agg.item.map(self.item_idx).values.astype(int)
            self._mat[tau] = sp.csr_matrix((w.astype(np.float32), (r, c)), shape=(len(self.users), self.n_items))
        return self._mat[tau]

    def summary(self) -> str:
        s = f"cutoff {self.cutoff:%d/%m/%Y}: train {len(self.train):,} dòng, {len(self.users):,} khách, {self.n_items:,} mẫu gợi ý được"
        if self.end is not None:
            s += f" | kiểm tra {len(self.eval_users):,} khách, trần Recall {self.ceiling:.3f}"
        return s


def build_ctx(inter: pd.DataFrame, items_df: pd.DataFrame, age_tab: pd.DataFrame, vocab: dict,
              cutoff, end=None, active_days: int = C.ACTIVE_DAYS) -> Ctx:
    cutoff = pd.Timestamp(cutoff)
    end = pd.Timestamp(end) if end is not None else None
    train = inter[inter.ts < cutoff]
    recent = train[train.ts >= cutoff - pd.Timedelta(days=active_days)]
    items = np.array(sorted(recent.item.unique()))
    item_idx = {x: i for i, x in enumerate(items)}
    users = np.array(sorted(train.customer_id.unique()))
    user_idx = {u: i for i, u in enumerate(users)}

    tc = train[train.item.isin(item_idx)]
    agg = tc.groupby(["customer_id", "item"]).agg(qty=("qty", "sum"), last=("ts", "max"), n=("invoice_id", "nunique")).reset_index()
    seen = sp.csr_matrix((np.ones(len(agg), np.float32),
                          (agg.customer_id.map(user_idx).values.astype(int), agg.item.map(item_idx).values.astype(int))),
                         shape=(len(users), len(items)))

    im = F.item_features(items, items_df, age_tab, vocab, train, cutoff)
    um = F.user_features(users, items_df, age_tab, train, cutoff)
    cat = items_df.drop_duplicates("item").set_index("item")
    names = cat.product_name.reindex(items).fillna("").values
    groups = cat.group_cat.reindex(items).fillna("(không rõ)").values

    def share(codes, n):
        oh = sp.csr_matrix((np.ones(len(items)), (np.arange(len(items)), np.clip(codes, 0, n - 1))), shape=(len(items), n))
        cnt = (seen @ oh).toarray()
        return cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
    ug = share(im.group_code.values, len(vocab["group_cat"]))
    ul = share(im.lic_code.values, len(vocab["licenses"]))

    ctx = Ctx(cutoff, end, train, items, item_idx, users, user_idx, seen, agg, im, um, names, groups, ug.astype(np.float32), ul.astype(np.float32))
    if end is not None:
        win = inter[(inter.ts >= cutoff) & (inter.ts < end)]
        win = win[win.customer_id.isin(set(users))]
        # hóa đơn cuối cùng của mỗi khách trong khoảng [cutoff, end)
        last = win.sort_values(["ts", "invoice_id"]).groupby("customer_id").tail(1)[["customer_id", "invoice_id"]]
        tgt = win.merge(last, on=["customer_id", "invoice_id"])[["customer_id", "item"]].drop_duplicates()
        ctx.eval_users = np.array(sorted(tgt.customer_id.unique()))
        ctx.rows = np.array([user_idx[u] for u in ctx.eval_users], dtype=int)
        ctx.target_size = tgt.groupby("customer_id").item.nunique().reindex(ctx.eval_users).values.astype(int)
        inspace = tgt[tgt.item.isin(item_idx)]
        r = pd.Index(ctx.eval_users).get_indexer(inspace.customer_id)
        c = inspace.item.map(item_idx).values.astype(int)
        ctx.target = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(ctx.eval_users), len(items)))
        ctx.ceiling = len(inspace) / max(1, len(tgt))
    assert_no_leakage(ctx)
    return ctx


def assert_no_leakage(ctx: Ctx) -> None:
    """Kiểm tra tự động mỗi lần dựng ngữ cảnh: train hoàn toàn trước ngày chốt, nhãn hoàn toàn từ ngày chốt trở đi."""
    assert ctx.train.ts.max() < ctx.cutoff, "train chứa dữ liệu từ ngày chốt trở đi"
    assert (ctx.agg["last"] < ctx.cutoff).all(), "ma trận tương tác chứa dữ liệu từ ngày chốt trở đi"
    if ctx.end is not None:
        assert ctx.end > ctx.cutoff
        # không khách nào có nhãn mà không có lịch sử trước ngày chốt
        assert set(ctx.eval_users) <= set(ctx.users), "khách kiểm tra không có lịch sử"
