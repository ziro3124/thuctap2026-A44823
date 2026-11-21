"""Đọc dữ liệu từ PostgreSQL và tạo feature cho bài toán gợi ý.

Quy tắc chống rò rỉ: mọi hàm tạo feature nhận `train` (chỉ gồm dòng có ts < ngày chốt) và `cutoff`,
và KHÔNG đọc dữ liệu nào khác. tests/test_leakage.py kiểm tra tính chất này bằng cách thêm dữ liệu tương lai
vào đầu vào rồi xác nhận feature không đổi.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C
from .db import query

VALID = "i.is_revenue_valid::text::boolean IS TRUE AND i.is_cancelled::text::boolean IS NOT TRUE"

# --------------------------------------------------------------------------- SQL
SQL_INTERACTIONS = f"""
SELECT NULLIF(TRIM(i.customer_id), '')              AS customer_id,
       s.item_no                                     AS item,
       i.invoice_id                                  AS invoice_id,
       i.invoice_date::timestamp                     AS ts,
       MAX(s.size_name)                              AS size_name,
       SUM(d.quantity::numeric)                      AS qty,
       SUM(d.total_amount::numeric)                  AS amount
FROM invoice_detail d
JOIN invoice i ON i.invoice_id = d.invoice_id
JOIN sku s     ON UPPER(TRIM(s.sku)) = UPPER(TRIM(d.item_code))
WHERE {VALID}
  AND d.quantity::numeric > 0
  AND NULLIF(TRIM(i.customer_id), '') IS NOT NULL
  AND i.invoice_date::timestamp >= DATE '2025-01-01' AND i.invoice_date::timestamp < DATE '2026-01-01'
GROUP BY 1, 2, 3, 4
"""

SQL_ITEM_STATS = f"""
SELECT s.item_no AS item,
       SUM(d.quantity::numeric)     FILTER (WHERE d.quantity::numeric > 0) AS qty,
       SUM(d.total_amount::numeric) FILTER (WHERE d.quantity::numeric > 0) AS rev,
       COUNT(DISTINCT i.invoice_id) FILTER (WHERE d.quantity::numeric > 0) AS n_inv,
       COUNT(*) AS n_dong,
       SUM(CASE WHEN COALESCE(d.line_type, 'item') <> 'item' THEN 1 ELSE 0 END) AS n_dong_khac_item
FROM invoice_detail d
JOIN invoice i ON i.invoice_id = d.invoice_id
JOIN sku s     ON UPPER(TRIM(s.sku)) = UPPER(TRIM(d.item_code))
WHERE {VALID}
GROUP BY 1
"""

SQL_N_INVOICES = f"SELECT COUNT(*) AS n FROM invoice i WHERE {VALID} AND i.actual_amount::numeric > 0"
SQL_ITEMS = "SELECT item, product_name, group_cat, gender, season, licenses, unit_price FROM item"
SQL_SKU = "SELECT item_no AS item, size_name FROM sku"


# --------------------------------------------------------------------------- nạp dữ liệu
def load_raw(use_cache: bool = True) -> dict:
    """Đọc dữ liệu thô từ PostgreSQL (hoặc cache parquet trong data/ nếu có)."""
    names = ["inter", "item_stats", "items", "sku", "meta"]
    paths = {n: C.DATA / f"raw_{n}.parquet" for n in names}
    if use_cache and all(p.exists() for p in paths.values()):
        raw = {n: pd.read_parquet(p) for n, p in paths.items()}
    else:
        raw = {
            "inter": query(SQL_INTERACTIONS), "item_stats": query(SQL_ITEM_STATS),
            "items": query(SQL_ITEMS), "sku": query(SQL_SKU),
            "meta": pd.DataFrame({"n_inv_total": [int(query(SQL_N_INVOICES).n[0])]}),
        }
        for n, p in paths.items():
            raw[n].to_parquet(p, index=False)
    raw["inter"]["ts"] = pd.to_datetime(raw["inter"]["ts"])
    for c in ("qty", "amount"):
        raw["inter"][c] = raw["inter"][c].astype(float)
    for n in ("inter", "item_stats", "items", "sku"):
        raw[n]["item"] = raw[n]["item"].astype(str)
    raw["inter"]["customer_id"] = raw["inter"]["customer_id"].astype(str)
    return raw


# --------------------------------------------------------------------------- mã không phải hàng hóa
def find_non_merch(item_stats: pd.DataFrame, n_inv_total: int, manual=frozenset()) -> tuple[set, pd.DataFrame]:
    """Quy tắc tự động: doanh thu <= 0, hoặc giá trung bình < 5.000đ, hoặc > 50% dòng không phải 'item'.

    Trả về (tập mã bị loại, bảng thống kê để con người kiểm tra lại)."""
    s = item_stats.copy()
    s["gia_tb"] = s.rev / s.qty.replace(0, np.nan)
    s["ty_le_hd_pct"] = 100 * s.n_inv / max(n_inv_total, 1)
    s["pct_dong_khac_item"] = s.n_dong_khac_item / s.n_dong
    s["tu_dong_loai"] = (s.rev.fillna(0) <= 0) | (s.gia_tb < 5000) | (s.pct_dong_khac_item > 0.5)
    return set(s.loc[s.tu_dong_loai, "item"]) | set(manual), s


# --------------------------------------------------------------------------- size -> tuổi
SIZE_RULES = [
    (re.compile(r"^(\d+(?:\.\d+)?)\s*Y(?:-\d+-\d+kg)?$", re.I), lambda m: float(m.group(1))),            # 4Y, 1Y-9-11kg
    (re.compile(r"^(\d+)\s*M$", re.I), lambda m: float(m.group(1)) / 12),                                  # 18M
    (re.compile(r"^(\d+)\s*-\s*(\d+)\s*Y$", re.I), lambda m: (float(m.group(1)) + float(m.group(2))) / 2),  # 4-5Y
    (re.compile(r"^Size\s*\d+\s*-\s*(\d+(?:\.\d+)?)\s*cm$", re.I),                                          # giày: chiều dài chân (cm)
     lambda m: max(0.0, (float(m.group(1)) - 11.5) * 0.9)),
]


def size_to_age(name) -> float:
    """Quy size ra tuổi xấp xỉ (năm). Không nhận ra -> NaN (không dùng để lọc).
    Bổ sung quy tắc vào SIZE_RULES theo danh sách trong reports/size_chua_quy_doi.csv."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return np.nan
    s = str(name).strip()
    for rx, fn in SIZE_RULES:
        m = rx.match(s)
        if m:
            return float(fn(m))
    return np.nan


def item_age_table(sku: pd.DataFrame) -> pd.DataFrame:
    """Khoảng tuổi (nhỏ nhất, lớn nhất) mà mỗi mẫu có size phù hợp."""
    t = sku.copy()
    t["age"] = t.size_name.map(size_to_age)
    g = t.dropna(subset=["age"]).groupby("item").age.agg(age_min="min", age_max="max")
    g["age_mid"] = (g.age_min + g.age_max) / 2
    return g


def unmapped_sizes(sku: pd.DataFrame) -> pd.DataFrame:
    u = sku.assign(age=sku.size_name.map(size_to_age))
    u = u[u.age.isna()].groupby("size_name").size().rename("so_sku").reset_index().sort_values("so_sku", ascending=False)
    return u


# --------------------------------------------------------------------------- mùa
def _norm_season(s) -> str:
    return re.sub(r"\s*-\s*", "-", str(s).strip().lower())


def season_of(date: pd.Timestamp) -> str:
    return "spring-summer" if date.month in C.SPRING_SUMMER_MONTHS else "fall-winter"


# --------------------------------------------------------------------------- feature theo ngày chốt
def build_vocab(items: pd.DataFrame) -> dict:
    """Bảng mã hóa cố định cho thuộc tính phân loại (tính từ danh mục, không phụ thuộc giao dịch)."""
    v = {}
    for col in ("group_cat", "gender", "season", "licenses"):
        vals = items[col].fillna("(blank)").astype(str).replace("", "(blank)")
        v[col] = {x: i for i, x in enumerate(sorted(vals.unique()))}
    return v


def item_features(items_active: np.ndarray, items_df: pd.DataFrame, age_tab: pd.DataFrame, vocab: dict,
                  train: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Feature của từng mẫu, chỉ dùng `train` (ts < cutoff). Thứ tự dòng = items_active."""
    assert train.empty or train.ts.max() < cutoff, "rò rỉ: train chứa dữ liệu từ ngày chốt trở đi"
    meta = items_df.drop_duplicates("item").set_index("item").reindex(items_active)
    f = pd.DataFrame(index=pd.Index(items_active, name="item"))
    for col, key in (("group_cat", "group_code"), ("gender", "gender_code"), ("season", "season_code"), ("licenses", "lic_code")):
        vals = meta[col].fillna("(blank)").astype(str).replace("", "(blank)")
        f[key] = vals.map(vocab[col]).fillna(-1).astype(int).values
    f["price"] = pd.to_numeric(meta.unit_price, errors="coerce").values
    f["log_price"] = np.log1p(f.price.fillna(f.price.median()))
    cur = season_of(cutoff - pd.Timedelta(days=1))
    f["season_match"] = [1.0 if (_norm_season(s) == cur or _norm_season(s) == "year-round") else 0.0 for s in meta.season.values]
    f["is_girl_item"] = (meta.gender.astype(str) == "Bé gái").astype(float).values
    f["is_boy_item"] = (meta.gender.astype(str) == "Bé trai").astype(float).values
    at = age_tab.reindex(items_active)
    f["age_min"], f["age_max"], f["age_mid"] = at.age_min.values, at.age_max.values, at.age_mid.values
    t = train[train.item.isin(set(items_active))]
    for days in (7, 30, 90):
        w = t[t.ts >= cutoff - pd.Timedelta(days=days)]
        f[f"buyers_{days}d"] = w.groupby("item").customer_id.nunique().reindex(items_active).fillna(0).values
    f["log_buyers_all"] = np.log1p(t.groupby("item").customer_id.nunique().reindex(items_active).fillna(0).values)
    first = t.groupby("item").ts.min().reindex(items_active)
    f["days_since_first_sale"] = (cutoff - first).dt.days.fillna(0).values
    f["trend"] = (f.buyers_7d * 4.0 + 1) / (f.buyers_30d + 1)        # đang lên (>1) hay xuống (<1)
    return f


def user_features(users: np.ndarray, items_df: pd.DataFrame, age_tab: pd.DataFrame, train: pd.DataFrame,
                  cutoff: pd.Timestamp) -> pd.DataFrame:
    """Feature của từng khách, chỉ dùng `train`. Thứ tự dòng = users."""
    assert train.empty or train.ts.max() < cutoff, "rò rỉ: train chứa dữ liệu từ ngày chốt trở đi"
    meta = items_df.drop_duplicates("item").set_index("item")
    t = train[train.customer_id.isin(set(users))].copy()
    g = t.groupby("customer_id")
    f = pd.DataFrame(index=pd.Index(users, name="customer_id"))
    f["n_inv"] = g.invoice_id.nunique().reindex(users).fillna(0).values
    f["n_items"] = t.drop_duplicates(["customer_id", "item"]).groupby("customer_id").size().reindex(users).fillna(0).values
    f["days_since_last"] = (cutoff - g.ts.max().reindex(users)).dt.days.fillna(9999).values
    f["days_since_first"] = (cutoff - g.ts.min().reindex(users)).dt.days.fillna(0).values
    amt = g.amount.sum().reindex(users) / g.qty.sum().reindex(users)
    f["avg_price"] = amt.values
    gen = t.item.map(meta.gender)
    pairs = t.assign(girl=(gen == "Bé gái").astype(float), boy=(gen == "Bé trai").astype(float)).drop_duplicates(["customer_id", "item"])
    gb = pairs.groupby("customer_id")[["girl", "boy"]].sum().reindex(users).fillna(0)
    f["girl_cnt"], f["boy_cnt"] = gb.girl.values, gb.boy.values
    tot = (gb.girl + gb.boy).replace(0, np.nan)
    f["girl_share"] = (gb.girl / tot).fillna(0.5).values
    f["seg"] = np.where(gb.girl > gb.boy, 0, np.where(gb.boy > gb.girl, 1, 2))      # 0 bé gái, 1 bé trai, 2 chưa rõ
    # tuổi ước tính của bé hiện tại = tuổi theo size đã mua ở 2 hóa đơn gần nhất + thời gian đã trôi qua
    t["age_row"] = t.size_name.map(size_to_age)
    t = t.dropna(subset=["age_row"])
    if len(t):
        order = t.groupby("customer_id").ts.rank(method="dense", ascending=False)
        t = t[order <= 2]
        t["age_now"] = t.age_row + (cutoff - t.ts).dt.days / 365.25
        f["age_est"] = t.groupby("customer_id").age_now.median().reindex(users).values
    else:
        f["age_est"] = np.nan
    return f


def size_compatible(item_age_min, item_age_max, user_age, lo: float = C.SIZE_FILTER_LO, hi: float = C.SIZE_FILTER_HI):
    """True nếu mẫu có size hợp tuổi bé; thiếu thông tin thì không loại (True). Nhận mảng numpy cùng shape."""
    imin, imax, ua = (np.asarray(x, dtype=float) for x in (item_age_min, item_age_max, user_age))
    unknown = np.isnan(imin) | np.isnan(imax) | np.isnan(ua)
    ok = (imax >= ua - lo) & (imin <= ua + hi)
    return np.where(unknown, True, ok)
