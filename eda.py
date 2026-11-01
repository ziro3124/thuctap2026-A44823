#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eda.py - Tuần 5: Khai phá dữ liệu (EDA) doanh thu bán lẻ thời trang trẻ em năm 2025
====================================================================================

Trả lời câu hỏi của Ban giám đốc: "Năm 2025 bán cho ai, bán gì, ở đâu, khi nào
và có vấn đề gì cần xử lý?" - theo 7 chủ đề của đề bài Tuần 5:

    A. Mùa vụ và thời gian        E. Khách hàng thành viên
    B. Điểm bán                   F. Trả hàng
    C. Sản phẩm                   G. Online và offline
    D. Giá và khuyến mãi

Cách chạy
---------
    python eda.py                      # chạy cả 7 chủ đề
    python eda.py --topics A B E       # chỉ chạy một số chủ đề
    python eda.py --refresh            # bỏ cache, truy vấn lại PostgreSQL
    python eda.py --show               # hiện biểu đồ trên màn hình (mặc định chỉ lưu file)

Thư viện:  pandas numpy matplotlib scipy sqlalchemy psycopg2-binary python-dotenv

Cấu hình kết nối (file .env hoặc biến môi trường - KHÔNG ghi mật khẩu vào code):
    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD, PGSCHEMA=masked_data

Đầu ra
------
    data/eda_cache/*.csv            dữ liệu tổng hợp từ SQL (đã thêm data/ vào .gitignore)
    reports/figures/*.png           biểu đồ, tiêu đề là kết luận (số liệu tính từ dữ liệu)
    reports/eda_tables/*.csv        các bảng số liệu phía sau biểu đồ
    reports/eda_insights.md         insight có số liệu + kết quả kiểm định, để dán vào báo cáo

Quy ước dữ liệu (nhất quán với notebook Tuần 4)
------------------------------------------------
    * Hóa đơn hợp lệ : is_revenue_valid = TRUE và is_cancelled không TRUE.
    * Doanh thu      : actual_amount của hóa đơn hợp lệ. Hóa đơn trả hàng có số tiền âm,
                       nên tổng của cả hai là DOANH THU THUẦN. "Đơn bán" = actual_amount > 0.
    * Mọi cột trong DB đang là text -> ép kiểu trong SQL (::float8, ::timestamp).
    * Đọc DB theo hướng "tổng hợp bằng SQL trước, pandas sau": invoice_detail (730 nghìn dòng)
      không bao giờ được kéo nguyên bảng về máy.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")          # chạy không cần màn hình
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------------------
# 0. Cấu hình
# --------------------------------------------------------------------------------------
DATA = Path("data") / "eda_cache"
REPORTS = Path("reports")
FIG = REPORTS / "figures"
TAB = REPORTS / "eda_tables"

YEAR_START, YEAR_END = "2025-01-01", "2026-01-01"
ONLINE_CHANNELS = {"shopee", "tiktok", "adayroi", "haravan"}      # sale_channel (so sánh không phân biệt hoa/thường)
LEVELS = ["Thỏ Nhí", "Thỏ Bạc", "Thỏ Vàng", "Thỏ Bạch Kim", "Thỏ Kim Cương"]
NO_LICENSE = {"không rõ", "none", "nan", "n/a", "na", "null", "no license", "không", "-", "0"}
DOW = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
G_RETAIL, G_ONLINE, G_OTHER = "Cửa hàng bán lẻ", "Online", "Khác (sỉ/sự kiện/không rõ)"

# Các dịp đáng chú ý của 2025: (bắt đầu, kết thúc, nhãn ngắn)
EVENTS = {
    "Trước Tết (15–28/1)":        ("2025-01-15", "2025-01-28", "Trước Tết"),
    "Nghỉ Tết (29/1–4/2)":        ("2025-01-29", "2025-02-04", "Nghỉ Tết"),
    "1/6 Thiếu nhi (25/5–1/6)":   ("2025-05-25", "2025-06-01", "1/6"),
    "Tựu trường (15/8–5/9)":      ("2025-08-15", "2025-09-05", "Tựu trường"),
    "11/11 (9–11/11)":            ("2025-11-09", "2025-11-11", "11/11"),
    "Black Friday (24–30/11)":    ("2025-11-24", "2025-11-30", "Black Fri"),
    "Noel – cuối năm (15–25/12)": ("2025-12-15", "2025-12-25", "Noel"),
}

C1, C2, C3, C4 = "#2a6f97", "#e76f51", "#8ab17d", "#adb5bd"
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "figure.dpi": 100,
})

INSIGHTS: list[tuple[str, str]] = []      # (chủ đề, nội dung)
TESTS: list[dict] = []                     # kết quả kiểm định
FIGURES: list[str] = []


# --------------------------------------------------------------------------------------
# 1. Kết nối DB + SQL
# --------------------------------------------------------------------------------------
_engine = None


def get_engine():
    """Tạo engine từ biến môi trường (giống notebook Tuần 4). Chỉ gọi khi cần truy vấn."""
    global _engine
    if _engine is None:
        from dotenv import load_dotenv
        from sqlalchemy import create_engine
        load_dotenv()
        url = "postgresql+psycopg2://{u}:{p}@{h}:{port}/{db}".format(
            u=os.environ["PGUSER"], p=os.environ["PGPASSWORD"], h=os.environ["PGHOST"],
            port=os.environ.get("PGPORT", "5432"), db=os.environ["PGDATABASE"])
        schema = os.environ.get("PGSCHEMA", "masked_data")
        _engine = create_engine(url, connect_args={"options": f"-csearch_path={schema},public"})
    return _engine


def q(sql: str) -> pd.DataFrame:
    from sqlalchemy import text
    with get_engine().connect() as con:
        return pd.read_sql(text(sql), con)


def load(name: str, sql: str, refresh: bool = False) -> pd.DataFrame:
    """Đọc từ cache CSV nếu có; nếu không thì truy vấn SQL rồi lưu cache.
    (Dùng CSV để không phụ thuộc pyarrow. Kiểu dữ liệu được ép lại trong prepare().)"""
    path = DATA / f"{name}.csv"
    if path.exists() and not refresh:
        print(f"  [cache] {name}")
        return pd.read_csv(path, low_memory=False)
    print(f"  [sql]   {name} ...")
    df = q(sql)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


VALID = "i.is_revenue_valid::text::boolean IS TRUE AND i.is_cancelled::text::boolean IS NOT TRUE"
IN_YEAR = f"i.invoice_date::timestamp >= DATE '{YEAR_START}' AND i.invoice_date::timestamp < DATE '{YEAR_END}'"

# Mức hóa đơn (~250 nghìn dòng): nhẹ, đủ cho chủ đề A, B, D, E, G
SQL_INVOICE = f"""
SELECT i.invoice_id,
       i.invoice_date::timestamp                                   AS ts,
       i.invoice_type::text                                        AS invoice_type,
       i.branch_code,
       COALESCE(NULLIF(TRIM(i.sale_channel::text), ''), 'Không rõ') AS sale_channel,
       NULLIF(TRIM(i.customer_id::text), '')                       AS customer_id,
       NULLIF(TRIM(i.member_level::text), '')                      AS member_level,
       i.actual_amount::float8                                     AS doanh_thu,
       i.total_item_amount::float8                                 AS tien_hang,
       i.discount_amount::float8                                   AS giam_gia,
       i.is_cod::text::boolean                                     AS is_cod,
       b.location_name, b.channel AS branch_channel, b.area, b.door_type, b.status AS branch_status
FROM invoice i
LEFT JOIN branch b ON b.location_code = i.branch_code
WHERE {VALID} AND {IN_YEAR}"""

# Số lượng bán / trả theo hóa đơn -> tính UPT
SQL_UPT = f"""
WITH l AS (
  SELECT d.invoice_id, NULLIF(TRIM(d.quantity::text), '')::float8 AS q
  FROM invoice_detail d JOIN invoice i ON i.invoice_id = d.invoice_id
  WHERE {VALID} AND {IN_YEAR})
SELECT invoice_id,
       COALESCE(SUM(q) FILTER (WHERE q > 0), 0)  AS so_luong_ban,
       COALESCE(-SUM(q) FILTER (WHERE q < 0), 0) AS so_luong_tra
FROM l GROUP BY invoice_id"""

SQL_QUALITY = f"""
SELECT COUNT(*) AS tong_hd,
       COUNT(*) FILTER (WHERE {VALID}) AS hd_hop_le
FROM invoice i"""

# Dòng chi tiết đã chuẩn hóa mã hàng (TRIM/UPPER) như Tuần 4
LINES_CTE = f"""
WITH l AS (
  SELECT UPPER(TRIM(d.item_code)) AS code,
         NULLIF(TRIM(d.quantity::text), '')::float8     AS q,
         NULLIF(TRIM(d.total_amount::text), '')::float8 AS a
  FROM invoice_detail d JOIN invoice i ON i.invoice_id = d.invoice_id
  WHERE {VALID} AND {IN_YEAR})"""

# Mức item (~13 nghìn dòng): chủ đề C và F
SQL_ITEM = LINES_CTE + """
SELECT COALESCE(s.item_no::text, 'Không rõ')     AS item,
       it.product_name, it.group_cat, it.gender, it.season, it.licenses, it.product_type,
       COALESCE(SUM(l.q) FILTER (WHERE l.q > 0), 0)      AS ban_sl,
       COALESCE(SUM(l.a) FILTER (WHERE l.q > 0), 0)      AS ban_dt,
       COALESCE(-SUM(l.q) FILTER (WHERE l.q < 0), 0)     AS tra_sl,
       COALESCE(SUM(ABS(l.a)) FILTER (WHERE l.q < 0), 0) AS tra_dt
FROM l
LEFT JOIN sku s  ON UPPER(TRIM(s.sku)) = l.code
LEFT JOIN item it ON it.item::text = s.item_no::text
GROUP BY 1, 2, 3, 4, 5, 6, 7"""

# Theo ngành hàng x size: chủ đề F
SQL_SIZE = LINES_CTE + """
SELECT it.group_cat, s.size_name,
       COALESCE(SUM(l.q) FILTER (WHERE l.q > 0), 0)  AS ban_sl,
       COALESCE(-SUM(l.q) FILTER (WHERE l.q < 0), 0) AS tra_sl
FROM l
LEFT JOIN sku s  ON UPPER(TRIM(s.sku)) = l.code
LEFT JOIN item it ON it.item::text = s.item_no::text
GROUP BY 1, 2"""


# --------------------------------------------------------------------------------------
# 2. Hàm tiện ích
# --------------------------------------------------------------------------------------
def note(topic: str, text: str) -> None:
    """Ghi một insight (in ra màn hình và lưu vào eda_insights.md)."""
    INSIGHTS.append((topic, text))
    print(f"  -> [{topic}] {text}")


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> None:
    df.to_csv(TAB / f"{name}.csv", index=index, encoding="utf-8-sig")


def save_fig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    FIGURES.append(f"{name}.png")
    if "--show" in sys.argv:
        plt.show()
    plt.close(fig)


def title(ax, text: str) -> None:
    ax.set_title(text, loc="left", wrap=True)


def fmt_p(p: float) -> str:
    return "< 0,001" if p < 0.001 else f"{p:.3f}"


def mann_whitney(a: pd.Series, b: pd.Series, topic: str, question: str, h0: str, h1: str,
                 name_a: str, name_b: str) -> None:
    """Mann–Whitney U hai phía + hệ số tương quan hạng (rank-biserial) làm cỡ ảnh hưởng.
    n rất lớn nên p gần như luôn nhỏ; hãy đọc kèm trung vị và cỡ ảnh hưởng."""
    a, b = a.dropna(), b.dropna()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 2 * u / (len(a) * len(b)) - 1
    verdict = "bác bỏ H0" if p < 0.05 else "chưa đủ bằng chứng bác bỏ H0"
    TESTS.append({
        "chu_de": topic, "cau_hoi": question, "H0": h0, "H1": h1, "kiem_dinh": "Mann–Whitney U (hai phía)",
        "so_lieu": (f"{name_a}: n={len(a):,}, trung vị={a.median():,.0f}, TB={a.mean():,.0f} | "
                    f"{name_b}: n={len(b):,}, trung vị={b.median():,.0f}, TB={b.mean():,.0f}"),
        "thong_ke": f"U={u:,.0f}; p {fmt_p(p)}; rank-biserial r={r:+.2f}",
        "ket_luan": f"{verdict} (mức 5%). r>0 nghĩa là {name_a} có xu hướng lớn hơn {name_b}.",
    })
    print(f"  => Kiểm định [{topic}] {question}: p {fmt_p(p)}, r={r:+.2f} -> {verdict}")


# --------------------------------------------------------------------------------------
# 3. Chuẩn bị dữ liệu
# --------------------------------------------------------------------------------------
def prepare(inv: pd.DataFrame, upt: pd.DataFrame) -> pd.DataFrame:
    d = inv.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    d["ngay"] = d.ts.dt.normalize()
    d["thang"] = d.ts.dt.month
    d["gio"] = d.ts.dt.hour
    d["thu"] = d.ts.dt.dayofweek                      # 0 = Thứ Hai
    d = d.merge(upt, on="invoice_id", how="left")
    d[["so_luong_ban", "so_luong_tra"]] = d[["so_luong_ban", "so_luong_tra"]].fillna(0.0)
    d["is_cod"] = d["is_cod"].fillna(False).astype(bool)
    d["co_khach"] = d.customer_id.notna()
    for c in ["location_name", "branch_channel", "area", "door_type", "branch_status"]:
        d[c] = d[c].fillna("Không rõ")
    d["location_name"] = d.location_name.where(d.location_name != "Không rõ", d.branch_code.astype(str))
    d["kenh_nhom"] = np.select(
        [d.sale_channel.str.lower().isin(ONLINE_CHANNELS), d.branch_channel.eq("Retail")],
        [G_ONLINE, G_RETAIL], default=G_OTHER)
    return d


# --------------------------------------------------------------------------------------
# 4. Tổng quan dữ liệu
# --------------------------------------------------------------------------------------
def overview(inv: pd.DataFrame, quality: pd.DataFrame) -> None:
    print("\n=== 0. TỔNG QUAN ===")
    T = "0"
    qa = quality.iloc[0]
    ban, tra = inv[inv.doanh_thu > 0], inv[inv.doanh_thu < 0]
    net, gross, ret = inv.doanh_thu.sum(), ban.doanh_thu.sum(), -tra.doanh_thu.sum()
    note(T, f"Dùng {len(inv):,}/{int(qa.tong_hd):,} hóa đơn hợp lệ ({100*len(inv)/qa.tong_hd:.1f}%), "
            f"ngày {inv.ts.min():%d/%m/%Y} – {inv.ts.max():%d/%m/%Y}.")
    note(T, f"Doanh thu bán {gross/1e9:,.1f} tỷ đồng; trả hàng {ret/1e9:,.2f} tỷ ({100*ret/gross:.1f}% doanh thu bán); "
            f"doanh thu thuần {net/1e9:,.1f} tỷ.")
    note(T, f"Hóa đơn trả hàng (số tiền âm): {len(tra):,} ({100*len(tra)/len(inv):.1f}% số hóa đơn).")
    p99, mx = ban.doanh_thu.quantile(.99), ban.doanh_thu.max()
    n_out = int((ban.doanh_thu > p99).sum())
    note(T, f"Ngoại lai: AOV trung vị {ban.doanh_thu.median():,.0f}đ, P99 {p99:,.0f}đ, lớn nhất {mx:,.0f}đ. "
            f"Không loại ngoại lai (có thể là đơn sỉ thật); dùng trung vị và kiểm định phi tham số.")
    bad = int((inv.giam_gia > inv.tien_hang.clip(lower=0)).sum())
    note(T, f"{bad:,} hóa đơn có giảm giá lớn hơn tiền hàng -> khi tính tỷ lệ giảm giá bị chặn ở 100%.")

    tbl = (inv.groupby("invoice_type")
              .agg(so_hd=("invoice_id", "count"), so_hd_am=("doanh_thu", lambda s: int((s < 0).sum())),
                   doanh_thu=("doanh_thu", "sum")))
    save_table(tbl, "0_loai_chung_tu")
    print(tbl.to_string(float_format=lambda v: f"{v:,.0f}"))


# --------------------------------------------------------------------------------------
# A. Mùa vụ và thời gian
# --------------------------------------------------------------------------------------
def heat(ax, df: pd.DataFrame, text: str) -> None:
    p = (df.pivot_table(index="thu", columns="gio", values="doanh_thu", aggfunc="sum")
           .reindex(index=range(7), columns=range(24)).fillna(0.0))
    s = p.values.sum()
    if s > 0:
        p = 100 * p / s
    im = ax.imshow(p.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(7)); ax.set_yticklabels(DOW)
    ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("Giờ trong ngày")
    ax.grid(False); title(ax, text)
    plt.colorbar(im, ax=ax, label="% doanh thu")


def topic_A(inv: pd.DataFrame) -> None:
    print("\n=== A. MÙA VỤ VÀ THỜI GIAN ===")
    T = "A"
    full = pd.date_range(YEAR_START, inv.ngay.max())   # đến ngày cuối có dữ liệu (tránh tính ngày chưa có số liệu thành 0)
    daily = inv.groupby("ngay").doanh_thu.sum().reindex(full, fill_value=0.0)
    monthly = inv.groupby("thang").doanh_thu.sum().reindex(range(1, 13), fill_value=0.0)
    total = monthly.sum()
    peak_m, peak_share = int(monthly.idxmax()), 100 * monthly.max() / total
    low_m = int(monthly.idxmin())
    q4 = 100 * monthly.loc[10:12].sum() / total
    save_table((monthly / 1e6).rename("doanh_thu_trieu").to_frame(), "A_doanh_thu_thang")

    # Hiệu ứng dịp lễ: doanh thu/ngày trong dịp so với "nền" (các ngày ngoài mọi dịp)
    in_event = np.zeros(len(full), dtype=bool)
    for s, e, _ in EVENTS.values():
        in_event |= (full >= s) & (full <= e)
    base = daily[~in_event].mean()
    rows = []
    for name, (s, e, _) in EVENTS.items():
        m = (full >= s) & (full <= e)
        rows.append({"dip": name, "doanh_thu_ngay_tb_trieu": daily[m].mean() / 1e6, "so_voi_nen": daily[m].mean() / base})
    ev = pd.DataFrame(rows).set_index("dip")
    save_table(ev, "A_hieu_ung_dip_le")

    retail = inv[inv.kenh_nhom == G_RETAIL]
    dow = retail.groupby("thu").doanh_thu.sum().reindex(range(7), fill_value=0.0)
    dow_share = 100 * dow / dow.sum()
    weekend = dow_share.loc[5:6].sum()
    hour = retail.groupby("gio").doanh_thu.sum()
    top3 = hour.nlargest(3).index.sort_values().tolist()
    top3_txt = ", ".join(f"{h}h" for h in top3)
    top3_share = 100 * hour.nlargest(3).sum() / hour.sum()

    note(T, f"Tháng {peak_m} cao nhất ({monthly.max()/1e9:,.1f} tỷ, {peak_share:.1f}% cả năm); tháng {low_m} thấp nhất "
            f"({monthly.min()/1e9:,.1f} tỷ). Quý 4 chiếm {q4:.1f}% doanh thu năm.")
    best = ev.so_voi_nen.idxmax(); worst = ev.so_voi_nen.idxmin()
    note(T, f"Dịp mạnh nhất: {best} - doanh thu/ngày gấp {ev.so_voi_nen.max():.2f} lần ngày thường; "
            f"dịp yếu nhất: {worst} ({ev.so_voi_nen.min():.2f} lần).")
    note(T, f"Cửa hàng bán lẻ: T7+CN chiếm {weekend:.1f}% doanh thu (mức đều là 28,6%); 3 giờ cao điểm "
            f"({top3_txt}) chiếm {top3_share:.1f}%.")

    # A1: đường doanh thu theo ngày + dịp lễ
    fig, ax = plt.subplots(figsize=(13, 4.2))
    s7 = (daily / 1e6).rolling(7, center=True).mean()
    ax.plot(full, daily / 1e6, color=C1, lw=.7, alpha=.45, label="Theo ngày")
    ax.plot(full, s7, color=C1, lw=2.2, label="Trung bình 7 ngày")
    ymax = (daily / 1e6).max()
    for s, e, short in EVENTS.values():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=C2, alpha=.15)
        ax.text(pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2, ymax, short,
                rotation=90, ha="center", va="top", fontsize=8)
    ax.set_ylabel("Doanh thu (triệu đồng/ngày)"); ax.legend(loc="lower left", frameon=False, ncol=2)
    title(ax, f"Dịp mạnh nhất là {best.split(' (')[0]}: doanh thu/ngày gấp {ev.so_voi_nen.max():.2f} lần ngày thường")
    save_fig(fig, "A1_doanh_thu_theo_ngay")

    # A2: theo tháng và theo thứ
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    cols = [C2 if m == peak_m else C1 for m in monthly.index]
    ax[0].bar(monthly.index, monthly / 1e6, color=cols)
    ax[0].set_xticks(range(1, 13)); ax[0].set_xlabel("Tháng"); ax[0].set_ylabel("Doanh thu (triệu đồng)")
    title(ax[0], f"Tháng {peak_m} là đỉnh ({peak_share:.0f}% doanh thu năm), quý 4 chiếm {q4:.0f}%")
    ax[1].bar(DOW, dow_share, color=[C2 if i >= 5 else C1 for i in range(7)])
    ax[1].axhline(100 / 7, color="k", ls="--", lw=1)
    ax[1].set_ylabel("% doanh thu cửa hàng bán lẻ")
    title(ax[1], f"Cuối tuần (T7+CN) chiếm {weekend:.0f}% doanh thu cửa hàng (đường đứt = mức đều)")
    save_fig(fig, "A2_theo_thang_va_thu")

    # A3: lift theo dịp
    fig, ax = plt.subplots(figsize=(8, 4))
    ev_s = ev.so_voi_nen.sort_values()
    ax.barh(ev_s.index, ev_s.values, color=[C2 if v >= 1 else C4 for v in ev_s.values])
    ax.axvline(1, color="k", ls="--", lw=1)
    ax.bar_label(ax.containers[0], fmt="%.2f×", padding=3)
    ax.set_xlabel("Doanh thu/ngày so với ngày thường (lần)")
    title(ax, "Dịp nào kéo được doanh thu, dịp nào làm doanh thu tụt?")
    save_fig(fig, "A3_hieu_ung_dip_le")

    # A4: bản đồ nhiệt thứ x giờ, cửa hàng và online
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    heat(ax[0], retail, f"Cửa hàng bán lẻ: cao điểm quanh {int(hour.idxmax())}h, ({top3_share:.0f}% doanh thu dồn vào 3 giờ)")
    on = inv[inv.kenh_nhom == G_ONLINE]
    on_hour = on.groupby("gio").doanh_thu.sum()
    on_peak = int(on_hour.idxmax()) if len(on_hour) else 0
    heat(ax[1], on, f"Online: cao điểm quanh {on_peak}h")
    save_fig(fig, "A4_nhiet_do_thu_gio")
    note(T, f"Online có giờ cao điểm quanh {on_peak}h; cửa hàng bán lẻ quanh {int(hour.idxmax())}h.")


# --------------------------------------------------------------------------------------
# B. Điểm bán
# --------------------------------------------------------------------------------------
def topic_B(inv: pd.DataFrame) -> None:
    print("\n=== B. ĐIỂM BÁN ===")
    T = "B"
    store = (inv.groupby("branch_code")
                .agg(ten=("location_name", "first"), kenh=("branch_channel", "first"), khu_vuc=("area", "first"),
                     mat_bang=("door_type", "first"), trang_thai=("branch_status", "first"),
                     doanh_thu=("doanh_thu", "sum"), so_hd=("invoice_id", "count"), so_ngay=("ngay", "nunique"))
                .reset_index())
    total = store.doanh_thu.sum()
    retail = store[store.kenh == "Retail"].sort_values("doanh_thu", ascending=False)
    if retail.empty:                                        # phòng trường hợp tên kênh khác
        retail = store.sort_values("doanh_thu", ascending=False)
    retail["luy_ke"] = retail.doanh_thu.cumsum() / retail.doanh_thu.sum()
    n_st = len(retail)
    n80 = int((retail.luy_ke < 0.8).sum() + 1)
    n_top10 = max(1, round(.1 * n_st))
    top10_share = 100 * retail.doanh_thu.head(n_top10).sum() / retail.doanh_thu.sum()
    retail["dt_ngay"] = retail.doanh_thu / retail.so_ngay.clip(lower=1)
    save_table(store.sort_values("doanh_thu", ascending=False), "B_doanh_thu_diem_ban", index=False)

    by_area = store.groupby("khu_vuc").doanh_thu.sum().sort_values(ascending=False)
    by_door = store.groupby("mat_bang").doanh_thu.sum().sort_values(ascending=False)
    closed = store[store.trang_thai.str.contains("closed", case=False, na=False)]
    closed_share = 100 * closed.doanh_thu.sum() / total
    unk = store[store.ten == store.branch_code.astype(str)]
    area_top = by_area.index[0]; door_top = by_door.index[0]

    note(T, f"{n80}/{n_st} cửa hàng bán lẻ ({100*n80/n_st:.0f}%) tạo 80% doanh thu cửa hàng; "
            f"10% cửa hàng hàng đầu ({n_top10} cửa hàng) chiếm {top10_share:.1f}%.")
    note(T, f"Khu vực '{area_top}' chiếm {100*by_area.iloc[0]/total:.1f}% doanh thu; mặt bằng '{door_top}' chiếm "
            f"{100*by_door.iloc[0]/total:.1f}%.")
    note(T, f"Cửa hàng đã đóng ('closed') vẫn đóng góp {closed.doanh_thu.sum()/1e9:,.1f} tỷ ({closed_share:.1f}% doanh thu năm) "
            f"-> cần tách khi lập kế hoạch 2026 để không ước lượng thừa.")
    if len(unk):
        note(T, f"{len(unk)} mã điểm bán không có trong bảng branch, doanh thu {unk.doanh_thu.sum()/1e6:,.0f} triệu "
                f"({100*unk.doanh_thu.sum()/total:.1f}%).")

    # B1: Pareto cửa hàng + cột theo vùng x mặt bằng
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    x = np.arange(1, n_st + 1) / n_st * 100
    ax[0].plot(np.r_[0, x], np.r_[0, retail.luy_ke.values * 100], color=C1, lw=2.2)
    ax[0].plot([0, 100], [0, 100], color=C4, ls="--", lw=1)
    ax[0].axhline(80, color=C2, ls=":", lw=1); ax[0].axvline(100 * n80 / n_st, color=C2, ls=":", lw=1)
    ax[0].set_xlabel("% cửa hàng (xếp từ doanh thu cao đến thấp)"); ax[0].set_ylabel("% doanh thu lũy kế")
    title(ax[0], f"{100*n80/n_st:.0f}% cửa hàng ({n80}/{n_st}) tạo ra 80% doanh thu cửa hàng")
    piv = store.pivot_table(index="khu_vuc", columns="mat_bang", values="doanh_thu", aggfunc="sum").fillna(0) / 1e9
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    piv.plot.bar(stacked=True, ax=ax[1], width=.7, rot=0, colormap="tab20c")
    ax[1].set_xlabel("Khu vực"); ax[1].set_ylabel("Doanh thu (tỷ đồng)"); ax[1].legend(title="Mặt bằng", frameon=False)
    title(ax[1], f"Khu vực '{area_top}' dẫn đầu với {100*by_area.iloc[0]/total:.0f}% doanh thu")
    save_fig(fig, "B1_tap_trung_va_khu_vuc")

    # B2: top 15 cửa hàng + hiệu suất theo mặt bằng
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    top = retail.head(15).iloc[::-1]
    ax[0].barh(top.ten.str.slice(0, 34), top.doanh_thu / 1e9, color=C1)
    ax[0].set_xlabel("Doanh thu (tỷ đồng)")
    title(ax[0], f"Top 15 cửa hàng: cửa hàng số 1 đạt {retail.doanh_thu.iloc[0]/1e9:,.1f} tỷ, "
                 f"gấp {retail.doanh_thu.iloc[0]/retail.doanh_thu.median():.1f} lần cửa hàng trung vị")
    eff = retail.groupby("mat_bang").agg(n=("branch_code", "count"), dt_ngay=("dt_ngay", "median"))
    eff = eff[eff.n >= 3].sort_values("dt_ngay", ascending=False)
    ax[1].bar(eff.index, eff.dt_ngay / 1e6, color=C1)
    ax[1].bar_label(ax[1].containers[0], labels=[f"n={int(v)}" for v in eff.n], padding=2, fontsize=8)
    ax[1].set_ylabel("Trung vị doanh thu/ngày bán (triệu đồng)")
    if len(eff) >= 2:
        title(ax[1], f"Mặt bằng '{eff.index[0]}' hiệu quả nhất: {eff.dt_ngay.iloc[0]/1e6:,.1f} triệu/ngày/cửa hàng")
        note(T, f"Hiệu suất trung vị: '{eff.index[0]}' {eff.dt_ngay.iloc[0]/1e6:,.1f} triệu/ngày vs "
                f"'{eff.index[-1]}' {eff.dt_ngay.iloc[-1]/1e6:,.1f} triệu/ngày.")
    else:
        title(ax[1], "Hiệu suất doanh thu/ngày theo mặt bằng")
    save_fig(fig, "B2_top_cua_hang_va_hieu_suat")


# --------------------------------------------------------------------------------------
# C. Sản phẩm
# --------------------------------------------------------------------------------------
def clean_items(items: pd.DataFrame) -> pd.DataFrame:
    it = items.copy()
    for c in ["group_cat", "gender", "season", "licenses", "product_type", "product_name"]:
        it[c] = it[c].fillna("Không rõ").astype(str).str.strip().replace("", "Không rõ")
    it["ban_quyen"] = ~it.licenses.str.lower().isin(NO_LICENSE)
    return it


def topic_C(items: pd.DataFrame) -> None:
    print("\n=== C. SẢN PHẨM ===")
    T = "C"
    it = clean_items(items)
    tot = it.ban_dt.sum()
    cat = it.groupby("group_cat").agg(doanh_thu=("ban_dt", "sum"), so_luong=("ban_sl", "sum"),
                                      so_item=("item", "nunique")).sort_values("doanh_thu", ascending=False)
    cat["ty_trong"] = 100 * cat.doanh_thu / tot
    gender = it.groupby("gender").ban_dt.sum().sort_values(ascending=False)
    season = it.groupby("season").ban_dt.sum().sort_values(ascending=False)
    save_table(cat, "C_nganh_hang")

    lic = it.groupby("ban_quyen").agg(doanh_thu=("ban_dt", "sum"), so_item=("item", "nunique"))
    lic["dt_item"] = lic.doanh_thu / lic.so_item
    lic_share = 100 * lic.doanh_thu.get(True, 0) / tot
    top_lic = (it[it.ban_quyen].groupby("licenses").ban_dt.sum().sort_values(ascending=False).head(10))

    note(T, f"Ngành hàng '{cat.index[0]}' chiếm {cat.ty_trong.iloc[0]:.1f}% doanh thu, '{cat.index[1]}' {cat.ty_trong.iloc[1]:.1f}%; "
            f"top 3 ngành chiếm {cat.ty_trong.head(3).sum():.1f}%.")
    note(T, f"Theo giới tính: '{gender.index[0]}' dẫn đầu ({100*gender.iloc[0]/tot:.1f}%). "
            f"Theo mùa: '{season.index[0]}' chiếm {100*season.iloc[0]/tot:.1f}%.")
    if True in lic.index and False in lic.index:
        note(T, f"Sản phẩm bản quyền nhân vật: {lic_share:.1f}% doanh thu từ {int(lic.so_item[True]):,} item "
                f"(TB {lic.dt_item[True]/1e6:,.1f} triệu/item) vs thương hiệu riêng {int(lic.so_item[False]):,} item "
                f"(TB {lic.dt_item[False]/1e6:,.1f} triệu/item).")

    # ABC theo doanh thu item
    abc = it[it.ban_dt > 0].sort_values("ban_dt", ascending=False).reset_index(drop=True)
    abc["luy_ke"] = abc.ban_dt.cumsum() / abc.ban_dt.sum()
    abc["lop"] = np.where(abc.luy_ke <= .8, "A", np.where(abc.luy_ke <= .95, "B", "C"))
    # item vượt ngưỡng 80% đầu tiên vẫn thuộc A
    first_over = abc.index[abc.luy_ke > .8]
    if len(first_over):
        abc.loc[first_over[0], "lop"] = "A"
    grp = abc.groupby("lop").agg(so_item=("item", "count"), doanh_thu=("ban_dt", "sum")).reindex(["A", "B", "C"]).fillna(0)
    grp["pct_item"] = 100 * grp.so_item / grp.so_item.sum()
    grp["pct_dt"] = 100 * grp.doanh_thu / grp.doanh_thu.sum()
    save_table(grp, "C_phan_tich_ABC")
    save_table(abc[["item", "product_name", "group_cat", "ban_sl", "ban_dt", "luy_ke", "lop"]].head(50), "C_top50_item", index=False)
    note(T, f"ABC: {int(grp.so_item['A']):,} item hạng A ({grp.pct_item['A']:.1f}% số item) tạo 80% doanh thu; "
            f"{int(grp.so_item['C']):,} item hạng C ({grp.pct_item['C']:.1f}%) chỉ tạo {grp.pct_dt['C']:.1f}%.")

    # C1: ngành hàng + giới tính/mùa
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    c10 = cat.head(10).iloc[::-1]
    ax[0].barh(c10.index, c10.ty_trong, color=C1)
    ax[0].bar_label(ax[0].containers[0], fmt="%.1f%%", padding=2, fontsize=8)
    ax[0].set_xlabel("% doanh thu bán")
    title(ax[0], f"'{cat.index[0]}' dẫn đầu với {cat.ty_trong.iloc[0]:.0f}% doanh thu")
    ax[1].bar(gender.index, 100 * gender / tot, color=C1)
    ax[1].bar_label(ax[1].containers[0], fmt="%.1f%%", padding=2, fontsize=8)
    ax[1].tick_params(axis="x", rotation=20); ax[1].set_ylabel("% doanh thu bán")
    title(ax[1], f"Giới tính: '{gender.index[0]}' chiếm {100*gender.iloc[0]/tot:.0f}%")
    ax[2].bar(season.index, 100 * season / tot, color=C3)
    ax[2].bar_label(ax[2].containers[0], fmt="%.1f%%", padding=2, fontsize=8)
    ax[2].tick_params(axis="x", rotation=20)
    title(ax[2], f"Mùa: '{season.index[0]}' chiếm {100*season.iloc[0]/tot:.0f}%")
    save_fig(fig, "C1_co_cau_nganh_gioi_mua")

    # C2: bản quyền vs thương hiệu riêng
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
    lab = {True: "Bản quyền\nnhân vật", False: "Thương hiệu riêng /\nkhông ghi bản quyền"}
    ax[0].bar([lab[i] for i in lic.index], 100 * lic.doanh_thu / tot, color=[C2 if i else C1 for i in lic.index])
    ax[0].bar_label(ax[0].containers[0], fmt="%.1f%%", padding=2)
    ax[0].set_ylabel("% doanh thu bán")
    title(ax[0], f"Sản phẩm bản quyền đóng góp {lic_share:.0f}% doanh thu")
    if len(top_lic):
        tl = top_lic.iloc[::-1]
        ax[1].barh(tl.index, tl / 1e6, color=C2)
        ax[1].set_xlabel("Doanh thu (triệu đồng)")
        title(ax[1], f"Bản quyền nào bán tốt nhất: '{top_lic.index[0]}' ({top_lic.iloc[0]/1e6:,.0f} triệu)")
    save_fig(fig, "C2_ban_quyen_vs_thuong_hieu_rieng")

    # C3: ABC
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    xs = np.arange(1, len(abc) + 1) / len(abc) * 100
    ax[0].plot(xs, abc.luy_ke * 100, color=C1, lw=2.2)
    ax[0].axhline(80, color=C2, ls=":"); ax[0].axhline(95, color=C2, ls=":")
    ax[0].set_xlabel("% item (xếp theo doanh thu giảm dần)"); ax[0].set_ylabel("% doanh thu lũy kế")
    title(ax[0], f"{grp.pct_item['A']:.0f}% item hạng A tạo 80% doanh thu")
    xl = [f"Hạng {k}" for k in grp.index]
    w = .38; xi = np.arange(3)
    ax[1].bar(xi - w / 2, grp.pct_item, w, label="% số item", color=C4)
    ax[1].bar(xi + w / 2, grp.pct_dt, w, label="% doanh thu", color=C1)
    ax[1].set_xticks(xi); ax[1].set_xticklabels(xl); ax[1].set_ylabel("%"); ax[1].legend(frameon=False)
    title(ax[1], f"Hạng C: {grp.pct_item['C']:.0f}% số item nhưng chỉ {grp.pct_dt['C']:.0f}% doanh thu")
    save_fig(fig, "C3_phan_tich_ABC")


# --------------------------------------------------------------------------------------
# D. Giá và khuyến mãi
# --------------------------------------------------------------------------------------
def topic_D(ban: pd.DataFrame) -> None:
    print("\n=== D. GIÁ VÀ KHUYẾN MÃI ===")
    T = "D"
    b = ban[ban.tien_hang > 0].copy()
    b["co_giam"] = b.giam_gia > 0
    b["ty_le_giam"] = (b.giam_gia / b.tien_hang).clip(0, 1)
    pct_disc = 100 * b.co_giam.mean()
    avg_rate = 100 * b.loc[b.co_giam, "ty_le_giam"].mean()
    w_rate = 100 * b.giam_gia.sum() / b.tien_hang.sum()
    note(T, f"{pct_disc:.1f}% đơn bán có giảm giá; trên đơn có giảm, mức giảm trung bình {avg_rate:.1f}% tiền hàng; "
            f"tính cả năm, giảm giá bằng {w_rate:.1f}% tổng tiền hàng.")

    groups = [g for g in [G_RETAIL, G_ONLINE] if (b.kenh_nhom == g).any()]
    m_pct = b.groupby(["thang", "kenh_nhom"]).co_giam.mean().unstack() * 100
    m_rate = b[b.co_giam].groupby(["thang", "kenh_nhom"]).ty_le_giam.mean().unstack() * 100
    save_table(m_pct, "D_ty_le_don_giam_gia_theo_thang"); save_table(m_rate, "D_muc_giam_theo_thang")

    for g in groups:
        s = m_pct[g].dropna()
        if len(s):
            note(T, f"{g}: tháng giảm giá nhiều nhất là tháng {int(s.idxmax())} ({s.max():.0f}% đơn), "
                    f"ít nhất tháng {int(s.idxmin())} ({s.min():.0f}%).")

    # AOV có/không giảm giá
    aov = b.groupby(["kenh_nhom", "co_giam"]).doanh_thu.agg(["median", "mean", "count"]).reset_index()
    save_table(aov, "D_aov_co_khong_giam_gia", index=False)
    r = b[b.kenh_nhom == G_RETAIL]
    if r.co_giam.nunique() == 2:
        med_y, med_n = r.loc[r.co_giam, "doanh_thu"].median(), r.loc[~r.co_giam, "doanh_thu"].median()
        note(T, f"Cửa hàng bán lẻ: AOV trung vị đơn có giảm giá {med_y:,.0f}đ vs không giảm {med_n:,.0f}đ "
                f"({100*(med_y/med_n-1):+.0f}%).")
        mann_whitney(r.loc[r.co_giam, "doanh_thu"], r.loc[~r.co_giam, "doanh_thu"], T,
                     "AOV đơn có giảm giá có khác đơn không giảm giá (cửa hàng bán lẻ)?",
                     "Phân phối giá trị đơn của hai nhóm như nhau", "Phân phối giá trị đơn của hai nhóm khác nhau",
                     "Có giảm giá", "Không giảm giá")

    # D1: theo tháng
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for g, col in zip(groups, [C1, C2]):
        ax[0].plot(m_pct.index, m_pct[g], marker="o", color=col, label=g)
        ax[1].plot(m_rate.index, m_rate[g], marker="o", color=col, label=g)
    for a, yl in zip(ax, ["% đơn có giảm giá", "Mức giảm TB trên đơn có giảm (% tiền hàng)"]):
        a.set_xticks(range(1, 13)); a.set_xlabel("Tháng"); a.set_ylabel(yl); a.legend(frameon=False)
    title(ax[0], f"Khuyến mãi phủ {pct_disc:.0f}% đơn bán; thay đổi rõ theo tháng và theo kênh")
    title(ax[1], f"Khi đã giảm, mức giảm trung bình khoảng {avg_rate:.0f}% tiền hàng")
    save_fig(fig, "D1_khuyen_mai_theo_thang_kenh")

    # D2: AOV + phân phối mức giảm
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    piv = b.pivot_table(index="kenh_nhom", columns="co_giam", values="doanh_thu", aggfunc="median").reindex(groups) / 1e3
    piv.columns = ["Không giảm giá" if not c else "Có giảm giá" for c in piv.columns]
    piv.plot.bar(ax=ax[0], rot=0, color=[C4, C1][:piv.shape[1]] if piv.shape[1] == 2 else C1)
    ax[0].set_xlabel(""); ax[0].set_ylabel("AOV trung vị (nghìn đồng)"); ax[0].legend(frameon=False)
    title(ax[0], (f"Cửa hàng bán lẻ: AOV trung vị đơn có giảm giá {med_y/1e3:,.0f} nghìn vs {med_n/1e3:,.0f} nghìn đơn thường" if r.co_giam.nunique() == 2 else "AOV trung vị theo kênh và có/không giảm giá"))
    ax[1].hist(100 * b.loc[b.co_giam, "ty_le_giam"], bins=np.arange(0, 101, 2.5), color=C1)
    ax[1].set_xlabel("Mức giảm (% tiền hàng)"); ax[1].set_ylabel("Số hóa đơn")
    title(ax[1], f"Mức giảm tập trung ở vài mốc cố định (TB {avg_rate:.0f}%)")
    save_fig(fig, "D2_aov_va_muc_giam")


# --------------------------------------------------------------------------------------
# E. Khách hàng thành viên
# --------------------------------------------------------------------------------------
def topic_E(ban: pd.DataFrame) -> None:
    print("\n=== E. KHÁCH HÀNG THÀNH VIÊN ===")
    T = "E"
    m = ban[ban.co_khach]
    share = 100 * m.doanh_thu.sum() / ban.doanh_thu.sum()
    share_m = 100 * m.groupby("thang").doanh_thu.sum() / ban.groupby("thang").doanh_thu.sum()
    share_ch = (100 * m.groupby("kenh_nhom").doanh_thu.sum() / ban.groupby("kenh_nhom").doanh_thu.sum()).dropna()
    note(T, f"Khách có mã thành viên tạo {share:.1f}% doanh thu bán ({100*len(m)/len(ban):.1f}% số đơn); "
            f"cao nhất tháng {int(share_m.idxmax())} ({share_m.max():.0f}%), thấp nhất tháng {int(share_m.idxmin())} ({share_m.min():.0f}%).")
    if len(share_ch):
        note(T, "Tỷ lệ doanh thu thành viên theo kênh: " + "; ".join(f"{k} {v:.0f}%" for k, v in share_ch.items()) + ".")

    # tần suất và khoảng cách giữa các lần mua (tính theo ngày mua)
    days = m.groupby("customer_id").ngay.nunique()
    n_cus = len(days)
    pct_repeat = 100 * (days >= 2).mean()
    pairs = m[["customer_id", "ngay"]].drop_duplicates().sort_values(["customer_id", "ngay"])
    gaps = pairs.groupby("customer_id").ngay.diff().dt.days.dropna()
    note(T, f"{n_cus:,} khách thành viên mua trong năm; {pct_repeat:.1f}% mua từ 2 ngày trở lên "
            f"(trung bình {days.mean():.2f} ngày mua/khách). Khoảng cách giữa hai lần mua: trung vị {gaps.median():.0f} ngày, "
            f"{100*(gaps <= 30).mean():.0f}% trong vòng 30 ngày.")
    if pct_repeat < 15:
        note(T, "Tỷ lệ mua lặp rất thấp -> cá nhân hóa theo lịch sử mua khó; cân nhắc gợi ý theo nội dung/giỏ hàng (nhánh AI).")

    # theo hạng thẻ (hạng ghi trên hóa đơn tại thời điểm mua)
    lv = m.dropna(subset=["member_level"])
    order = [x for x in LEVELS if x in set(lv.member_level)] + sorted(set(lv.member_level) - set(LEVELS))
    tl = (lv.groupby("member_level")
            .agg(so_khach=("customer_id", "nunique"), so_hd=("invoice_id", "count"),
                 doanh_thu=("doanh_thu", "sum"), aov_trung_vi=("doanh_thu", "median"), aov_tb=("doanh_thu", "mean"))
            .reindex(order))
    tl["hd_moi_khach"] = tl.so_hd / tl.so_khach
    tl["ty_trong_dt"] = 100 * tl.doanh_thu / tl.doanh_thu.sum()
    save_table(tl, "E_theo_hang_the")
    if len(tl):
        top = tl.ty_trong_dt.idxmax()
        note(T, f"Theo hạng thẻ: '{top}' đóng góp nhiều doanh thu thành viên nhất ({tl.ty_trong_dt.max():.1f}%); "
                f"AOV TB từ {tl.aov_tb.min():,.0f}đ ({tl.aov_tb.idxmin()}) đến {tl.aov_tb.max():,.0f}đ ({tl.aov_tb.idxmax()}).")

    # kiểm định: AOV thành viên vs vãng lai (cửa hàng bán lẻ)
    r = ban[ban.kenh_nhom == G_RETAIL]
    a, bb = r.loc[r.co_khach, "doanh_thu"], r.loc[~r.co_khach, "doanh_thu"]
    mann_whitney(a, bb, T, "AOV của khách thành viên có khác khách vãng lai (cửa hàng bán lẻ)?",
                 "Phân phối giá trị đơn của khách thành viên và vãng lai như nhau",
                 "Phân phối giá trị đơn của hai nhóm khác nhau", "Thành viên", "Vãng lai")
    note(T, f"AOV trung vị cửa hàng bán lẻ: thành viên {a.median():,.0f}đ vs vãng lai {bb.median():,.0f}đ "
            f"({100*(a.median()/bb.median()-1):+.0f}%).")

    # cohort giữ chân theo tháng mua đầu tiên (lưu ý: năm 2025 là dữ liệu bị cắt trái)
    first = m.groupby("customer_id").ngay.min().dt.month.rename("cohort")
    act = m[["customer_id", "thang"]].drop_duplicates().merge(first, on="customer_id")
    act["k"] = act.thang - act.cohort
    mat = act.groupby(["cohort", "k"]).customer_id.nunique().unstack(fill_value=0).reindex(columns=range(12), fill_value=0)
    ret = mat.div(mat[0], axis=0) * 100
    for c in ret.index:                           # ô chưa quan sát được -> NaN
        ret.loc[c, [k for k in ret.columns if c + k > 12]] = np.nan
    save_table(ret.round(1), "E_cohort_giu_chan")
    cohorts = [c for c in mat.index if c <= 11]
    w1 = 100 * mat.loc[cohorts, 1].sum() / mat.loc[cohorts, 0].sum() if cohorts else np.nan
    note(T, f"Cohort giữ chân: trung bình {w1:.1f}% khách quay lại ở tháng kế tiếp sau lần mua đầu "
            f"(cohort tháng 1 gồm cả khách cũ nên bị cao ảo).")

    # E1: tỷ trọng theo tháng + hạng thẻ
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(share_m.index, share_m.values, marker="o", color=C1)
    ax[0].set_ylim(0, max(100, share_m.max() + 5)); ax[0].set_xticks(range(1, 13))
    ax[0].set_xlabel("Tháng"); ax[0].set_ylabel("% doanh thu từ khách thành viên")
    title(ax[0], f"Khách thành viên tạo {share:.0f}% doanh thu bán")
    if len(tl):
        ax[1].bar(tl.index, tl.ty_trong_dt, color=C1); ax[1].tick_params(axis="x", rotation=25)
        ax[1].set_ylabel("% doanh thu thành viên")
        title(ax[1], f"Hạng '{tl.ty_trong_dt.idxmax()}' đóng góp nhiều nhất ({tl.ty_trong_dt.max():.0f}%)")
        ax[2].bar(tl.index, tl.aov_tb / 1e3, color=C2); ax[2].tick_params(axis="x", rotation=25)
        ax[2].set_ylabel("AOV trung bình (nghìn đồng)")
        title(ax[2], f"AOV TB từ {tl.aov_tb.min()/1e3:,.0f} nghìn ({tl.aov_tb.idxmin()}) đến {tl.aov_tb.max()/1e3:,.0f} nghìn ({tl.aov_tb.idxmax()})")
    save_fig(fig, "E1_ty_trong_va_hang_the")

    # E2: tần suất + khoảng cách
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    days.clip(upper=10).value_counts().sort_index().plot.bar(ax=ax[0], color=C1)
    ax[0].set_xlabel("Số ngày mua trong năm (10 = 10 trở lên)"); ax[0].set_ylabel("Số khách")
    title(ax[0], f"{100-pct_repeat:.0f}% khách thành viên chỉ mua 1 ngày trong năm")
    ax[1].hist(gaps.clip(upper=180), bins=36, color=C1)
    ax[1].axvline(gaps.median(), color=C2, ls="--")
    ax[1].set_xlabel("Số ngày giữa hai lần mua liên tiếp (180 = 180 trở lên)"); ax[1].set_ylabel("Số lần mua lại")
    title(ax[1], f"Khách mua lại thường sau {gaps.median():.0f} ngày (trung vị)")
    save_fig(fig, "E2_tan_suat_va_khoang_cach_mua")

    # E3: cohort
    fig, ax = plt.subplots(figsize=(9, 5.2))
    show = ret.values.copy(); show[:, 0] = np.nan          # cột tháng 0 luôn = 100%, ẩn để thang màu rõ hơn
    im = ax.imshow(np.ma.masked_invalid(show), cmap="Blues", vmin=0, vmax=40, aspect="auto")
    ax.set_xticks(range(12)); ax.set_yticks(range(len(ret))); ax.set_yticklabels([f"T{c}" for c in ret.index])
    ax.set_xlabel("Số tháng sau lần mua đầu tiên"); ax.set_ylabel("Cohort (tháng mua đầu tiên)")
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            v = ret.values[i, j]
            if j > 0 and not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="k" if v < 25 else "w")
    ax.grid(False); plt.colorbar(im, ax=ax, label="% khách quay lại")
    title(ax, f"Chỉ khoảng {w1:.0f}% khách quay lại ở tháng kế tiếp (thang màu cắt ở 40%)")
    save_fig(fig, "E3_cohort_giu_chan")


# --------------------------------------------------------------------------------------
# F. Trả hàng
# --------------------------------------------------------------------------------------
def topic_F(inv: pd.DataFrame, items: pd.DataFrame, sizes: pd.DataFrame) -> None:
    print("\n=== F. TRẢ HÀNG ===")
    T = "F"
    it = clean_items(items)
    MIN_UNITS = 1000
    cat = it.groupby("group_cat").agg(ban_sl=("ban_sl", "sum"), tra_sl=("tra_sl", "sum"))
    cat["ty_le_tra"] = 100 * cat.tra_sl / cat.ban_sl.replace(0, np.nan)
    catf = cat[cat.ban_sl >= MIN_UNITS].sort_values("ty_le_tra", ascending=False)
    save_table(cat.sort_values("ban_sl", ascending=False), "F_ty_le_tra_nganh_hang")

    sz = sizes.copy()
    sz["size_name"] = sz.size_name.fillna("Không rõ")
    sz = sz.groupby("size_name").agg(ban_sl=("ban_sl", "sum"), tra_sl=("tra_sl", "sum"))
    sz["ty_le_tra"] = 100 * sz.tra_sl / sz.ban_sl.replace(0, np.nan)
    szf = sz[sz.ban_sl >= 500].sort_values("ty_le_tra", ascending=False)
    save_table(sz.sort_values("ban_sl", ascending=False), "F_ty_le_tra_size")

    tot_b, tot_r = it.ban_sl.sum(), it.tra_sl.sum()
    note(T, f"Tổng cộng bán {tot_b:,.0f} sản phẩm, trả {tot_r:,.0f} ({100*tot_r/tot_b:.2f}% số lượng).")
    if len(catf):
        note(T, f"Ngành hàng trả nhiều nhất: '{catf.index[0]}' ({catf.ty_le_tra.iloc[0]:.2f}%); ít nhất: "
                f"'{catf.index[-1]}' ({catf.ty_le_tra.iloc[-1]:.2f}%) (chỉ xét ngành bán ≥ {MIN_UNITS:,} sản phẩm).")
    if len(szf):
        note(T, f"Size hay bị trả nhất: '{szf.index[0]}' ({szf.ty_le_tra.iloc[0]:.2f}%), kế đến '{szf.index[1] if len(szf) > 1 else '-'}'; "
                f"thấp nhất '{szf.index[-1]}' ({szf.ty_le_tra.iloc[-1]:.2f}%).")

    # chi-square: tỷ lệ trả có phụ thuộc ngành hàng?
    top = cat[cat.ban_sl >= MIN_UNITS].sort_values("ban_sl", ascending=False).head(8)
    if len(top) >= 2:
        table = np.c_[top.tra_sl.values, (top.ban_sl - top.tra_sl).clip(lower=0).values]
        chi2, p, dof, _ = stats.chi2_contingency(table)
        v = np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))
        TESTS.append({
            "chu_de": T, "cau_hoi": f"Tỷ lệ trả hàng (theo số lượng) có phụ thuộc ngành hàng? ({len(top)} ngành bán nhiều nhất)",
            "H0": "Tỷ lệ trả hàng như nhau giữa các ngành hàng", "H1": "Có ít nhất một ngành có tỷ lệ trả khác",
            "kiem_dinh": "Chi-square độc lập",
            "so_lieu": "; ".join(f"{i}: {r:.2f}%" for i, r in top.ty_le_tra.items()),
            "thong_ke": f"chi2={chi2:,.1f}; dof={dof}; p {fmt_p(p)}; Cramér's V={v:.3f}",
            "ket_luan": ("bác bỏ H0" if p < .05 else "chưa đủ bằng chứng bác bỏ H0") +
                        f" (mức 5%). Cỡ ảnh hưởng V={v:.3f} ({'nhỏ' if v < .1 else 'vừa' if v < .3 else 'lớn'}). "
                        "Lưu ý: đơn vị là sản phẩm chứ không độc lập tuyệt đối, p-value có xu hướng quá nhỏ."})
        print(f"  => Kiểm định [F] chi-square: p {fmt_p(p)}, V={v:.3f}")

    # trả hàng theo cửa hàng và theo tháng (theo giá trị, từ hóa đơn)
    pos = inv[inv.doanh_thu > 0].groupby("branch_code").doanh_thu.sum()
    neg = -inv[inv.doanh_thu < 0].groupby("branch_code").doanh_thu.sum()
    n_hd = inv.groupby("branch_code").invoice_id.count()
    nm = inv.groupby("branch_code").location_name.first()
    kn = inv.groupby("branch_code").branch_channel.first()
    st = pd.DataFrame({"ten": nm, "kenh": kn, "so_hd": n_hd, "ban": pos, "tra": neg}).fillna({"tra": 0, "ban": 0})
    st["ty_le_tra"] = 100 * st.tra / st.ban.replace(0, np.nan)
    stf = st[(st.so_hd >= 300) & (st.kenh == "Retail")].sort_values("ty_le_tra", ascending=False)
    save_table(st.sort_values("ty_le_tra", ascending=False), "F_ty_le_tra_cua_hang")
    mon_pos = inv[inv.doanh_thu > 0].groupby("thang").doanh_thu.sum()
    mon_neg = -inv[inv.doanh_thu < 0].groupby("thang").doanh_thu.sum()
    mon = (100 * mon_neg / mon_pos).reindex(range(1, 13))
    if len(stf):
        note(T, f"Theo giá trị: cửa hàng trả nhiều nhất '{stf.ten.iloc[0]}' ({stf.ty_le_tra.iloc[0]:.1f}% doanh thu bán), "
                f"trung vị các cửa hàng {stf.ty_le_tra.median():.1f}%.")
    note(T, f"Tỷ lệ trả theo giá trị cao nhất tháng {int(mon.idxmax())} ({mon.max():.1f}%), thấp nhất tháng {int(mon.idxmin())} ({mon.min():.1f}%).")

    # F1: ngành hàng + size
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    if len(catf):
        c = catf.head(12).iloc[::-1]
        ax[0].barh(c.index, c.ty_le_tra, color=C1)
        ax[0].bar_label(ax[0].containers[0], fmt="%.2f%%", padding=2, fontsize=8)
        ax[0].set_xlabel("Tỷ lệ trả (% số lượng bán)")
        title(ax[0], f"Ngành '{catf.index[0]}' có tỷ lệ trả cao nhất ({catf.ty_le_tra.iloc[0]:.1f}%)")
    if len(szf):
        s = szf.head(15).iloc[::-1]
        ax[1].barh(s.index, s.ty_le_tra, color=C2)
        ax[1].bar_label(ax[1].containers[0], fmt="%.2f%%", padding=2, fontsize=8)
        ax[1].set_xlabel("Tỷ lệ trả (% số lượng bán)")
        title(ax[1], f"Size '{szf.index[0]}' hay bị trả/đổi nhất ({szf.ty_le_tra.iloc[0]:.1f}%)")
    save_fig(fig, "F1_tra_hang_nganh_va_size")

    # F2: cửa hàng + theo tháng
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    if len(stf):
        t = stf.head(12).iloc[::-1]
        ax[0].barh(t.ten.str.slice(0, 34), t.ty_le_tra, color=C1)
        ax[0].axvline(stf.ty_le_tra.median(), color=C2, ls="--", lw=1)
        ax[0].set_xlabel("Giá trị trả (% doanh thu bán)")
        title(ax[0], f"12 cửa hàng trả nhiều nhất, cao hơn trung vị {stf.ty_le_tra.median():.1f}% (đường đứt)")
    ax[1].plot(mon.index, mon.values, marker="o", color=C1)
    ax[1].set_xticks(range(1, 13)); ax[1].set_xlabel("Tháng"); ax[1].set_ylabel("Giá trị trả (% doanh thu bán)")
    title(ax[1], f"Trả hàng đỉnh vào tháng {int(mon.idxmax())} ({mon.max():.1f}%)")
    save_fig(fig, "F2_tra_hang_cua_hang_va_thang")


# --------------------------------------------------------------------------------------
# G. Online và offline
# --------------------------------------------------------------------------------------
def topic_G(inv: pd.DataFrame, ban: pd.DataFrame) -> None:
    print("\n=== G. ONLINE VÀ OFFLINE ===")
    T = "G"
    tot = ban.doanh_thu.sum()
    ch = (ban.groupby("sale_channel")
             .agg(so_hd=("invoice_id", "count"), doanh_thu=("doanh_thu", "sum"), aov=("doanh_thu", "median"))
             .sort_values("doanh_thu", ascending=False))
    ch["ty_trong"] = 100 * ch.doanh_thu / tot
    save_table(ch, "G_theo_kenh_ban")

    def stat(g):
        x = g[g.tien_hang > 0]
        return pd.Series({
            "so_hd": len(g), "doanh_thu": g.doanh_thu.sum(), "aov_trung_vi": g.doanh_thu.median(), "aov_tb": g.doanh_thu.mean(),
            "upt": g.loc[g.so_luong_ban > 0, "so_luong_ban"].mean(),
            "giam_gia_pct": 100 * x.giam_gia.sum() / x.tien_hang.sum() if len(x) else np.nan,
            "cod_pct": 100 * g.is_cod.mean(),
        })
    grp = ban.groupby("kenh_nhom").apply(stat)
    pos = inv[inv.doanh_thu > 0].groupby("kenh_nhom").doanh_thu.sum()
    neg = -inv[inv.doanh_thu < 0].groupby("kenh_nhom").doanh_thu.sum()
    grp["tra_pct"] = (100 * neg / pos).reindex(grp.index).fillna(0)
    grp["ty_trong"] = 100 * grp.doanh_thu / grp.doanh_thu.sum()
    save_table(grp, "G_online_vs_offline")

    if G_ONLINE in grp.index and G_RETAIL in grp.index:
        o, r = grp.loc[G_ONLINE], grp.loc[G_RETAIL]
        note(T, f"Online chiếm {o.ty_trong:.1f}% doanh thu bán ({o.so_hd:,.0f} đơn); cửa hàng bán lẻ {r.ty_trong:.1f}% ({r.so_hd:,.0f} đơn).")
        note(T, f"AOV trung vị: online {o.aov_trung_vi:,.0f}đ vs cửa hàng {r.aov_trung_vi:,.0f}đ; "
                f"UPT: {o.upt:.2f} vs {r.upt:.2f} sản phẩm/đơn.")
        note(T, f"Giảm giá bằng {o.giam_gia_pct:.1f}% tiền hàng ở online vs {r.giam_gia_pct:.1f}% ở cửa hàng; "
                f"tỷ lệ COD online {o.cod_pct:.1f}%; giá trị trả hàng {o.tra_pct:.1f}% (online) vs {r.tra_pct:.1f}% (cửa hàng).")
        mann_whitney(ban.loc[ban.kenh_nhom == G_ONLINE, "doanh_thu"], ban.loc[ban.kenh_nhom == G_RETAIL, "doanh_thu"], T,
                     "AOV đơn online có khác đơn tại cửa hàng bán lẻ?",
                     "Phân phối giá trị đơn online và cửa hàng như nhau", "Phân phối giá trị đơn hai kênh khác nhau",
                     "Online", "Cửa hàng")
    on_ch = ban[ban.kenh_nhom == G_ONLINE].groupby("sale_channel").agg(so_hd=("invoice_id", "count"),
                                                                        cod=("is_cod", lambda s: 100 * s.mean()))
    if len(on_ch):
        note(T, "Tỷ lệ COD theo sàn: " + "; ".join(f"{k} {v:.0f}%" for k, v in on_ch.cod.items()) + ".")

    # G1: kênh bán + so sánh chỉ số
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
    c = ch.iloc[::-1]
    ax[0].barh(c.index, c.doanh_thu / 1e9, color=[C2 if k.lower() in ONLINE_CHANNELS else C1 for k in c.index])
    ax[0].bar_label(ax[0].containers[0], labels=[f"{v:.1f}%" for v in c.ty_trong], padding=2, fontsize=8)
    ax[0].set_xlabel("Doanh thu bán (tỷ đồng)")
    title(ax[0], f"Cửa hàng chiếm đa số; online (cam) {grp.ty_trong.get(G_ONLINE, 0):.0f}% doanh thu")
    metrics = [("aov_trung_vi", "AOV trung vị\n(nghìn đồng)", 1e3), ("upt", "UPT\n(SP/đơn)", 1),
               ("giam_gia_pct", "Giảm giá\n(% tiền hàng)", 1), ("tra_pct", "Trả hàng\n(% doanh thu bán)", 1)]
    gs = [g for g in [G_RETAIL, G_ONLINE] if g in grp.index]
    ax[1].set_axis_off()
    for k, (col, lab, div) in enumerate(metrics):
        a = ax[1].inset_axes([k / 4 + .02, .12, .21, .72])
        vals = [grp.loc[g, col] / div for g in gs]
        a.bar(range(len(gs)), vals, color=[C1, C2][:len(gs)])
        a.bar_label(a.containers[0], fmt="%.1f", padding=2, fontsize=8)
        a.set_xticks(range(len(gs))); a.set_xticklabels(["Cửa hàng" if g == G_RETAIL else "Online" for g in gs], fontsize=8)
        a.set_title(lab, fontsize=9)
    ax[1].set_title("Online vs cửa hàng bán lẻ: quy mô đơn, giỏ hàng, giảm giá, trả hàng", loc="left")
    save_fig(fig, "G1_kenh_ban_va_chi_so")

    # G2: theo tháng + COD
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    mm = ban.pivot_table(index="thang", columns="kenh_nhom", values="doanh_thu", aggfunc="sum").reindex(range(1, 13)).fillna(0) / 1e6
    for g, col in zip([G_RETAIL, G_ONLINE], [C1, C2]):
        if g in mm:
            ax[0].plot(mm.index, mm[g], marker="o", color=col, label=g)
    ax[0].set_xticks(range(1, 13)); ax[0].set_xlabel("Tháng"); ax[0].set_ylabel("Doanh thu bán (triệu đồng)"); ax[0].legend(frameon=False)
    if G_ONLINE in mm:
        om = int(mm[G_ONLINE].idxmax())
        title(ax[0], f"Online đạt đỉnh tháng {om} ({mm[G_ONLINE].max():,.0f} triệu), lệch pha với cửa hàng")
    if len(on_ch):
        ax[1].bar(on_ch.index, on_ch.cod, color=C2)
        ax[1].bar_label(ax[1].containers[0], fmt="%.0f%%", padding=2)
        ax[1].set_ylabel("% đơn thanh toán COD"); ax[1].tick_params(axis="x", rotation=20)
        title(ax[1], "Tỷ lệ thanh toán khi nhận hàng (COD) theo sàn")
    save_fig(fig, "G2_online_theo_thang_va_cod")


# --------------------------------------------------------------------------------------
# 5. Xuất báo cáo insight
# --------------------------------------------------------------------------------------
TOPIC_NAMES = {"0": "Tổng quan dữ liệu", "A": "A. Mùa vụ và thời gian", "B": "B. Điểm bán", "C": "C. Sản phẩm",
               "D": "D. Giá và khuyến mãi", "E": "E. Khách hàng thành viên", "F": "F. Trả hàng", "G": "G. Online và offline"}


def write_report() -> None:
    lines = ["# Kết quả EDA - Tuần 5", "",
             "_File sinh tự động bởi `eda.py`. Số liệu tính từ hóa đơn hợp lệ năm 2025; doanh thu = `actual_amount`; "
             "doanh thu thuần = bán trừ trả._", ""]
    for key in TOPIC_NAMES:
        items = [t for k, t in INSIGHTS if k == key]
        if items:
            lines += [f"## {TOPIC_NAMES[key]}", ""] + [f"- {t}" for t in items] + [""]
    if TESTS:
        lines += ["## Kiểm định thống kê", ""]
        for i, t in enumerate(TESTS, 1):
            lines += [f"### {i}. [{t['chu_de']}] {t['cau_hoi']}", "",
                      f"- **H0:** {t['H0']}", f"- **H1:** {t['H1']}", f"- **Kiểm định:** {t['kiem_dinh']}",
                      f"- **Số liệu:** {t['so_lieu']}", f"- **Thống kê:** {t['thong_ke']}",
                      f"- **Kết luận:** {t['ket_luan']}", ""]
    if FIGURES:
        lines += ["## Danh sách biểu đồ", ""] + [f"- `reports/figures/{f}`" for f in FIGURES] + [""]
    (REPORTS / "eda_insights.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nĐã ghi {REPORTS / 'eda_insights.md'} ({len(INSIGHTS)} insight, {len(TESTS)} kiểm định, {len(FIGURES)} biểu đồ).")


# --------------------------------------------------------------------------------------
# 6. Điều phối
# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="EDA Tuần 5 - doanh thu bán lẻ 2025")
    ap.add_argument("--topics", nargs="*", default=list("ABCDEFG"), type=str.upper,
                    help="các chủ đề cần chạy (A-G), mặc định chạy tất cả")
    ap.add_argument("--refresh", action="store_true", help="bỏ cache, truy vấn lại PostgreSQL")
    ap.add_argument("--show", action="store_true", help="hiện biểu đồ trên màn hình")
    args = ap.parse_args()
    topics = [t for t in args.topics if t in set("ABCDEFG")]
    if not topics:
        sys.exit("Không có chủ đề hợp lệ. Chọn trong A B C D E F G.")

    for p in (DATA, FIG, TAB):
        p.mkdir(parents=True, exist_ok=True)

    print("Đọc dữ liệu:")
    inv = prepare(load("invoice_valid", SQL_INVOICE, args.refresh), load("upt", SQL_UPT, args.refresh))
    quality = load("quality", SQL_QUALITY, args.refresh)
    ban = inv[inv.doanh_thu > 0]
    items = sizes = None
    if set(topics) & set("CF"):
        items = load("item_dt", SQL_ITEM, args.refresh)
    if "F" in topics:
        sizes = load("size_dt", SQL_SIZE, args.refresh)

    overview(inv, quality)
    if "A" in topics: topic_A(inv)
    if "B" in topics: topic_B(inv)
    if "C" in topics: topic_C(items)
    if "D" in topics: topic_D(ban)
    if "E" in topics: topic_E(ban)
    if "F" in topics: topic_F(inv, items, sizes)
    if "G" in topics: topic_G(inv, ban)
    write_report()


if __name__ == "__main__":
    main()