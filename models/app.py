"""Giao diện demo hệ thống gợi ý sản phẩm (Đề D).

Chạy (từ thư mục models/):   streamlit run app.py
Trang "Demo gợi ý" gọi thẳng mô hình đã huấn luyện (artifacts/bundle.pkl); nếu chưa có bundle,
trang tự chuyển sang tra cứu file reports/goi_y_top10.csv. Các trang còn lại đọc file kết quả trong reports/.
"""
from __future__ import annotations

import json
import re

import altair as alt
import pandas as pd
import streamlit as st

from src import config as C

REP = C.REPORTS
st.set_page_config(page_title="Gợi ý sản phẩm", page_icon="🛍️", layout="wide")

BEST = "LGBMRanker"
NICE = {"PopGlobal": "Bán chạy toàn chuỗi", "PopSegment": "Bán chạy theo giới tính/mùa",
        "ItemKNN k=50 tau=120": "ItemKNN", "ALS f=64 reg=0.05": "ALS", "TwoTower": "Two-Tower (deep learning)",
        "LGBMRanker": "LightGBM xếp hạng hai tầng"}
METRICS = {"recall@10": "Recall@10", "ndcg@10": "NDCG@10", "precision@10": "Precision@10",
           "hit_rate@10": "Hit rate@10", "coverage": "Coverage", "da_dang_nganh": "Đa dạng ngành"}
GROUP_VN = {"tat_ca": "Tất cả", "mua_2_lan": "Khách mua 1 lần trước đó", "mua_3_lan_tro_len": "Khách mua ≥2 lần trước đó"}


# ----------------------------------------------------------------------------- đọc dữ liệu
@st.cache_data
def read_csv(name: str):
    p = REP / name
    return pd.read_csv(p) if p.exists() else None


def need(df, name):
    if df is None:
        st.warning(f"Chưa có `reports/{name}`. Chạy lại bước tạo kết quả (train/evaluate) rồi tải lại trang.")
        return False
    return True


def base_name(m: str) -> str:
    return m.replace(" + lọc size", "")


def nice(m: str) -> str:
    return NICE.get(base_name(m), base_name(m)) + (" + lọc size" if m.endswith("lọc size") else "")


@st.cache_resource(show_spinner="Đang nạp mô hình…")
def load_reco():
    try:
        from src.recommend import Recommender
        return Recommender(), None
    except Exception as e:  # chưa train, thiếu thư viện, ...
        return None, f"{type(e).__name__}: {e}"


def bar(df, x, y, title, fmt=".3f", color=None, horizontal=True):
    enc_x, enc_y = (alt.X(f"{x}:Q", title=title), alt.Y(f"{y}:N", sort="-x", title=None)) if horizontal \
        else (alt.X(f"{x}:N", title=None), alt.Y(f"{y}:Q", title=title))
    base = alt.Chart(df).encode(x=enc_x, y=enc_y, tooltip=list(df.columns))
    if color:
        base = base.encode(color=alt.Color(f"{color}:N", legend=None))
    ch = base.mark_bar()
    txt = base.mark_text(align="left", dx=3).encode(text=alt.Text(f"{x}:Q", format=fmt)) if horizontal else None
    return (ch + txt if txt is not None else ch).properties(height=max(120, 34 * len(df)) if horizontal else 280)


# ----------------------------------------------------------------------------- các trang
def page_overview():
    st.title("🛍️ Gợi ý sản phẩm cho khách hàng chuỗi thời trang trẻ em")
    st.caption("Đề D · Mô hình học máy / học sâu gợi ý 10 sản phẩm tiếp theo cho từng khách hàng")
    test = read_csv("ket_qua_test.csv")
    if not need(test, "ket_qua_test.csv"):
        return
    t = test[(test.nhom == "tat_ca") & (~test.loc_size)].set_index("mo_hinh")
    best, g, s = t.loc[BEST], t.loc["PopGlobal"], t.loc["PopSegment"]
    c = st.columns(4)
    c[0].metric("Recall@10 (mô hình tốt nhất)", f"{best['recall@10']:.3f}",
                f"{best['recall@10'] / g['recall@10']:.1f}× so với bán chạy toàn chuỗi", delta_color="off")
    c[1].metric("Recall@10 so với baseline mạnh nhất", f"{best['recall@10']:.3f}",
                f"{best['recall@10'] / s['recall@10'] - 1:+.0%} vs bán chạy theo giới tính", delta_color="off")
    c[2].metric("Hit rate@10", f"{best['hit_rate@10']:.1%}", "khách có ≥1 gợi ý đúng", delta_color="off")
    c[3].metric("Số khách kiểm tra", f"{int(best['so_khach']):,}", "tập test 11/2025 – 12/2025", delta_color="off")

    st.subheader("Bài toán và cách đánh giá")
    st.markdown(
        "- **Đầu vào:** lịch sử mua hàng của khách (mẫu, size, ngành hàng, thời điểm).\n"
        "- **Đầu ra:** 10 mẫu sản phẩm khách có khả năng mua trong lần mua kế tiếp, không gồm mẫu đã mua.\n"
        "- **Chia dữ liệu theo thời gian:** huấn luyện bằng dữ liệu trước ngày chốt, kiểm tra trên giao dịch sau ngày chốt, "
        "nên không có rò rỉ dữ liệu tương lai (có kiểm thử tự động).\n"
        "- **Chỉ gợi ý mẫu còn bán** (có giao dịch trong 60 ngày gần nhất) và loại các mã không phải hàng hóa (túi, quà tặng).\n"
        "- **Chọn mô hình** trên 2 fold validation, chạm tập test đúng một lần.")
    loi = read_csv("phan_tich_loi.csv")
    if loi is not None:
        ceil = loi[loi.chieu_phan_tich == "tran_recall"]["recall@10"]
        if len(ceil):
            st.info(f"**Trần Recall:** chỉ {ceil.iloc[0]:.0%} số mẫu khách mua trong tập test là mẫu có thể gợi ý được "
                    f"(mẫu mới ra, mã ngừng bán nằm ngoài danh sách ứng viên). Recall {best['recall@10']:.3f} vì vậy "
                    "cần đọc cùng con số trần này.")
    st.subheader("Các mô hình đã so sánh")
    st.dataframe(pd.DataFrame({"Mô hình": [nice(m) for m in t.index],
                               "Loại": ["Baseline", "Baseline", "Học máy", "Học máy", "Học sâu", "Học máy (hai tầng)"]
                               if len(t) == 6 else ["" for _ in t.index]}), hide_index=True, width="stretch")


def page_compare():
    st.title("So sánh mô hình")
    test = read_csv("ket_qua_test.csv")
    if not need(test, "ket_qua_test.csv"):
        return
    c1, c2 = st.columns(2)
    grp = c1.selectbox("Nhóm khách", list(GROUP_VN), format_func=GROUP_VN.get)
    size = c2.toggle("Có lọc size theo tuổi bé", value=False)
    d = test[(test.nhom == grp) & (test.loc_size == size)].copy()
    d["Mô hình"] = d.mo_hinh.map(nice)
    tabs = st.tabs(["Recall@10", "NDCG@10", "Hit rate@10", "Coverage", "Đa dạng ngành"])
    for tab, m in zip(tabs, ["recall@10", "ndcg@10", "hit_rate@10", "coverage", "da_dang_nganh"]):
        with tab:
            fmt = ".2f" if m == "da_dang_nganh" else (".1%" if m in ("coverage", "hit_rate@10") else ".4f")
            dd = d[["Mô hình", m]].copy()
            dd["tốt nhất"] = dd["Mô hình"].str.startswith(NICE[BEST])
            st.altair_chart(bar(dd, m, "Mô hình", METRICS[m], fmt, color="tốt nhất"), width="stretch")
    st.caption("Coverage thấp của mô hình xếp hạng là do danh sách ứng viên chỉ gồm mẫu còn bán; "
               "ItemKNN có coverage cao nhưng độ chính xác thấp.")

    st.subheader("Chênh lệch có ý nghĩa thống kê không? (bootstrap 95%)")
    bs = read_csv("ket_qua_bootstrap.csv")
    if bs is not None:
        b = bs.copy()
        b["so_voi"] = b.so_voi.map(nice)
        b = b.rename(columns={"so_voi": "So với", "chi_so": "Chỉ số", "chenh_lech": "Chênh lệch",
                              "ktc95_thap": "KTC 95% thấp", "ktc95_cao": "KTC 95% cao", "co_y_nghia": "Có ý nghĩa"})
        r = b[b["Chỉ số"] == "recall@10"]
        pts = alt.Chart(r).mark_point(filled=True, size=70).encode(
            y=alt.Y("So với:N", title=None, sort=None), x=alt.X("Chênh lệch:Q", title="Recall@10 hơn mô hình đối chiếu"))
        rule = alt.Chart(r).mark_rule().encode(y="So với:N", x="KTC 95% thấp:Q", x2="KTC 95% cao:Q")
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart((zero + rule + pts).properties(height=200), width="stretch")
        st.caption(f"{nice(BEST)} so với từng mô hình; khoảng tin cậy không chứa 0 nghĩa là hơn thật sự, "
                   "không phải do may rủi mẫu.")
        st.dataframe(b[["So với", "Chỉ số", "Chênh lệch", "KTC 95% thấp", "KTC 95% cao", "Có ý nghĩa"]],
                     hide_index=True, width="stretch")

    st.subheader("Xếp hạng trên validation (fold cuối)")
    val = read_csv("ket_qua_validation.csv")
    if val is not None:
        v = val[(val.nhom == "tat_ca") & (val.fold == val.fold.max())].sort_values("ndcg@10", ascending=False)
        v = v[["mo_hinh", "ndcg@10", "recall@10", "precision@10"]].rename(columns={"mo_hinh": "Cấu hình"})
        st.dataframe(v, hide_index=True, width="stretch")


def page_analysis():
    st.title("Phân tích lỗi và giải thích mô hình")
    loi = read_csv("phan_tich_loi.csv")
    if need(loi, "phan_tich_loi.csv"):
        dims = {"so_lan_mua_truoc": "Số lần mua trước đó", "gioi_tinh_suy_ra": "Giới tính bé (suy ra)",
                "tuoi_uoc_tinh": "Tuổi ước tính", "thoi_gian_tu_lan_mua_cuoi": "Số ngày từ lần mua cuối",
                "nganh_hang_cua_mau_nhan": "Ngành hàng của mẫu cần đoán"}
        dim = st.selectbox("Chiều phân tích", list(dims), format_func=dims.get)
        d = loi[loi.chieu_phan_tich == dim].copy()
        d["nhom"] = d.nhom.astype(str)
        if dim == "tuoi_uoc_tinh":
            order = {"(0, 2]": 0, "(2, 4]": 1, "(4, 6]": 2, "(6, 8]": 3, "(8, 20]": 4}
            d = d.assign(o=d.nhom.map(order).fillna(9)).sort_values("o")
        st.altair_chart(alt.Chart(d).mark_bar().encode(
            x=alt.X("nhom:N", sort=None, title=None, axis=alt.Axis(labelAngle=-20, labelLimit=220)),
            y=alt.Y("recall@10:Q", title="Recall@10"),
            tooltip=["nhom", "so_khach", alt.Tooltip("recall@10:Q", format=".3f")]).properties(height=300),
            width="stretch")
        if dim == "nganh_hang_cua_mau_nhan":
            st.caption("Giày dép dễ đoán nhất (ít mẫu, mua lặp theo nhu cầu); áo và đầm khó nhất vì có quá nhiều mẫu tương tự nhau.")
        st.dataframe(d.drop(columns=[c for c in ("o",) if c in d]).rename(columns={"nhom": "Nhóm", "so_khach": "Số khách"}),
                     hide_index=True, width="stretch")

    st.subheader("Đặc trưng nào quan trọng nhất với mô hình xếp hạng?")
    fi = read_csv("ranker_feature_importance.csv")
    if fi is not None:
        top = fi.sort_values("gain", ascending=False).head(15)
        st.altair_chart(bar(top, "gain", "feature", "Gain", ".0f"), width="stretch")
        st.caption("Độ phổ biến của mẫu (pop_*), độ phổ biến theo nhóm giới tính/mùa (seg_*) và thứ hạng từ ItemKNN chiếm phần lớn; "
                   "độ lệch tuổi bé so với tuổi của mẫu (age_gap) cũng đóng góp đáng kể.")
    shap = REP / "ranker_shap.png"
    if shap.exists():
        st.image(str(shap), caption="SHAP: mức ảnh hưởng của từng đặc trưng lên điểm xếp hạng")

    st.subheader("Quá trình huấn luyện Two-Tower")
    tl = read_csv("twotower_loss.csv")
    if tl is not None:
        st.altair_chart(alt.Chart(tl).mark_line(point=True).encode(
            x=alt.X("epoch:Q", title="Epoch"), y=alt.Y("loss:Q", title="Loss", scale=alt.Scale(zero=False))
        ).properties(height=260), width="stretch")


def page_demo():
    st.title("Demo: gợi ý cho một khách hàng")
    reco, err = load_reco()
    csv = read_csv("goi_y_top10.csv")
    if reco is not None:
        st.success(f"Đang dùng mô hình **{nice(reco.name)}**, huấn luyện đến {reco.trained_until}.")
    else:
        st.warning("Chưa nạp được mô hình (`artifacts/bundle.pkl`), đang tra cứu từ `reports/goi_y_top10.csv`."
                   + (f"  \nChi tiết: `{err}`" if err else ""))
        if csv is None:
            st.error("Cũng chưa có `goi_y_top10.csv`. Chạy `python -m src.train --stage all` và `python -m src.predict_batch`.")
            return

    ids = list(csv.customer_id.drop_duplicates().head(300)) if csv is not None else \
        list(list(reco.ctx.user_idx)[:300])
    c1, c2, c3 = st.columns([2, 2, 1])
    pick = c1.selectbox("Chọn nhanh một khách", ids)
    typed = c2.text_input("…hoặc nhập mã khách", placeholder="C_xxxxxxxxxxxxxxxx")
    k = c3.slider("Số gợi ý", 3, 20, 10) if reco is not None else 10
    size_f = st.checkbox("Lọc size theo tuổi ước tính của bé", value=True, disabled=reco is None)
    cid = (typed or pick).strip()

    if reco is not None:
        recs = reco.recommend(cid, k, size_f)
        known = reco.known(cid)
        hist = []
        if known:
            r = reco.ctx.user_idx[cid]
            hist = [str(reco.ctx.names[j]) for j in reco.ctx.seen[r].indices]
    else:
        recs = csv[csv.customer_id == cid].sort_values("rank")
        known, hist = len(recs) > 0, []

    if recs.empty:
        st.error("Không có gợi ý cho mã này trong file kết quả.")
        return
    if reco is not None and not known:
        st.info("Khách chưa có lịch sử trong dữ liệu huấn luyện, nên hệ thống trả về các mẫu bán chạy toàn chuỗi (nguồn: fallback_popularity).")
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Đã mua")
        if hist:
            st.caption(f"{len(hist)} mẫu khác nhau")
            for h in hist[:25]:
                st.markdown(f"- {h}")
            if len(hist) > 25:
                st.caption(f"… và {len(hist) - 25} mẫu khác")
        else:
            st.caption("Không hiển thị lịch sử ở chế độ tra cứu file, hoặc khách mới.")
    with right:
        st.subheader(f"Top {len(recs)} gợi ý")
        show = recs[["rank", "product_name", "group_cat", "score", "source"]].rename(columns={
            "rank": "Hạng", "product_name": "Sản phẩm", "group_cat": "Ngành hàng", "score": "Điểm", "source": "Nguồn"})
        st.dataframe(show, hide_index=True, width="stretch")
        st.altair_chart(alt.Chart(recs).mark_bar().encode(
            x=alt.X("group_cat:N", title=None, axis=alt.Axis(labelAngle=-15, labelLimit=200)),
            y=alt.Y("count():Q", title="Số gợi ý")).properties(height=180), width="stretch")
        st.caption("Ngành hàng của danh sách gợi ý: danh sách đa dạng thì khách có nhiều lựa chọn hơn là chỉ lặp một loại áo.")


def page_examples():
    st.title("Ba ví dụ cụ thể (tập test)")
    p = REP / "vi_du_goi_y.md"
    if not p.exists():
        st.warning("Chưa có `reports/vi_du_goi_y.md`. Chạy `python -m src.evaluate`.")
        return
    txt = p.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^## ", txt)[1:]
    for part in parts:
        head, _, body = part.partition("\n")
        with st.expander(head.strip(), expanded=False):
            st.markdown(body.replace("✓", "✅"))
    st.caption("✅ là sản phẩm khách thực sự mua trong tập test. Ví dụ gồm cả trường hợp trúng và trượt để thấy giới hạn của mô hình.")


def page_limits():
    st.title("Giới hạn và hướng cải thiện")
    sz = read_csv("size_chua_quy_doi.csv")
    st.markdown(
        "**Giới hạn đã nhận diện từ kết quả thật**\n"
        "- Hầu hết khách chỉ mua vài lần nên tín hiệu cá nhân hóa yếu; mô hình xếp hạng chủ yếu học cách "
        "kết hợp độ phổ biến và nhóm giới tính/mùa, và hơn baseline mạnh nhất một mức nhỏ nhưng có ý nghĩa thống kê.\n"
        "- Two-Tower (học sâu) chưa vượt baseline vì dữ liệu thưa (hơn 99% ô khách×mẫu trống) và mẫu hàng thay đổi theo mùa.\n"
        "- Lọc size gần như không đổi kết quả vì nhiều SKU chưa quy đổi được size về tuổi.\n"
        "- Khách mới (chưa có lịch sử) chỉ nhận gợi ý bán chạy.\n")
    if sz is not None:
        st.subheader("Tên size chưa quy đổi được về tuổi")
        st.dataframe(sz.head(15).rename(columns={"size_name": "Tên size", "so_sku": "Số SKU"}),
                     hide_index=True, width="stretch")
    bp = REP / "best_config.json"
    if bp.exists():
        with st.expander("Tham số tốt nhất tìm được (Optuna / lưới)"):
            st.json({k: v for k, v in json.loads(bp.read_text(encoding="utf-8")).items() if k != "val_ndcg"})


PAGES = {"Tổng quan": page_overview, "So sánh mô hình": page_compare, "Phân tích lỗi": page_analysis,
         "Demo gợi ý": page_demo, "Ví dụ cụ thể": page_examples, "Giới hạn": page_limits}

with st.sidebar:
    st.header("Gợi ý sản phẩm")
    choice = st.radio("Trang", list(PAGES), label_visibility="collapsed")
    st.caption("Dữ liệu: giao dịch chuỗi bán lẻ thời trang trẻ em, đã che danh tính khách hàng.")
PAGES[choice]()
