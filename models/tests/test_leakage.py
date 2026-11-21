"""Kiểm tra không rò rỉ dữ liệu. Chạy: pytest -q

Ý tưởng: nếu feature chỉ dùng thông tin trước ngày chốt thì thêm bất kỳ dữ liệu TƯƠNG LAI nào vào đầu vào
cũng không được làm đổi feature. Test dựng dữ liệu giả, thêm dữ liệu tương lai (kể cả dữ liệu cực đoan) và so sánh.
"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import features as F
from src.split import build_ctx

CUT = pd.Timestamp("2025-09-01")


def make_data(seed=0):
    r = np.random.default_rng(seed)
    n = 4000
    inter = pd.DataFrame({
        "customer_id": r.choice([f"C{i}" for i in range(300)], n), "item": r.choice([f"I{i}" for i in range(60)], n),
        "ts": pd.Timestamp("2025-01-01") + pd.to_timedelta(r.integers(0, 365 * 24, n), unit="h"),
        "qty": 1.0, "amount": r.choice([150e3, 250e3, 350e3], n), "size_name": r.choice(["2Y", "4Y", "6Y"], n)})
    inter["invoice_id"] = inter.customer_id + "_" + inter.ts.dt.strftime("%Y%m%d")
    inter = inter.drop_duplicates(["customer_id", "item", "invoice_id"]).reset_index(drop=True)
    items = pd.DataFrame({"item": [f"I{i}" for i in range(60)], "product_name": "x", "group_cat": r.choice(["A", "B", "C"], 60),
                          "gender": r.choice(["Bé gái", "Bé trai", "Unisex"], 60), "season": "Year-round",
                          "licenses": r.choice(["Disney", "Own"], 60), "unit_price": "250000"})
    sku = pd.DataFrame({"item": items.item.repeat(3).values, "size_name": ["2Y", "4Y", "6Y"] * 60})
    return inter, items, sku


def future_junk(inter):
    """Dữ liệu tương lai cực đoan: mọi khách mua mọi mẫu sau ngày chốt."""
    ids = inter.customer_id.unique()
    its = inter.item.unique()
    j = pd.DataFrame([(c, i) for c in ids for i in its[:20]], columns=["customer_id", "item"])
    j["ts"] = CUT + pd.Timedelta(days=5)
    j["qty"], j["amount"], j["size_name"], j["invoice_id"] = 99.0, 9e9, "12Y", "FUTURE"
    return pd.concat([inter, j], ignore_index=True)


def test_features_invariant_to_future_data():
    inter, items, sku = make_data()
    age, vocab = F.item_age_table(sku), F.build_vocab(items)
    a = build_ctx(inter, items, age, vocab, CUT)
    b = build_ctx(future_junk(inter), items, age, vocab, CUT)       # cùng ngày chốt, thêm dữ liệu tương lai
    assert list(a.items) == list(b.items) and list(a.users) == list(b.users)
    pd.testing.assert_frame_equal(a.item_meta, b.item_meta)
    pd.testing.assert_frame_equal(a.user_meta, b.user_meta)
    assert (a.seen != b.seen).nnz == 0
    np.testing.assert_allclose(a.ug, b.ug)


def test_train_strictly_before_cutoff_and_labels_after():
    inter, items, sku = make_data()
    age, vocab = F.item_age_table(sku), F.build_vocab(items)
    ctx = build_ctx(inter, items, age, vocab, CUT, pd.Timestamp("2025-11-01"))
    assert ctx.train.ts.max() < CUT
    win = inter[(inter.ts >= CUT) & (inter.ts < pd.Timestamp("2025-11-01"))]
    assert set(ctx.eval_users) <= set(win.customer_id), "khách kiểm tra phải có giao dịch trong kỳ kiểm tra"
    assert set(ctx.eval_users) <= set(ctx.users), "khách kiểm tra phải có lịch sử trước ngày chốt"


def test_target_is_last_invoice_only():
    inter, items, sku = make_data()
    age, vocab = F.item_age_table(sku), F.build_vocab(items)
    end = pd.Timestamp("2025-12-31")
    ctx = build_ctx(inter, items, age, vocab, CUT, end)
    for i in range(min(30, len(ctx.eval_users))):
        u = ctx.eval_users[i]
        w = inter[(inter.customer_id == u) & (inter.ts >= CUT) & (inter.ts < end)].sort_values(["ts", "invoice_id"])
        last_inv = w.invoice_id.iloc[-1]
        expect = set(w[w.invoice_id == last_inv].item)
        got = set(ctx.items[ctx.target[i].nonzero()[1]])
        assert got <= expect


def test_label_cannot_be_inferred_from_features():
    """Đổi nhãn (đảo toàn bộ giao dịch sau ngày chốt) không được làm đổi feature."""
    inter, items, sku = make_data()
    age, vocab = F.item_age_table(sku), F.build_vocab(items)
    other = inter.copy()
    m = other.ts >= CUT
    other.loc[m, "item"] = other.loc[m, "item"].sample(frac=1, random_state=1).values
    a = build_ctx(inter, items, age, vocab, CUT)
    b = build_ctx(other, items, age, vocab, CUT)
    pd.testing.assert_frame_equal(a.user_meta, b.user_meta)
    pd.testing.assert_frame_equal(a.item_meta, b.item_meta)


def test_feature_functions_reject_future_rows():
    inter, items, sku = make_data()
    age, vocab = F.item_age_table(sku), F.build_vocab(items)
    with pytest.raises(AssertionError):
        F.user_features(np.array(["C1"]), items, age, inter, CUT)         # inter chứa dòng >= CUT
