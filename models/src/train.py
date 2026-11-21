"""Huấn luyện, chọn mô hình, đánh giá test, đóng gói. Theo dõi thí nghiệm bằng MLflow.

    python -m src.train --stage all            # toàn bộ pipeline
    python -m src.train --stage tune           # chọn mô hình/tham số trên validation (CV theo thời gian)
    python -m src.train --stage final          # huấn luyện lại tại ngày chốt test và đánh giá (một lần)
    python -m src.train --stage deploy         # huấn luyện lại bằng toàn bộ 2025, lưu artifacts/bundle.pkl
    python -m src.train --stage all --quick    # chạy nhanh để kiểm tra pipeline
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import time

import mlflow
import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import models as M
from .metrics import evaluate, summarize
from .split import build_ctx

EXPERIMENT = "recsys-de-d-goi-y-san-pham"
NDCG = f"ndcg@{C.K}"


# =============================================================================== dữ liệu
def prepare(use_cache: bool = True) -> dict:
    raw = F.load_raw(use_cache=use_cache)
    n_inv = int(raw["meta"].n_inv_total[0])
    blacklist, stats = F.find_non_merch(raw["item_stats"], n_inv, C.NON_MERCH_MANUAL)
    stats.sort_values("qty", ascending=False).head(200).to_csv(C.REPORTS / "item_kiem_tra_khong_phai_hang_hoa.csv", index=False)
    F.unmapped_sizes(raw["sku"]).to_csv(C.REPORTS / "size_chua_quy_doi.csv", index=False)
    inter = raw["inter"][~raw["inter"].item.isin(blacklist)].reset_index(drop=True)
    age = F.item_age_table(raw["sku"])
    vocab = F.build_vocab(raw["items"])
    D = dict(raw=raw, blacklist=blacklist, inter=inter, age=age, vocab=vocab, items=raw["items"])
    D["val"] = [build_ctx(inter, raw["items"], age, vocab, *f) for f in C.VAL_FOLDS]
    D["test"] = build_ctx(inter, raw["items"], age, vocab, *C.TEST)
    print(f"Loại {len(blacklist)} mã không phải hàng hóa. " + " | ".join(c.summary() for c in D["val"] + [D["test"]]))
    return D


# =============================================================================== MLflow
def _log(run_name: str, params: dict, metrics: dict, tags: dict, artifacts: list | None = None):
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(tags)
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(v)})
        for a in artifacts or []:
            mlflow.log_artifact(str(a))


def _metrics(df: pd.DataFrame, prefix: str) -> dict:
    out = {}
    for _, r in df.iterrows():
        for m in (f"precision@{C.K}", f"recall@{C.K}", NDCG, f"hit_rate@{C.K}"):
            out[f"{prefix}{r.nhom}_{m.replace('@', '_at_')}"] = r[m]
        if r.nhom == "tat_ca":
            out[f"{prefix}coverage"] = r.coverage
    return out


# =============================================================================== tune
def _score_cv(ctxs, build, name, params) -> dict:
    """Huấn luyện `build()` trên từng fold, đánh giá và log MLflow. Trả về điểm của fold cuối (fold gần ngày test nhất)."""
    rows, t0 = [], time.time()
    for k, ctx in enumerate(ctxs):
        m = build().fit(ctx)
        df = summarize(evaluate(ctx, m), name)
        df["fold"] = k + 1
        rows.append(df)
    allf = pd.concat(rows)
    last = allf[allf.fold == len(ctxs)]
    mean_ndcg = allf[allf.nhom == "tat_ca"][NDCG].mean()
    mt = _metrics(last, "val_")
    mt["val_cv_mean_ndcg"] = mean_ndcg
    mt["giay"] = time.time() - t0
    _log(f"val/{name}", params, mt, {"stage": "validation", "model": name.split(" ")[0]})
    allf["params"] = json.dumps(params)
    return {"name": name, "params": params, "val_ndcg": float(last[last.nhom == "tat_ca"][NDCG].iloc[0]),
            "cv_mean_ndcg": float(mean_ndcg), "table": allf}


def tune(D: dict, quick: bool = False, trials: int = 15) -> dict:
    mlflow.set_tracking_uri(C.MLRUNS_URI)
    mlflow.set_experiment(EXPERIMENT)
    v = D["val"]
    res = []
    res.append(_score_cv(v, lambda: M.PopGlobal(30), "PopGlobal", {"days": 30}))
    res.append(_score_cv(v, lambda: M.PopSegment(30), "PopSegment", {"days": 30}))
    for k, tau in itertools.product([20, 50], [60, 120]) if not quick else [(50, 120)]:
        res.append(_score_cv(v, lambda k=k, tau=tau: M.ItemKNN(k, tau), f"ItemKNN k={k} tau={tau}", {"k": k, "tau": tau}))
    for f, reg in itertools.product([32, 64], [0.05, 0.5]) if not quick else [(32, 0.5)]:
        res.append(_score_cv(v, lambda f=f, reg=reg: M.ALS(f, reg, 40), f"ALS f={f} reg={reg}", {"factors": f, "reg": reg, "alpha": 40}))
    tt_grid = [(32, 15), (32, 30), (64, 30)] if not quick else [(32, 6)]
    for dim, ep in tt_grid:
        res.append(_score_cv(v, lambda dim=dim, ep=ep: M.TwoTower(dim, ep), f"TwoTower dim={dim} ep={ep}", {"dim": dim, "epochs": ep, "tau": 0.1}))

    def best_of(prefix):
        c = [r for r in res if r["name"].startswith(prefix)]
        return max(c, key=lambda r: r["val_ndcg"])
    bk, ba, bt = best_of("ItemKNN"), best_of("ALS"), best_of("TwoTower")

    # ---- Ranker LightGBM (tiến tới: huấn luyện trên fold 1, kiểm tra trên fold 2; Optuna chọn tham số) -----------------
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    c1, c2 = v[0], v[-1]
    l1_1 = M.make_level1(c1, (bk["params"]["k"], bk["params"]["tau"]), (ba["params"]["factors"], ba["params"]["reg"], 40))
    l1_2 = M.make_level1(c2, (bk["params"]["k"], bk["params"]["tau"]), (ba["params"]["factors"], ba["params"]["reg"], 40))
    fr1, fr2 = rank_frames(c1, l1_1), rank_frames(c2, l1_2)
    print(f"Ranker: {len(fr1[0]):,} cặp huấn luyện (fold 1), {len(fr2[0]):,} cặp kiểm tra (fold 2), {fr1[0].shape[1]} feature")

    def objective(trial):
        p = dict(num_leaves=trial.suggest_int("num_leaves", 7, 63), learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0), reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10, log=True))
        b = M.train_ranker([fr1], [fr2], p)
        r = evaluate(c2, M.RankerModel(l1_2, b))
        trial.set_user_attr("best_iter", int(b.best_iteration_ or b.n_estimators))
        sc = float(r.ndcg.mean())
        _log(f"val/LGBMRanker trial {trial.number}", p, {"val_tat_ca_ndcg_at_10": sc}, {"stage": "validation", "model": "LGBMRanker"})
        return sc
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(objective, n_trials=3 if quick else trials)
    bp = dict(study.best_params); bp["n_estimators"] = max(50, int(study.best_trial.user_attrs["best_iter"]))
    # điểm chuẩn của ranker trên fold 2 (cùng cách tính với các mô hình khác)
    b = M.train_ranker([fr1], [fr2], bp)
    rr = summarize(evaluate(c2, M.RankerModel(l1_2, b)), "LGBMRanker")
    rr["fold"] = len(v)
    res.append({"name": "LGBMRanker", "params": bp, "val_ndcg": float(study.best_value), "cv_mean_ndcg": float(study.best_value), "table": rr.assign(params=json.dumps(bp))})

    table = pd.concat([r["table"] for r in res], ignore_index=True)
    table.to_csv(C.REPORTS / "ket_qua_validation.csv", index=False)
    best = {"ItemKNN": bk["params"], "ALS": ba["params"], "TwoTower": bt["params"], "LGBMRanker": bp,
            "val_ndcg": {r["name"]: r["val_ndcg"] for r in res}}
    (C.REPORTS / "best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    top = sorted(res, key=lambda r: -r["val_ndcg"])
    print("Xếp hạng trên validation (NDCG@10, fold cuối):")
    for r in top[:8]:
        print(f"  {r['val_ndcg']:.4f}  {r['name']}")
    return best


def rank_frames(ctx, l1):
    """Khung feature + nhãn cho mọi khách kiểm tra của ctx (xử lý theo khối để tiết kiệm bộ nhớ)."""
    Xs, ys, us, is_ = [], [], [], []
    for a in range(0, len(ctx.rows), 1500):
        sl = slice(a, a + 1500)
        X, y, u, i = M.build_rank_frame(ctx, l1, ctx.rows[sl], target=ctx.target[sl])
        Xs.append(X); ys.append(y); us.append(u + a); is_.append(i)
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(us), np.concatenate(is_)


# =============================================================================== final (test, chạy một lần)
def build_models(D: dict, best: dict, ctx) -> dict:
    """Huấn luyện mọi mô hình đã chọn trên `ctx`. Ranker dùng booster huấn luyện từ nhãn của các fold validation (< ngày test)."""
    kp, ap = best["ItemKNN"], best["ALS"]
    l1 = M.make_level1(ctx, (kp["k"], kp["tau"]), (ap["factors"], ap["reg"], 40))
    # booster: huấn luyện trên nhãn của CẢ HAI fold validation (nhãn đều < ngày chốt test)
    parts = []
    for c in D["val"]:
        l1v = M.make_level1(c, (kp["k"], kp["tau"]), (ap["factors"], ap["reg"], 40))
        parts.append(rank_frames(c, l1v))
    booster = M.train_ranker(parts, None, best["LGBMRanker"])
    tp = best["TwoTower"]
    return {"PopGlobal": l1["pop"], "PopSegment": l1["seg"], f"ItemKNN k={kp['k']} tau={kp['tau']}": l1["knn"],
            f"ALS f={ap['factors']} reg={ap['reg']}": l1["als"],
            "TwoTower": M.TwoTower(tp["dim"], tp["epochs"], tp.get("tau", 0.1)).fit(ctx),
            "LGBMRanker": M.RankerModel(l1, booster)}, booster


def final(D: dict, best: dict) -> pd.DataFrame:
    mlflow.set_tracking_uri(C.MLRUNS_URI)
    mlflow.set_experiment(EXPERIMENT)
    ctx = D["test"]
    models, booster = build_models(D, best, ctx)
    rows, results = [], {}
    for name, m in models.items():
        for sf in (False, True):
            label = name + (" + lọc size" if sf else "")
            r = evaluate(ctx, m, size_filter=sf)
            results[label] = r
            df = summarize(r, label)
            df["loc_size"] = sf
            rows.append(df)
            _log(f"test/{label}", {"model": name, "size_filter": sf}, _metrics(df, "test_"), {"stage": "test", "model": name.split(" ")[0]})
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(C.REPORTS / "ket_qua_test.csv", index=False)
    imp = pd.DataFrame({"feature": booster.feature_name_, "gain": booster.booster_.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp.to_csv(C.REPORTS / "ranker_feature_importance.csv", index=False)
    tt = models["TwoTower"]
    pd.DataFrame({"epoch": range(1, len(tt.loss_curve) + 1), "loss": tt.loss_curve}).to_csv(C.REPORTS / "twotower_loss.csv", index=False)
    with open(C.ART / "test_run.pkl", "wb") as f:
        pickle.dump({"ctx": ctx, "models": models, "best": best}, f)
    print(table[(table.nhom == "tat_ca") & (~table.loc_size)][["mo_hinh", f"precision@{C.K}", f"recall@{C.K}", NDCG, "coverage"]].round(4).to_string(index=False))
    return table


# =============================================================================== deploy
def deploy(D: dict, best: dict, override: str | None = None) -> str:
    """Chọn mô hình học máy tốt nhất theo NDCG validation (không theo test; baseline Pop* không được triển khai trừ khi
    chỉ định bằng --deploy-model), huấn luyện lại bằng toàn bộ 2025, lưu bundle."""
    cand = {k: v for k, v in best["val_ndcg"].items() if not k.startswith("Pop")}
    chosen = override or (max(cand, key=cand.get) if cand else "PopSegment")
    ctx = build_ctx(D["inter"], D["items"], D["age"], D["vocab"], C.DEPLOY_CUTOFF, None)
    kp, ap = best["ItemKNN"], best["ALS"]
    l1 = M.make_level1(ctx, (kp["k"], kp["tau"]), (ap["factors"], ap["reg"], 40))
    if chosen.startswith("LGBMRanker"):
        parts = [rank_frames(c, M.make_level1(c, (kp["k"], kp["tau"]), (ap["factors"], ap["reg"], 40))) for c in D["val"]]
        model = M.RankerModel(l1, M.train_ranker(parts, None, best["LGBMRanker"]))
    elif chosen.startswith("TwoTower"):
        tp = best["TwoTower"]; model = M.TwoTower(tp["dim"], tp["epochs"], tp.get("tau", 0.1)).fit(ctx)
    elif chosen.startswith("ItemKNN"):
        model = l1["knn"]
    elif chosen.startswith("ALS"):
        model = l1["als"]
    else:
        model = l1["seg"]
    pop = M.PopGlobal(30).fit(ctx)
    ctx.train = ctx.train.iloc[:0]                    # bundle nhẹ hơn: mô hình đã huấn luyện không cần giao dịch thô
    ctx._mat = {}
    path = C.ART / "bundle.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model_name": chosen, "model": model, "ctx": ctx, "fallback": pop, "blacklist": sorted(D["blacklist"]), "trained_until": C.DEPLOY_CUTOFF}, f)
    print(f"Đã lưu {path} (mô hình: {chosen})")
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "final", "deploy", "all"], default="all")
    ap.add_argument("--quick", action="store_true", help="lưới nhỏ, ít epoch/trial: chỉ để kiểm tra pipeline chạy được")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--deploy-model", default=None, help="ép mô hình triển khai, ví dụ LGBMRanker, TwoTower, PopSegment")
    ap.add_argument("--no-cache", action="store_true", help="đọc lại từ PostgreSQL thay vì cache trong data/")
    a = ap.parse_args()
    np.random.seed(C.SEED)
    D = prepare(use_cache=not a.no_cache)
    if a.stage in ("tune", "all"):
        best = tune(D, a.quick, a.trials)
    else:
        best = json.loads((C.REPORTS / "best_config.json").read_text(encoding="utf-8"))
    if a.stage in ("final", "all"):
        final(D, best)
    if a.stage in ("deploy", "all"):
        deploy(D, best, a.deploy_model)


if __name__ == "__main__":
    main()
