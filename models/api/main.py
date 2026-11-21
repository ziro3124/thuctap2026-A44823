"""API gợi ý sản phẩm (FastAPI).

    uvicorn api.main:app --reload --port 8000
    GET /recommend?customer_id=C_000123&k=10
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from src import config as C
from src.recommend import Recommender

app = FastAPI(title="Gợi ý sản phẩm cho khách thành viên", version="1.0")
_rec: Recommender | None = None


def get_rec() -> Recommender:
    global _rec
    if _rec is None:
        try:
            _rec = Recommender()
        except FileNotFoundError:
            raise HTTPException(503, "Chưa có artifacts/bundle.pkl. Chạy: python -m src.train --stage all")
    return _rec


@app.get("/health")
def health():
    r = get_rec()
    return {"status": "ok", "model": r.name, "trained_until": r.trained_until, "n_customers": len(r.ctx.users), "n_items": r.ctx.n_items}


@app.get("/recommend")
def recommend(customer_id: str = Query(..., min_length=1), k: int = Query(C.K, ge=1, le=50), size_filter: bool = True):
    """Top-k sản phẩm gợi ý. Khách chưa có lịch sử nhận top bán chạy (source = fallback_popularity)."""
    r = get_rec()
    df = r.recommend(customer_id, k, size_filter)
    return {"customer_id": customer_id, "known_customer": r.known(customer_id), "k": k,
            "items": df[["rank", "item", "product_name", "group_cat", "score", "source"]].to_dict("records")}
