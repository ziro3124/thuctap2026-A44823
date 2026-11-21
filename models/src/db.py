"""Kết nối PostgreSQL (dùng lại từ Tuần 1). Đọc thông tin từ file .env, không đưa mật khẩu vào code.

Tìm file .env theo thứ tự: biến môi trường DOTENV_PATH; thư mục đang chạy và các thư mục cha; thư mục models/, thư mục gốc
repo và notebooks/. Dùng file đầu tiên tìm thấy.

Chấp nhận tên biến chuẩn của PostgreSQL (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD, PGSCHEMA), các tên thường gặp khác
(DB_HOST, DB_USER, POSTGRES_USER, ...) hoặc một DATABASE_URL.
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

_ALIASES = {
    "host": ["PGHOST", "DB_HOST", "POSTGRES_HOST", "DATABASE_HOST"],
    "port": ["PGPORT", "DB_PORT", "POSTGRES_PORT", "DATABASE_PORT"],
    "database": ["PGDATABASE", "DB_NAME", "DB_DATABASE", "POSTGRES_DB", "DATABASE_NAME"],
    "user": ["PGUSER", "DB_USER", "DB_USERNAME", "POSTGRES_USER", "DATABASE_USER"],
    "password": ["PGPASSWORD", "DB_PASSWORD", "DB_PASS", "POSTGRES_PASSWORD", "DATABASE_PASSWORD"],
    "schema": ["PGSCHEMA", "DB_SCHEMA", "POSTGRES_SCHEMA", "DATABASE_SCHEMA"],
}


def _candidates():
    if os.environ.get("DOTENV_PATH"):
        yield Path(os.environ["DOTENV_PATH"])
    here = Path(__file__).resolve()
    models_dir, repo = here.parents[1], here.parents[2]
    for d in [Path.cwd(), *Path.cwd().parents, models_dir, repo, repo / "notebooks"]:
        yield d / ".env"


def _load_env():
    seen = set()
    for p in _candidates():
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        if p.is_file():
            load_dotenv(p, encoding="utf-8-sig", override=False)      # utf-8-sig: bỏ ký tự BOM nếu trình soạn thảo lưu kèm
            return p
    return None


_ENV_FILE = _load_env()
_engine = None


def _get(key):
    for name in _ALIASES[key]:
        v = os.environ.get(name)
        if v not in (None, ""):
            return v
    return None


def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    schema = _get("schema") or "masked_data"
    # mọi cột trong database là TEXT, truy vấn tự ép kiểu; đặt search_path để không cần ghi tên schema
    connect_args = {"options": f"-csearch_path={schema},public"}
    url = os.environ.get("DATABASE_URL")
    if url:
        for prefix in ("postgres://", "postgresql://"):          # ép dùng psycopg2, nhất quán với requirements.txt
            if url.startswith(prefix):
                url = "postgresql+psycopg2://" + url[len(prefix):]
        _engine = create_engine(url, connect_args=connect_args)
        return _engine
    host, user, db = _get("host"), _get("user"), _get("database")
    missing = [k for k, v in (("host", host), ("database", db), ("user", user)) if not v]
    if missing:
        loaded = sorted(k for k in os.environ if k.startswith(("PG", "DB_", "POSTGRES", "DATABASE")))
        raise RuntimeError(
            f"Thiếu thông tin kết nối: {missing}.\n"
            f"File .env đã nạp: {_ENV_FILE or 'KHÔNG TÌM THẤY (đã tìm ở thư mục chạy, thư mục cha, models/, gốc repo, notebooks/)'}.\n"
            f"Các biến liên quan đang có: {loaded or 'không có'}.\n"
            "Cần các dòng (tên biến chính xác, không có dấu cách thừa):\n"
            "PGHOST=...\nPGPORT=5432\nPGDATABASE=...\nPGUSER=...\nPGPASSWORD=...\nPGSCHEMA=masked_data")
    _engine = create_engine(
        URL.create("postgresql+psycopg2", username=user, password=_get("password") or "", host=host,
                   port=int(_get("port") or 5432), database=db),         # URL.create tự mã hóa ký tự đặc biệt trong mật khẩu (@, :, /)
        connect_args=connect_args)
    return _engine


def query(sql: str) -> pd.DataFrame:
    with get_engine().connect() as con:
        return pd.read_sql(text(sql), con)
