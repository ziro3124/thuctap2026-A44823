import os

# implicit (ALS) khuyến nghị mỗi tiến trình dùng một luồng BLAS
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")     # LightGBM >= 4.7 đổi tên tham số; vẫn chạy được ở bản cũ
