"""Cấu hình chung của pipeline gợi ý sản phẩm (Đề D). Mọi tham số thí nghiệm nằm ở đây."""
from pathlib import Path

SEED = 42
K = 10                                   # số sản phẩm gợi ý cho mỗi khách

# ---- Cách chia dữ liệu theo thời gian ------------------------------------------------
# Mỗi "ngày chốt" T: huấn luyện chỉ dùng dữ liệu < T. Tập kiểm tra gồm các khách đã mua trước T
# và có mua trong [T, END); nhãn = các sản phẩm của hóa đơn CUỐI CÙNG của khách trong khoảng đó.
VAL_FOLDS = [("2025-07-01", "2025-09-01"), ("2025-09-01", "2025-11-01")]   # chọn mô hình (CV theo thời gian)
TEST = ("2025-11-01", "2026-01-01")                                         # kiểm tra cuối, chạy một lần
DEPLOY_CUTOFF = "2026-01-01"                                                # huấn luyện lại bằng toàn bộ 2025

ACTIVE_DAYS = 60                         # chỉ gợi ý mẫu có giao dịch trong 60 ngày trước ngày chốt (lọc hàng ngừng bán)
CANDIDATES_PER_SOURCE = 50               # số ứng viên mỗi nguồn cho mô hình xếp hạng hai tầng

# ---- Lọc size so với tuổi của bé -------------------------------------------------------
SIZE_FILTER_LO = 1.0                     # chấp nhận mẫu có tuổi lớn nhất >= tuổi ước tính - LO
SIZE_FILTER_HI = 2.0                     # và tuổi nhỏ nhất <= tuổi ước tính + HI (bé lớn dần, hay mua size rộng)

# ---- Mùa ------------------------------------------------------------------------------
SPRING_SUMMER_MONTHS = range(2, 8)       # tháng 2–7: Spring - Summer, còn lại Fall - Winter (giả định, đổi nếu cần)

# ---- Đường dẫn ------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts"
REPORTS = ROOT.parent / "reports" if (ROOT.parent / "reports").exists() else ROOT / "reports"
MLRUNS_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
for _p in (DATA, ART, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# Mã không phải hàng hóa cần loại thủ công (bổ sung sau khi xem bảng kiểm tra)
NON_MERCH_MANUAL = {"71110000"}          # hộp quà nhỏ
