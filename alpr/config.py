"""Cấu hình chung cho toàn dự án."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best.pt"   # trọng số YOLO biển số (pre-trained, KHÔNG huấn luyện lại)
DB_PATH = ROOT / "data" / "plates.db"
CROP_DIR = ROOT / "data" / "crops"
RESULT_DIR = ROOT / "data" / "results"

DET_CONF = 0.40          # ngưỡng tin cậy khi phát hiện biển số
IMGSZ = 640              # kích thước ảnh đưa vào YOLO
BOX_PAD = 0.05           # nới khung biển số 5% mỗi cạnh trước khi cắt
TWO_LINE_ASPECT = 2.5    # rộng/cao < 2.5 -> coi là biển 2 dòng
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."   # có "-" và "." để dấu không bị ép thành chữ


def ensure_dirs():
    for d in (MODEL_PATH.parent, CROP_DIR, RESULT_DIR):
        d.mkdir(parents=True, exist_ok=True)