"""Tiền xử lý vùng biển số: phóng to, chuyển xám, khử nhiễu, CLAHE, chỉnh nghiêng."""
import cv2
import numpy as np


def upscale(img, min_h: int = 96):
    h, w = img.shape[:2]
    if h >= min_h:
        return img
    s = min_h / h
    return cv2.resize(img, (max(1, int(w * s)), min_h), interpolation=cv2.INTER_CUBIC)


def estimate_skew(gray, max_angle: float = 10.0, step: float = 1.0) -> float:
    """Ước lượng góc nghiêng (độ) bằng cách tìm góc làm profile hàng của ảnh nhị phân sắc nét nhất."""
    h, w = gray.shape[:2]
    s = 200.0 / w if w > 200 else 1.0
    small = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))))
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:          # đảm bảo chữ là phần trắng (thiểu số), nền đen
        bw = 255 - bw
    c = (bw.shape[1] / 2, bw.shape[0] / 2)
    best_a, best_v = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + 1e-6, step):
        M = cv2.getRotationMatrix2D(c, float(a), 1.0)
        r = cv2.warpAffine(bw, M, (bw.shape[1], bw.shape[0]))
        v = float(np.var(r.sum(axis=1)))
        if v > best_v:
            best_a, best_v = float(a), v
    return best_a


def find_line_split(gray) -> int:
    """Hàng nằm giữa 2 dòng chữ: điểm thấp nhất của profile hàng trong khoảng 35-65% chiều cao."""
    h, w = gray.shape[:2]
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:
        bw = 255 - bw
    prof = bw[:, int(w * 0.1):int(w * 0.9)].sum(axis=1).astype(float)   # bỏ viền trái/phải
    k = max(3, h // 20)
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    lo, hi = int(h * 0.35), max(int(h * 0.35) + 1, int(h * 0.65))
    return lo + int(np.argmin(prof[lo:hi]))


def deskew(gray, max_angle: float = 10.0):
    a = estimate_skew(gray, max_angle)
    if abs(a) < 0.5:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess(bgr):
    """BGR crop -> ảnh xám đã phóng to, khử nhiễu, tăng tương phản, chỉnh nghiêng."""
    img = upscale(bgr, 96)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return deskew(gray)