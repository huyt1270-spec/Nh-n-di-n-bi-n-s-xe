"""Phát hiện biển số: YOLO pre-trained (chính) hoặc OpenCV contour (dự phòng khi chưa có trọng số)."""
from pathlib import Path

import cv2

from . import config


def crop_with_pad(img, box, pad: float = config.BOX_PAD):
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    return img[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def detect_contours(bgr, min_area_ratio: float = 0.002):
    """Dự phòng: tìm khung chữ nhật có tỉ lệ giống biển số bằng Canny + contour.
    Độ chính xác thấp, chỉ để chạy thử khi chưa tải được mô hình YOLO."""
    gray = cv2.bilateralFilter(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 11, 17, 17)
    edges = cv2.dilate(cv2.Canny(gray, 30, 200), None)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    boxes = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:30]:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area_ratio * W * H or w * h > 0.5 * W * H or h == 0:
            continue
        if not 1.2 <= w / h <= 6.0:
            continue
        if len(cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)) != 4:
            continue
        box = (x, y, x + w, y + h, 0.30)
        if all(_iou(box, b) < 0.5 for b in boxes):
            boxes.append(box)
    return boxes


class PlateDetector:
    def __init__(self, weights=None, conf: float = config.DET_CONF, imgsz: int = config.IMGSZ):
        self.conf, self.imgsz, self.model = conf, imgsz, None
        if weights and Path(weights).exists():
            from ultralytics import YOLO  # import muộn để test/CLI nhẹ không cần torch
            self.model = YOLO(str(weights))

    @property
    def mode(self) -> str:
        return "yolo" if self.model is not None else "opencv-fallback"

    def detect(self, bgr):
        """Trả về list (x1, y1, x2, y2, conf), sắp theo conf giảm dần."""
        if self.model is None:
            return detect_contours(bgr)
        r = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        h, w = bgr.shape[:2]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2), float(b.conf[0])))
        return sorted(out, key=lambda t: -t[4])
