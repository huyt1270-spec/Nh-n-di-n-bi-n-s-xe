"""Luồng xử lý: Ảnh/Video -> Phát hiện -> Cắt -> Tiền xử lý -> OCR -> Kiểm tra -> Kết quả."""
import datetime as dt
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config
from .detector import crop_with_pad
from .plate_rules import analyze, merge_similar, normalize
from .preprocess import preprocess, upscale


# ---- đọc/ghi ảnh an toàn với đường dẫn có dấu tiếng Việt (cv2.imread/imwrite hay lỗi trên Windows) ----
def imread(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img) -> bool:
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img)
    if ok:
        Path(path).write_bytes(buf.tobytes())
    return ok


def save_crop(crop, plate_text: str) -> str:
    config.ensure_dirs()
    name = f"{dt.datetime.now():%Y%m%d_%H%M%S_%f}_{normalize(plate_text) or 'unk'}.jpg"
    path = config.CROP_DIR / name
    imwrite(path, crop)
    return str(path)


@dataclass
class PlateResult:
    box: tuple
    det_conf: float
    raw: str
    text: str
    ocr_conf: float
    valid: bool
    province_ok: bool
    review: bool = False
    crop: np.ndarray = field(repr=False, default=None)


@dataclass
class VideoSummary:
    plates: list           # dict(text, votes, avg_conf, crop, frame, preview) sắp theo votes giảm dần
    frames_read: int
    frames_processed: int
    seconds: float

    @property
    def fps(self) -> float:
        return self.frames_processed / self.seconds if self.seconds else 0.0


class ALPRPipeline:
    def __init__(self, detector, ocr):
        self.detector, self.ocr = detector, ocr

    def recognize_crop(self, crop):
        """Thử OCR trên ảnh đã tiền xử lý; nếu chưa chắc thì thử thêm ảnh gốc phóng to. Chọn kết quả tốt nhất."""
        score = lambda info, conf: (info["valid"] and not info["review"], info["valid"], conf)
        best = None
        for img in (preprocess(crop), upscale(crop, 128)):
            raw, conf = self.ocr.read(img)
            info = analyze(raw)
            if best is None or score(info, conf) > score(best[2], best[1]):
                best = (raw, conf, info)
            if info["valid"] and not info["review"] and conf >= 0.6:   # chắc chắn -> khỏi thử thêm
                break
        return best

    def process_image(self, bgr):
        results = []
        for x1, y1, x2, y2, dc in self.detector.detect(bgr):
            crop = crop_with_pad(bgr, (x1, y1, x2, y2))
            if crop.size == 0:
                continue
            raw, conf, info = self.recognize_crop(crop)
            results.append(PlateResult((x1, y1, x2, y2), dc, raw, info["text"], conf,
                                       info["valid"], info["province_ok"], info["review"], crop))
        return results

    @staticmethod
    def draw(bgr, results):
        out = bgr.copy()
        scale = max(0.5, out.shape[1] / 1000)
        for r in results:
            x1, y1, x2, y2 = r.box
            color = (0, 200, 0) if r.valid and not r.review else (0, 165, 255)   # xanh: chắc, cam: cần kiểm tra
            cv2.rectangle(out, (x1, y1), (x2, y2), color, max(2, int(2 * scale)))
            label = f"{r.text} ({r.ocr_conf:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, 2)
            top = max(0, y1 - th - 10)
            cv2.rectangle(out, (x1, top), (x1 + tw + 6, top + th + 10), color, -1)
            cv2.putText(out, label, (x1 + 3, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7 * scale, (0, 0, 0), 2)
        return out

    def process_video(self, path, stride: int = 3, max_frames: int = 600, progress_cb=None):
        """Xử lý video theo từng khung hình (mỗi `stride` khung), bỏ phiếu để chọn biển ổn định."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError("Không mở được video")
        total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames, max_frames)
        new = lambda: dict(count=0, conf_sum=0.0, best=-1.0, crop=None, frame=0, preview=None)
        votes, idx, processed, t0 = {}, 0, 0, time.perf_counter()
        while idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                res = self.process_image(frame)
                processed += 1
                ann = None
                for r in res:
                    if not r.valid:
                        continue
                    v = votes.setdefault(r.text, new())
                    v["count"] += 1
                    v["conf_sum"] += r.ocr_conf
                    if r.ocr_conf > v["best"]:
                        ann = self.draw(frame, res) if ann is None else ann
                        v.update(best=r.ocr_conf, crop=r.crop, frame=idx, preview=ann)
            idx += 1
            if progress_cb and total:
                progress_cb(min(idx / total, 1.0))
        cap.release()
        secs = time.perf_counter() - t0

        mapping = merge_similar({t: v["count"] for t, v in votes.items()})
        merged = {}
        for t, v in votes.items():
            m = merged.setdefault(mapping[t], new())
            m["count"] += v["count"]
            m["conf_sum"] += v["conf_sum"]
            if v["best"] > m["best"]:
                m.update(best=v["best"], crop=v["crop"], frame=v["frame"], preview=v["preview"])
        plates = sorted(
            [dict(text=t, votes=m["count"], avg_conf=m["conf_sum"] / m["count"],
                  crop=m["crop"], frame=m["frame"], preview=m["preview"]) for t, m in merged.items()],
            key=lambda p: -p["votes"])
        return VideoSummary(plates, idx, processed, secs)