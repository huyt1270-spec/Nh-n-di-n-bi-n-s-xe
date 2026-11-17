"""Nhận dạng ký tự bằng EasyOCR, có xử lý biển 1 dòng và 2 dòng."""
import cv2

from . import config
from .preprocess import find_line_split


class PlateOCR:
    def __init__(self, gpu: bool = False, reader=None):
        if reader is None:
            import easyocr  # import muộn
            reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self.reader = reader

    def _strip(self, img):
        """Đọc một dải ảnh (1 dòng): ghép các cụm chữ từ trái sang phải."""
        res = self.reader.readtext(img, allowlist=config.OCR_ALLOWLIST, detail=1, paragraph=False)
        res = sorted(res, key=lambda r: min(p[0] for p in r[0]))
        text = "".join(r[1] for r in res)
        conf = sum(r[2] * len(r[1]) for r in res) / max(1, len(text))
        return text, float(conf)

    def read(self, img):
        """Trả về (chuỗi thô, độ tin cậy).
        Biển 2 dòng: cắt tại khe giữa hai dòng, đọc từng dòng; chuỗi thô dạng 'dòng_trên dòng_dưới'."""
        h, w = img.shape[:2]
        if w / max(1, h) < config.TWO_LINE_ASPECT:
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cut = find_line_split(gray)
            (t1, c1), (t2, c2) = self._strip(img[:cut]), self._strip(img[cut:])
            conf = (c1 * len(t1) + c2 * len(t2)) / max(1, len(t1) + len(t2))
            return f"{t1} {t2}", conf
        return self._strip(img)