"""Kiểm thử phần lõi (không cần YOLO/EasyOCR/GPU). Chạy: python tests/test_core.py  (hoặc pytest)"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpr.database import PlateDB
from alpr.detector import detect_contours
from alpr.ocr import PlateOCR
from alpr.pipeline import ALPRPipeline
from alpr.plate_rules import analyze, merge_similar
from alpr.preprocess import estimate_skew, preprocess


def test_rules_valid_formats():
    assert analyze("51F-123.45")["text"] == "51F-123.45"        # ô tô
    assert analyze("59-C1 123.45")["text"] == "59-C1 123.45"    # xe máy
    assert analyze("30A-1234")["valid"]                         # biển cũ 4 số
    assert analyze("29AB12345")["text"] == "29AB-123.45"        # seri 2 chữ


def test_rules_fix_confusions():
    assert analyze("5IF-I23.45")["text"] == "51F-123.45"        # I->1
    assert analyze("S1F12345")["clean"] == "51F12345"           # S->5 ở đầu
    assert analyze("51F1234S")["clean"] == "51F12345"           # S->5 ở cuối
    assert analyze("510-12345")["text"] == "51D-123.45"         # 0->D ở vị trí seri


def test_rules_invalid_and_noise():
    assert not analyze("ABC")["valid"]
    assert not analyze("")["valid"]
    assert analyze("X51F12345")["text"] == "51F-123.45"         # dư ký tự rác đầu
    assert analyze("51F12345")["province_ok"]
    r = analyze("13A12345")                                      # đúng cấu trúc, mã tỉnh lạ
    assert r["valid"] and not r["province_ok"]


def test_merge_similar():
    m = merge_similar({"51F-123.45": 8, "51F-123.46": 1, "30A-999.99": 3})
    assert m["51F-123.46"] == "51F-123.45" and m["30A-999.99"] == "30A-999.99"


def test_database():
    with tempfile.TemporaryDirectory() as d:
        db = PlateDB(Path(d) / "t.db")
        db.add("51F-123.45", ocr_conf=0.9)
        db.add("30A-999.99")
        assert db.count() == 2
        assert len(db.search("51F12345")) == 1       # tìm không cần dấu
        assert len(db.search("30A")) == 1
        assert len(db.search(date_from="2000-01-01")) == 2
        assert len(db.search(date_from="2999-01-01")) == 0


def _text_img(angle=0.0):
    img = np.full((120, 480, 3), 255, np.uint8)
    cv2.putText(img, "51F 123.45", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 5)
    if angle:
        M = cv2.getRotationMatrix2D((240, 60), angle, 1.0)
        img = cv2.warpAffine(img, M, (480, 120), borderValue=(255, 255, 255))
    return img


def test_deskew_reduces_skew():
    skewed = cv2.cvtColor(_text_img(7), cv2.COLOR_BGR2GRAY)
    assert abs(estimate_skew(skewed)) >= 5
    out = preprocess(_text_img(7))
    assert out.ndim == 2 and abs(estimate_skew(out)) <= 2


def test_fallback_detector_finds_plate():
    scene = np.full((400, 600, 3), 90, np.uint8)
    cv2.rectangle(scene, (150, 250), (450, 330), (255, 255, 255), -1)       # biển 300x80
    cv2.putText(scene, "51F 123.45", (170, 310), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)
    boxes = detect_contours(scene)
    assert any(abs(b[0] - 150) < 20 and abs(b[1] - 250) < 20 for b in boxes), boxes


class FakeReader:
    """Giả lập easyocr.Reader: trả lần lượt các kết quả đã định sẵn."""
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def readtext(self, img, **kw):
        out = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return out


def box(x):
    return [[x, 0], [x + 10, 0], [x + 10, 5], [x, 5]]


class FakeDetector:
    mode, conf = "fake", 0.4

    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, img):
        return self.boxes


def test_ocr_one_line_sorted_by_x():
    ocr = PlateOCR(reader=FakeReader([[(box(100), "12345", 0.8), (box(0), "51F", 0.9)]]))
    text, conf = ocr.read(np.zeros((100, 400, 3), np.uint8))   # tỉ lệ 4:1 -> 1 dòng
    assert text == "51F12345" and 0.8 < conf < 0.9


def test_ocr_two_lines_called_twice():
    reader = FakeReader([[(box(0), "59C1", 0.9)], [(box(0), "12345", 0.8)]])
    text, _ = PlateOCR(reader=reader).read(np.zeros((140, 190, 3), np.uint8))  # 2 dòng
    assert text == "59C112345" and reader.calls == 2


def test_pipeline_image_end_to_end():
    img = np.full((300, 500, 3), 128, np.uint8)
    pipe = ALPRPipeline(FakeDetector([(100, 100, 400, 180, 0.9)]),
                        PlateOCR(reader=FakeReader([[(box(0), "51F12345", 0.95)]])))
    res = pipe.process_image(img)
    assert len(res) == 1 and res[0].text == "51F-123.45" and res[0].valid
    assert pipe.draw(img, res).shape == img.shape


def test_pipeline_video_voting():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "v.avi")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
        for _ in range(20):
            vw.write(np.full((240, 320, 3), 100, np.uint8))
        vw.release()
        outs = [[(box(0), "51F12345", 0.9)]] * 6 + [[(box(0), "51F12845", 0.5)]] + [[(box(0), "51F12345", 0.9)]] * 3
        pipe = ALPRPipeline(FakeDetector([(50, 80, 270, 160, 0.9)]),
                            PlateOCR(reader=FakeReader(outs)))
        seen = []
        s = pipe.process_video(path, stride=2, progress_cb=seen.append)
        assert s.frames_processed == 10 and s.fps > 0 and seen[-1] == 1.0
        assert len(s.plates) == 1 and s.plates[0]["text"] == "51F-123.45" and s.plates[0]["votes"] == 10


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:  # noqa
                fails += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
