"""Đánh giá hệ thống cho Chương 5 (độ chính xác phát hiện/nhận dạng, tốc độ).

Chuẩn bị:  data/samples/ chứa ảnh  +  labels.csv có các cột:
    filename,plate[,group]
    xe1.jpg,51F12345,ro_net
    xe2.jpg,59C112345,nghieng
(cột group là tùy chọn: ro_net / nghieng / thieu_sang / loa_sang / bien_ban ...)

Chạy:  python evaluate.py --images data/samples --labels data/samples/labels.csv
"""
import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

from alpr import config
from alpr.detector import PlateDetector
from alpr.ocr import PlateOCR
from alpr.pipeline import ALPRPipeline, imread
from alpr.plate_rules import normalize


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(pred: str, gt: str) -> float:
    return max(0.0, 1 - edit_distance(pred, gt) / max(1, len(gt)))


def pick_best(results):
    """Chọn biển đáng tin nhất trong ảnh: ưu tiên hợp lệ, rồi độ tin cậy OCR."""
    return max(results, key=lambda r: (r.valid, r.ocr_conf)) if results else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/samples")
    ap.add_argument("--labels", default="data/samples/labels.csv")
    ap.add_argument("--weights", default=str(config.MODEL_PATH))
    ap.add_argument("--conf", type=float, default=config.DET_CONF)
    ap.add_argument("--imgsz", type=int, default=config.IMGSZ)
    ap.add_argument("--out", default="data/results/eval_detail.csv")
    a = ap.parse_args()

    pipe = ALPRPipeline(PlateDetector(a.weights, a.conf, a.imgsz), PlateOCR())
    rows = list(csv.DictReader(open(a.labels, encoding="utf-8-sig")))
    detail, groups = [], defaultdict(list)
    for row in rows:
        img = imread(Path(a.images) / row["filename"])
        if img is None:
            print("Bỏ qua (không đọc được):", row["filename"])
            continue
        gt = normalize(row["plate"])
        t0 = time.perf_counter()
        best = pick_best(pipe.process_image(img))
        ms = (time.perf_counter() - t0) * 1000
        pred = normalize(best.text) if best else ""
        rec = dict(filename=row["filename"], group=row.get("group", "all"), truth=gt, pred=pred,
                   detected=int(best is not None), exact=int(pred == gt),
                   char_acc=round(char_accuracy(pred, gt), 4), ms=round(ms, 1))
        detail.append(rec)
        groups[rec["group"]].append(rec)
        groups["TỔNG"].append(rec)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys()) if detail else ["filename"])
        w.writeheader()
        w.writerows(detail)

    print(f"\n{'Nhóm':<14}{'Số ảnh':>8}{'Phát hiện':>11}{'Đúng cả biển':>14}{'Đúng ký tự':>12}{'ms/ảnh':>9}")
    for g, items in groups.items():
        n = len(items)
        print(f"{g:<14}{n:>8}{sum(i['detected'] for i in items) / n:>11.1%}"
              f"{sum(i['exact'] for i in items) / n:>14.1%}"
              f"{sum(i['char_acc'] for i in items) / n:>12.1%}{sum(i['ms'] for i in items) / n:>9.0f}")
    print(f"\nChi tiết từng ảnh: {a.out}")


if __name__ == "__main__":
    main()
