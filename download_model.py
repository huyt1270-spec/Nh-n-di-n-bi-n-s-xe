"""Tải trọng số YOLO biển số xe Việt Nam (pre-trained) về models/best.pt.

    python download_model.py            # YOLOv8n biển số VN (nhẹ ~6 MB, chạy được trên CPU)
    python download_model.py --which server   # YOLOv8l (nặng hơn, chính xác hơn, ~84 MB)
    python download_model.py --which y26      # YOLO26n biển số VN (cần ultralytics bản mới)
Nếu tải lỗi: tải tay file .pt từ trang Hugging Face của repo rồi đổi tên thành models/best.pt.
"""
import argparse
import shutil

from alpr import config

SOURCES = {
    "edge": ("neitpm/vn-plate-detection-benchmark", "weights/yolov8n_edge.pt"),
    "server": ("neitpm/vn-plate-detection-benchmark", "weights/yolov8l_server.pt"),
    "y26": ("Cuonglam/vn-lp-y26", "best.pt"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=SOURCES, default="edge")
    repo, fname = SOURCES[ap.parse_args().which]
    config.ensure_dirs()
    try:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(repo_id=repo, filename=fname)
        shutil.copy(src, config.MODEL_PATH)
        print(f"Đã tải {repo}/{fname} -> {config.MODEL_PATH}")
        print("Nhớ ghi nguồn + giấy phép của mô hình (xem trang https://huggingface.co/" + repo + ") vào báo cáo.")
    except Exception as e:  # noqa
        print(f"Tải thất bại: {e}\nHãy tải tay từ https://huggingface.co/{repo} (file {fname}) "
              f"và đặt vào {config.MODEL_PATH}")
