"""Giao diện Streamlit: nhận diện biển số từ ảnh/video, lưu và tra cứu lịch sử. Chạy: streamlit run app.py"""
import datetime as dt
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from alpr import config
from alpr.database import PlateDB
from alpr.detector import PlateDetector
from alpr.ocr import PlateOCR
from alpr.pipeline import ALPRPipeline, imwrite, save_crop

config.ensure_dirs()
st.set_page_config(page_title="Nhận diện biển số xe - ITS", page_icon="🚗", layout="wide")
db = PlateDB(config.DB_PATH)
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


@st.cache_resource(show_spinner="Đang nạp mô hình YOLO...")
def load_detector(weights, imgsz):
    return PlateDetector(weights, imgsz=imgsz)


@st.cache_resource(show_spinner="Đang nạp EasyOC...")
def load_ocr(gpu):
    return PlateOCR(gpu=gpu)


# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")
    weights = st.text_input("Trọng số YOLO (.pt)", str(config.MODEL_PATH))
    det_conf = st.slider("Ngưỡng tin cậy phát hiện", 0.10, 0.95, config.DET_CONF, 0.05)
    imgsz = st.selectbox("Kích thước ảnh vào YOLO (imgsz)", [224, 320, 416, 640, 960], index=3)
    gpu = st.checkbox("Dùng GPU (CUDA)", value=False)
    save_db = st.checkbox("Lưu kết quả hợp lệ vào CSDL", value=True)
    st.caption(f"CSDL hiện có **{db.count()}** bản ghi")

detector = load_detector(weights, imgsz)
detector.conf = det_conf
pipe = ALPRPipeline(detector, load_ocr(gpu))

st.title("🚗 Nhận diện biển số xe — ứng dụng trong giao thông công cộng thông minh")
if detector.mode != "yolo":
    st.warning("Chưa tìm thấy trọng số YOLO tại đường dẫn trên → đang dùng **chế độ dự phòng OpenCV** "
               "(độ chính xác thấp). Chạy `python download_model.py` rồi tải lại trang.")

tab_img, tab_vid, tab_hist = st.tabs(["📷 Ảnh", "🎬 Video", "🗂️ Lịch sử"])


def store(r, source):
    """Lưu 1 kết quả vào SQLite kèm ảnh biển đã cắt."""
    db.add(r.text, r.raw, r.ocr_conf, r.det_conf, r.valid, r.province_ok, source, save_crop(r.crop, r.text))


# ------------------------------------------------------------------ Tab ảnh
with tab_img:
    files = st.file_uploader("Chọn một hoặc nhiều ảnh", type=["jpg", "jpeg", "png", "bmp", "webp"],
                             accept_multiple_files=True)
    if files and st.button("Nhận diện", type="primary", key="run_img"):
        out = []
        for f in files:
            img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                st.error(f"Không đọc được ảnh {f.name}")
                continue
            t0 = time.perf_counter()
            res = pipe.process_image(img)
            ms = (time.perf_counter() - t0) * 1000
            ann = pipe.draw(img, res)
            if save_db:
                for r in res:
                    if r.valid:
                        store(r, f.name)
            imwrite(config.RESULT_DIR / f"{Path(f.name).stem}_result.jpg", ann)   # ảnh minh họa cho báo cáo
            out.append((f.name, ann, res, ms))
        st.session_state["img_out"] = out

    for name, ann, res, ms in st.session_state.get("img_out", []):
        st.subheader(f"{name} — {ms:.0f} ms")
        c1, c2 = st.columns([3, 2])
        c1.image(rgb(ann), use_container_width=True)
        with c2:
            if not res:
                st.info("Không phát hiện biển số.")
            for r in res:
                st.image(rgb(r.crop), width=240)
                if not r.valid:
                    badge = "⚠️ chưa đúng định dạng"
                elif r.review:
                    badge = "⚠️ đúng định dạng nhưng **nên kiểm tra lại** (seri 2 chữ hoặc mã tỉnh lạ)"
                else:
                    badge = "✅ hợp lệ"
                st.markdown(f"### `{r.text}`  \n{badge} · OCR {r.ocr_conf:.2f} · Phát hiện {r.det_conf:.2f}  \n")
        ok, buf = cv2.imencode(".jpg", ann)
        st.download_button("⬇️ Tải ảnh kết quả", buf.tobytes(), f"{Path(name).stem}_result.jpg",
                           "image/jpeg", key=f"dl_{name}")
        st.divider()

# ------------------------------------------------------------------ Tab video
with tab_vid:
    vf = st.file_uploader("Chọn video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    c1, c2, c3 = st.columns(3)
    stride = c1.slider("Xử lý mỗi N khung hình", 1, 15, 3)
    max_frames = c2.slider("Số khung hình tối đa", 60, 3000, 600, 60)
    min_votes = c3.slider("Số phiếu tối thiểu để chấp nhận", 1, 10, 2)
    if vf and st.button("Nhận diện video", type="primary", key="run_vid"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(vf.name).suffix, dir=config.ROOT / "data") as tmp:
            tmp.write(vf.getvalue())
        bar = st.progress(0.0, text="Đang xử lý video...")
        try:
            summary = pipe.process_video(tmp.name, stride, max_frames, lambda p: bar.progress(p))
            st.session_state["vid_out"] = (summary, vf.name)
            if save_db:
                for p in summary.plates:
                    if p["votes"] >= min_votes:
                        db.add(p["text"], p["text"], p["avg_conf"], 0.0, True, True,
                               f"{vf.name} ({p['votes']} phiếu)", save_crop(p["crop"], p["text"]))
        except ValueError as e:
            st.error(str(e))
        finally:
            bar.empty()
            Path(tmp.name).unlink(missing_ok=True)

    if "vid_out" in st.session_state:
        s, vname = st.session_state["vid_out"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Khung hình đã đọc", s.frames_read)
        m2.metric("Khung hình đã xử lý", s.frames_processed)
        m3.metric("Tốc độ xử lý (FPS)", f"{s.fps:.2f}")
        kept = [p for p in s.plates if p["votes"] >= min_votes]
        if not kept:
            st.info("Không có biển số nào đủ số phiếu tối thiểu. Thử giảm ngưỡng phiếu hoặc `stride`.")
        for p in kept:
            a, b = st.columns([1, 3])
            a.image(rgb(p["crop"]), width=220)
            b.markdown(f"### `{p['text']}`  \n**{p['votes']}** phiếu · độ tin cậy TB {p['avg_conf']:.2f} "
                       f"· khung hình #{p['frame']}")
            b.image(rgb(p["preview"]), width=480)

# ------------------------------------------------------------------ Tab lịch sử
with tab_hist:
    q = st.text_input("Tra cứu theo biển số (gõ một phần cũng được)")
    d1, d2 = st.columns(2)
    date_from = d1.date_input("Từ ngày", value=None)
    date_to = d2.date_input("Đến ngày", value=None)
    rows = db.search(q, date_from, date_to)
    if rows:
        df = pd.DataFrame(rows)[["id", "plate", "ocr_conf", "det_conf", "source", "created_at", "crop_path"]]
        st.dataframe(df.drop(columns=["crop_path"]), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Xuất CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           "lich_su_bien_so.csv", "text/csv")
        shown = [r for r in rows[:12] if r["crop_path"] and Path(r["crop_path"]).exists()]
        if shown:
            st.image([rgb(cv2.imdecode(np.fromfile(r["crop_path"], np.uint8), cv2.IMREAD_COLOR)) for r in shown],
                     caption=[r["plate"] for r in shown], width=160)
    else:
        st.info("Chưa có bản ghi nào khớp.")