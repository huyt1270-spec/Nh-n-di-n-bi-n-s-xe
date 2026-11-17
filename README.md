# Nhận diện biển số xe Việt Nam — ứng dụng trong giao thông công cộng thông minh (ITS)

Quy trình: **Ảnh/Video → YOLO phát hiện biển số → cắt vùng biển → tiền xử lý (xám, khử nhiễu, CLAHE, chỉnh nghiêng)
→ EasyOCR → sửa nhầm ký tự + kiểm tra định dạng (regex, mã tỉnh) → hiển thị / lưu SQLite.**
Dùng mô hình pre-trained, không huấn luyện lại.
## 1. Giới thiệu

Nhận diện biển số tự động là một thành phần quan trọng của giao thông thông minh: kiểm soát xe ra vào bến bãi, thu phí không dừng, giám sát phương tiện công cộng, tra cứu lịch sử phương tiện. Dự án xây dựng một ứng dụng hoàn chỉnh từ ảnh/video đầu vào đến kết quả được chuẩn hóa và lưu trữ.

## 2. Tính năng

- **Nhận diện từ ảnh**: tải một hoặc nhiều ảnh, hiển thị khung biển số, ảnh biển đã cắt, chuỗi biển số và độ tin cậy.
- **Nhận diện từ video**: xử lý theo khung hình, **bỏ phiếu** giữa nhiều khung để chọn biển số ổn định, hiển thị tốc độ xử lý (FPS).
- **Hỗ trợ biển 1 dòng và 2 dòng** (ô tô, xe máy).
- **Hậu xử lý theo quy tắc biển số Việt Nam**: sửa nhầm ký tự theo vị trí (O/0, I/1, S/5, 8/B...), kiểm tra định dạng bằng regex và mã tỉnh, gắn cờ ⚠️ khi kết quả đáng ngờ.
- **Lưu và tra cứu lịch sử** bằng SQLite: tìm theo biển số (gõ một phần cũng được), lọc theo ngày, xuất CSV.
- **Script đánh giá** `evaluate.py`: tính tỉ lệ phát hiện, đúng cả biển, đúng từng ký tự và thời gian xử lý theo từng nhóm ảnh.

## 3. Quy trình xử lý

```
Ảnh / Video
   │
   ▼
YOLO phát hiện biển số ──► cắt vùng biển
   │
   ▼
Tiền xử lý (phóng to, xám, khử nhiễu, CLAHE, chỉnh nghiêng)
   │
   ▼
EasyOCR đọc ký tự (biển 2 dòng: cắt tại khe giữa hai dòng, đọc từng dòng)
   │
   ▼
Hậu xử lý: sửa nhầm ký tự theo vị trí → kiểm tra định dạng biển số VN
   │
   ▼
Hiển thị kết quả  ──►  Lưu SQLite (biển số, độ tin cậy, thời gian, ảnh biển)
```

## 4. Công nghệ

| Công nghệ | Vai trò |
|---|---|
| Python | Ngôn ngữ lập trình |
| YOLO (Ultralytics) | Phát hiện biển số |
| OpenCV | Xử lý ảnh và video |
| EasyOCR | Nhận dạng ký tự |
| Streamlit | Giao diện web |
| SQLite | Lưu trữ kết quả |

---

## 5. Cấu trúc thư mục

```
├── app.py                 # Giao diện Streamlit (chạy file này)
├── evaluate.py            # Đánh giá độ chính xác và tốc độ
├── download_model.py      # Tải trọng số YOLO biển số Việt Nam
├── requirements.txt       # Danh sách thư viện
├── alpr/
│   ├── config.py          # Cấu hình: đường dẫn, ngưỡng, kích thước
│   ├── detector.py        # Phát hiện biển số (YOLO, có chế độ dự phòng OpenCV)
│   ├── preprocess.py      # Tiền xử lý vùng biển, tìm khe giữa 2 dòng
│   ├── ocr.py             # EasyOCR, xử lý biển 1 và 2 dòng
│   ├── plate_rules.py     # Quy tắc biển số VN: chuẩn hóa, sửa ký tự, regex, mã tỉnh
│   ├── pipeline.py        # Ghép các bước, xử lý ảnh/video, bỏ phiếu
│   └── database.py        # SQLite: lưu và tra cứu
├── tests/test_core.py     # Kiểm thử phần lõi (không cần GPU/YOLO/EasyOCR)
├── models/                # Chứa best.pt (tải bằng download_model.py)
└── data/
    ├── samples/           # Ảnh thử nghiệm và labels.csv
    ├── crops/             # Ảnh biển đã cắt (tự tạo khi chạy)
    ├── results/           # Ảnh kết quả có khung (tự tạo khi chạy)
    └── plates.db          # Cơ sở dữ liệu SQLite (tự tạo khi chạy)
```

---

## 6. Cài đặt (Python 3.10–3.12 khuyến nghị)

```bash
python -m venv venv
venv\Scripts\activate          # Windows   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python download_model.py       # tải YOLO biển số VN về models/best.pt
streamlit run app.py
```
