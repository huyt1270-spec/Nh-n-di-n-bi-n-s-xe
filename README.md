# Nhận diện biển số xe Việt Nam — ứng dụng trong giao thông công cộng thông minh (ITS)

Quy trình: **Ảnh/Video → YOLO phát hiện biển số → cắt vùng biển → tiền xử lý (xám, khử nhiễu, CLAHE, chỉnh nghiêng)
→ EasyOCR → sửa nhầm ký tự + kiểm tra định dạng (regex, mã tỉnh) → hiển thị / lưu SQLite.**
Dùng mô hình pre-trained, không huấn luyện lại.

## 1. Cài đặt (Python 3.10–3.12 khuyến nghị)

> ⚠️ Đặt project ở đường dẫn **không dấu** (vd `C:\alpr_vn`) để OpenCV không lỗi khi mở video trên Windows.

```bash
python -m venv venv
venv\Scripts\activate          # Windows   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python download_model.py       # tải YOLO biển số VN về models/best.pt
streamlit run app.py
```
Lần đầu chạy, EasyOCR tự tải mô hình nhận dạng (cần Internet, vài chục MB).
Nếu `download_model.py` lỗi: tải tay file `.pt` từ trang Hugging Face
[neitpm/vn-plate-detection-benchmark](https://huggingface.co/neitpm/vn-plate-detection-benchmark) (thư mục `weights/`)
rồi đặt tên `models/best.pt`. Chưa có file này thì app vẫn chạy nhưng ở **chế độ dự phòng OpenCV** (kém chính xác).

## 2. Cấu trúc & đối chiếu dàn ý Chương 4

| Dàn ý | File |
|---|---|
| 4.1 Công nghệ, môi trường | `requirements.txt`, `alpr/config.py` |
| 4.2 Module phát hiện biển số | `alpr/detector.py` (YOLO, ngưỡng tin cậy, cắt vùng biển) |
| 4.3 Tiền xử lý, OCR, hậu xử lý | `alpr/preprocess.py`, `alpr/ocr.py` (xử lý biển 2 dòng), `alpr/plate_rules.py` (sửa O/0, 8/B, regex, mã tỉnh) |
| 4.4 Giao diện, bỏ phiếu video | `app.py`, `alpr/pipeline.py` (`process_video`) |
| 4.5 Lưu trữ & tra cứu | `alpr/database.py` (SQLite), tab **Lịch sử** trong `app.py` |
| Chương 5 Kiểm thử, đánh giá | `evaluate.py`, `tests/test_core.py` |

## 3. Bàn giao cho người làm Chương 5

1. Bỏ ảnh thử nghiệm vào `data/samples/` và tạo `data/samples/labels.csv`:
   ```
   filename,plate,group
   xe1.jpg,51F12345,ro_net
   xe2.jpg,59C112345,nghieng
   ```
   (`group` tùy chọn: ro_net, nghieng, thieu_sang, loa_sang, bien_ban… khớp mục 5.1)
2. `python evaluate.py` → in bảng **tỉ lệ phát hiện / đúng cả biển / đúng từng ký tự / ms mỗi ảnh** theo từng nhóm
   và xuất `data/results/eval_detail.csv` (từng ảnh đúng/sai, dùng cho mục 5.5).
3. Tốc độ video (FPS) hiển thị ở tab **Video**. Ảnh kết quả có khung + biển số tự lưu ở `data/results/`
   (dùng làm hình trong báo cáo); ảnh biển đã cắt ở `data/crops/`.

## 4. Đưa lên GitHub

```bash
git init
git add .
git commit -m "ALPR VN: YOLO + EasyOCR + Streamlit"
git branch -M main
git remote add origin https://github.com/<tên-bạn>/<tên-repo>.git   # tạo repo trống trên github.com trước
git push -u origin main
```
`.gitignore` đã loại trọng số, CSDL và ảnh kết quả; thành viên khác chỉ cần chạy `python download_model.py`.

## 5. Hạn chế đã biết (nên đưa vào mục 5.5)

- Mô hình phát hiện là bản pre-trained của bên thứ ba; **ghi nguồn và giấy phép** trong báo cáo.
- Nếu phát hiện kém, thử đổi `imgsz` trong thanh bên (tác giả mô hình ví dụ chạy ở 224) hoặc dùng `--which server`.
- Danh sách mã tỉnh trong `plate_rules.py` chỉ gây **cảnh báo**, không loại biển; biển đặc biệt (quân đội, ngoại giao…) sẽ bị đánh dấu "chưa đúng định dạng".
- Gộp biển khi bỏ phiếu video: hai xe có biển chỉ khác 1 ký tự sẽ bị gộp chung.
- Chưa theo dõi (tracking) từng xe; số phiếu là theo từng biển số, không theo từng xe.

## 6. Kiểm thử phần lõi (không cần GPU/YOLO/EasyOCR)

```bash
python tests/test_core.py
```
