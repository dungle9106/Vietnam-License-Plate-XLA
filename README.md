# Nhận diện biển số xe Việt Nam 

Pipeline: **ảnh / video → YOLOv8s phát hiện biển số → cắt vùng biển (từ ảnh gốc độ phân giải cao) → tiền xử lý OpenCV + nắn thẳng → PaddleOCR đọc ký tự → hậu xử lý theo định dạng biển VN → Web / ứng dụng desktop / Excel**



## Tính năng

- **Đọc các loại biển số Việt Nam**: ô tô 1 dòng & 2 dòng, xe máy, xe máy dưới 50 cm³, xe máy điện (MĐ),
  seri đặc biệt (LD, KT, DA, TĐ, HC), biển ngoại giao / nước ngoài (NG, QT, CV, NN — dạng mới `80-NG-123.45`
  và dạng cũ `41-291-NN-01`), biển quân đội (`BH-52-62`). Tự nhận màu nền, loại phương tiện, tỉnh/thành.
- **Ảnh, thư mục ảnh, video, webcam**; đọc nhiều biển trong 1 ảnh (mỗi ảnh ra một mảng biển số).
- **Video / webcam**: theo dõi từng xe qua các khung hình (dự đoán chuyển động), bỏ phiếu chuỗi biển số
  qua nhiều khung nên kết quả ổn định dần; gộp các mã theo dõi trùng biển.
- **Xuất Excel** (3 sheet: từng biển kèm ảnh vùng biển, theo ảnh, thống kê), JSON, CSV.
- **Trang gán nhãn** (`/label`) để tạo bộ đáp án đo độ chính xác và tinh chỉnh PaddleOCR.

## Cấu trúc thư mục

```
lpr/                       Thư viện lõi
  detector.py              Giai đoạn 1: YOLOv8s phát hiện biển (ảnh tĩnh 960 px, video 640 px)
  paddle_ocr.py            Giai đoạn 2: PaddleOCR đọc ký tự - 2 cách đọc rồi chọn kết quả tốt hơn:
                             (1) tách dòng bằng code + nhận dạng từng dòng, (2) mô hình phát hiện dòng chữ
  ocr.py                   Dùng chung: tiền xử lý (đảo cực, CLAHE, lọc nhiễu, Otsu), tách dòng,
                           cắt sát ký tự, hậu xử lý kết quả, tạo bộ đọc (make_ocr)
  rectify.py               Nắn thẳng biển chụp xiên (tìm tứ giác nền biển -> biến đổi phối cảnh)
  plate_format.py          Sửa nhầm chữ/số theo vị trí, kiểm tra định dạng biển VN, định dạng hiển thị
  plate_type.py            Màu nền, loại biển, tỉnh/thành theo mã số
  quality.py               Đánh giá ảnh biển nhỏ / mờ -> cảnh báo kết quả có thể sai
  pipeline.py              Ghép các giai đoạn, xử lý video (theo dõi + bỏ phiếu), vẽ kết quả
  tracker.py               Theo dõi biển qua các khung hình (IoU + dự đoán chuyển động)
  batch.py                 Đọc cả thư mục ảnh, xuất Excel / JSON
  visualize.py             Hình Matplotlib cho báo cáo: kết quả phát hiện, các bước tiền xử lý,
                           thống kê dữ liệu, phân tích lỗi OCR
  metrics.py, io.py        IoU, Levenshtein; đọc/ghi ảnh đường dẫn có dấu tiếng Việt
web/
  server.py                Web app (Flask) + REST API
  templates/index.html     Giao diện nhận diện (trái: ảnh, phải: thông tin chi tiết)
  templates/label.html     Trang gán nhãn biển số
app.py                     Ứng dụng desktop (Tkinter)
predict.py                 Chạy bằng dòng lệnh (ảnh, thư mục -> Excel/JSON, video, webcam)
scripts/
  prepare_dataset.py       Gộp dữ liệu YOLO, khử trùng lặp, chia train/val/test 70/20/10
  train.py                 Huấn luyện YOLOv8 phát hiện biển -> weights/best.pt
  eval_detector.py         Precision, Recall, mAP của YOLO
  export_ocr_labels.py     Cắt biển ra trang gán nhãn (từ tập dữ liệu, thư mục ảnh hoặc video)
  eval_ocr.py              Độ chính xác đọc ký tự trên tập test đã gán nhãn
  eval_e2e.py              Đánh giá toàn bộ quy trình (phát hiện + đọc)
  export_paddle_rec.py     Xuất dữ liệu tinh chỉnh PaddleOCR (ảnh từng dòng chữ + nhãn)
  visualize.py             Xuất hình cho báo cáo vào runs/figures/
notebooks/
  train_colab.ipynb        Huấn luyện YOLOv8s trên Colab (GPU)
  finetune_paddleocr_colab.ipynb  Tinh chỉnh PaddleOCR trên Colab (GPU) - tuỳ chọn
tests/                     Kiểm thử (dữ liệu giả lập, không cần tập dữ liệu)
weights/best.pt            Mô hình YOLOv8s đã huấn luyện
weights/paddle_rec/        (tuỳ chọn) mô hình PaddleOCR đã tinh chỉnh - có thì tự dùng
data/lp_full/              Dữ liệu YOLO đã chia train/val/test (nguồn Kaggle tanhphp/vietnamese-license-plates)
data/ocr_labels/           Biển số đã cắt + đáp án gán nhãn tay (labels.csv)
Bo_anh_test_nhan_dien_bien_so/   Ảnh thử nghiệm
```

## 1. Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # PyTorch bản CPU
pip install -r requirements.txt
pip install --no-deps opencv-contrib-python==5.0.0.93
```

- `opencv-contrib-python` phải **cùng phiên bản** với `opencv-python` và cài bằng `--no-deps`: PaddleOCR đòi gói
  này và nếu để pip tự chọn sẽ kéo bản 4.10 về ghi đè OpenCV 5.0.
- Lần đầu chạy, PaddleOCR tự tải mô hình (~100 MB) về `C:\Users\<tên>\.paddlex\official_models`.
- Windows: nạp PaddlePaddle trước PyTorch sẽ lỗi DLL (`torch\lib\shm.dll`); thư viện `lpr` đã tự nạp PyTorch trước.

## 2. Chạy

**Web app** (giao diện chính):

```bash
python web/server.py                  # mở http://127.0.0.1:5000
python web/server.py --host 0.0.0.0   # cho máy khác cùng mạng LAN truy cập (webcam chỉ chạy qua 127.0.0.1)
```

- Kéo thả ảnh / thư mục / video vào trang, hoặc dùng các nút *Chọn ảnh*, *Thư mục*, *Video*, *Webcam*.
- Bên trái là ảnh kết quả (bấm để phóng to, ← → để chuyển ảnh), bên phải là thông tin từng biển;
  rê chuột lên thẻ biển số để tô sáng vùng biển trên ảnh.
- *Xuất Excel*, *JSON*, *Sao chép*. Trang **Gán nhãn dữ liệu** ở `/label`.

**Ứng dụng desktop**: `python app.py`

**Dòng lệnh**:

```bash
python predict.py --source anh.jpg
python predict.py --source thu_muc_anh/ --excel ket_qua.xlsx
python predict.py --source thu_muc_anh/ --single --json ket_qua.json   # 1 biển / ảnh
python predict.py --source video.mp4 --show
python predict.py --source 0 --show                                   # webcam
```

**REST API**:

```bash
curl -F files=@anh1.jpg -F multi=1 http://127.0.0.1:5000/api/recognize
# -> {"images": [{"anh": "anh1.jpg", "so_bien": 1, "bien_so": ["51F-123.45"],
#                 "chi_tiet": [{"bien_so", "loai", "mau", "tinh", "hop_le", "conf_ocr", "chat_luong", ...}]}]}
```

## 3. Huấn luyện YOLOv8s (trên Colab, GPU)

Mở [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb), chọn *Thời gian chạy → T4 GPU*, chạy lần lượt:
tải dữ liệu Kaggle [tanhphp/vietnamese-license-plates](https://www.kaggle.com/datasets/tanhphp/vietnamese-license-plates)
→ `prepare_dataset.py` → `train.py` → `eval_detector.py`. Tải `best.pt` về đặt vào `weights/`.

```bash
python scripts/prepare_dataset.py --src <thư mục dữ liệu Kaggle> --out data/lp_full
python scripts/train.py --data data/lp_full/data.yaml --model yolov8s.pt --epochs 100 --device 0
python scripts/eval_detector.py --weights weights/best.pt --data data/lp_full/data.yaml
```

Chia dữ liệu dùng hạt giống cố định (`--seed 42`) nên chia lại luôn ra đúng tập test cũ.

## 4. Gán nhãn và đo độ chính xác đọc ký tự

```bash
python scripts/export_ocr_labels.py --split test                      # biển tập test YOLO
python scripts/export_ocr_labels.py --data-dir data/lp_full --split train
python scripts/export_ocr_labels.py --video video.mp4 --every 0.5     # biển từ video
python scripts/export_ocr_labels.py --images thu_muc_anh/             # biển từ thư mục ảnh
```

Mở `http://127.0.0.1:5000/label`: PaddleOCR điền sẵn gợi ý; đúng thì Enter, sai thì sửa rồi Enter,
không đọc được thì Esc. Chạy lại lệnh trên không ghi đè nhãn đã sửa.

```bash
python scripts/eval_ocr.py                        # PaddleOCR bản chính xác, CHỈ tập test
python scripts/eval_ocr.py --ocr paddleocr_fast   # bản nhanh (video)
python scripts/eval_e2e.py                        # phát hiện + đọc trên ảnh tập test
```

Biển cắt từ tập train/val chỉ dùng để tinh chỉnh, không bao giờ tính vào độ chính xác.
`eval_ocr.py` in cả **raw** (PaddleOCR thuần) và **corrected** (sau hậu xử lý theo định dạng biển VN);
các biển đọc sai ghi vào `runs/ocr_errors.csv`.

## 5. Tinh chỉnh PaddleOCR (tuỳ chọn, cần GPU)

```bash
python scripts/export_paddle_rec.py --zip    # -> data/paddle_rec.zip (chỉ biển train/val đã gán nhãn)
```

Tải `data/paddle_rec.zip` lên thư mục gốc Google Drive, chạy
[notebooks/finetune_paddleocr_colab.ipynb](notebooks/finetune_paddleocr_colab.ipynb) trên Colab (T4 GPU),
giải nén mô hình tải về vào `weights/paddle_rec/` rồi đo lại bằng `scripts/eval_ocr.py`.
Notebook chưa được chạy trọn vẹn trên Colab (Colab chưa cấp được GPU lúc thử).

## 6. Hình minh hoạ cho báo cáo

```bash
python scripts/visualize.py steps --image anh.jpg          # kết quả YOLOv8s + các bước tiền xử lý OCR
python scripts/visualize.py dataset --data-dir data/lp_full
python scripts/visualize.py ocr-errors                    # loại lỗi + cặp ký tự hay nhầm (sau eval_ocr.py)
```

Hình lưu trong `runs/figures/`; biểu đồ huấn luyện YOLO do Ultralytics sinh trong `runs/lp_yolov8/`.

## 7. Kiểm thử

```bash
python -m unittest discover tests                          # nhanh, không cần dữ liệu / mô hình
set LPR_TEST_OCR=1 && python -m unittest discover tests    # thêm test đọc biển bằng PaddleOCR
```
