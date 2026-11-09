<div align="center">

# Boston Housing Price Prediction

<table>
<tr>
<td><b>Notebook chính</b></td>
<td><code>notebooks/01_boston_housing_mlp.ipynb</code></td>
</tr>
<tr>
<td><b>Dữ liệu</b></td>
<td><code>data/raw/Boston-house-price-data.csv</code></td>
</tr>
<tr>
<td><b>Bài toán</b></td>
<td>Hồi quy Giá trị Trung vị Nhà ở (Boston Housing Price Regression - Target: MEDV)</td>
</tr>
</table>
<p>
Project này xây dựng toàn bộ quy trình dự đoán Giá nhà Boston, đi từ dữ liệu thô,
phân tích tương quan EDA, kiểm tra phân phối lệch (Skewness), biến đổi Log/Reflect-Log,
chuẩn hóa Z-Score chống rò rỉ dữ liệu, huấn luyện mạng Nơ-ron PyTorch MLP kết hợp Early Stopping,
đến đối chuẩn với các mô hình Machine Learning cổ điển và triển khai Web API Flask.
</p>
</div>

## 1. Mục tiêu dự án

Định giá bất động sản là bài toán hồi quy kinh điển nhưng đầy thách thức do mối quan hệ phi tuyến tính phức tạp giữa các thuộc tính vật lý (số phòng, tuổi nhà), vị trí địa lý (khoảng cách tới trung tâm việc làm, khả năng tiếp cận đường cao tốc) và các chỉ số kinh tế xã hội (tỷ lệ tội phạm, thuế suất, tỷ lệ dân cư thu nhập thấp).

Mục tiêu chính:

- Xây dựng pipeline Học máy và Học sâu hoàn chỉnh giải quyết bài toán Hồi quy giá nhà Boston (`MEDV`).

- Thiết kế quy trình tiền xử lý chống **Rò rỉ dữ liệu (Zero Data Leakage)**: Mọi thông số thống kê (Mean, Std, Skewness, Max) chỉ được học trên tập Train (70%), sau đó áp dụng biến đổi lên tập Val (15%) và Test (15%).

- Khắc phục hiện tượng đuôi nặng và ngoại lai thông qua kỹ thuật tự động phát hiện độ lệch (|skewness| > 1.0) và biến đổi Logarit / Reflect-Log.

- Xây dựng kiến trúc mạng **Học Sâu PyTorch MLP** (`13 -> 64 -> 32 -> 1`) với hàm kích hoạt ReLU, tối ưu hóa Adam, và cơ chế ngắt sớm **Early Stopping** chống Overfitting.

- Huấn luyện và đối chuẩn (Benchmarking) hiệu năng giữa Deep Learning với các mô hình Machine Learning cổ điển (**Linear Regression, Decision Tree, Random Forest**).

- Đóng gói toàn bộ mô hình thành dịch vụ **Web REST API (Flask)** và các công cụ CLI phục vụ định giá thời gian thực.

---

## 2. Dữ liệu sử dụng & Xử lý (Data Overview)

Nguồn dữ liệu gốc: `data/raw/Boston-house-price-data.csv` gồm 506 dòng đại diện cho 506 khu vực ngoại ô tại Boston với 13 đặc trưng đầu vào và biến mục tiêu `MEDV` (đơn vị: $1,000 USD).

### 2.1. Dữ liệu gốc ban đầu

| CRIM | ZN | INDUS | CHAS | NOX | RM | AGE | DIS | RAD | TAX | PTRATIO | B | LSTAT | MEDV (Target) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00632 | 18.0 | 2.31 | 0 | 0.538 | 6.575 | 65.2 | 4.0900 | 1 | 296 | 15.3 | 396.90 | 4.98 | **24.0** |
| 0.02731 | 0.0 | 7.07 | 0 | 0.469 | 6.421 | 78.9 | 4.9671 | 2 | 242 | 17.8 | 396.90 | 9.14 | **21.6** |
| 0.02729 | 0.0 | 7.07 | 0 | 0.469 | 7.185 | 61.1 | 4.9671 | 2 | 242 | 17.8 | 392.83 | 4.03 | **34.7** |

### 2.2. Tiền xử lý & Kỹ nghệ đặc trưng (Feature Engineering & Skewness Mitigation)

1. **Phân chia tập dữ liệu nghiêm ngặt (Train/Val/Test Split 70/15/15):** Cố định `random_state=42` để đảm bảo khả năng tái lập kết quả và ngăn chặn triệt để hiện tượng nhìn trước tương lai.
2. **Xử lý phân phối lệch (Skewness Handling):**
   - Các đặc trưng lệch phải nặng như `CRIM` (tội phạm), `ZN` (đất ở lớn) được làm mượt bằng hàm `np.log1p(x)`.
   - Các đặc trưng lệch trái cực đoan được biến đổi bằng kỹ thuật phản chiếu `reflect_log` với giá trị cực đại lấy từ tập Train: $\log(1 + \max(x_{\text{train}}) - x)$.
3. **Chuẩn hóa Z-Score:** Đưa 13 đặc trưng về cùng thang đo chuẩn ($\mu=0, \sigma=1$) bằng tham số Mean và Std tính từ tập Train.

### 2.3. Bảng xếp hạng và Đối chuẩn mô hình (Model Benchmarking)

Đánh giá khách quan trên tập Kiểm thử độc lập (Test Set):

| Hạng | Mô hình | MSE | MAE | RMSE | $R^2$ Score | Đánh giá |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **PyTorch MLP** | **6.4960** | **2.0274** | **2.5487** | **0.9005** | **Vô địch**: Bắt trọn phi tuyến tính đa chiều |
| 🥈 | Random Forest | 6.7656 | 1.8944 | 2.6011 | 0.8964 | Ổn định cao, sai số tuyệt đối MAE thấp nhất |
| 🥉 | Linear Regression | 13.3456 | 2.7406 | 3.6532 | 0.7956 | Baseline tuyến tính tiêu chuẩn |
| 4 | Decision Tree | 25.3766 | 2.7237 | 5.0375 | 0.6114 | Dễ bị quá khớp nếu không cắt tỉa cành |

---

## 3. Cấu trúc thư mục hiện tại

```text
Lab4_Boston_Housing/
├── README.md                             # Hướng dẫn chi tiết dự án dự đoán giá nhà Boston Lab 4
├── train.py                              # Pipeline huấn luyện toàn bộ mô hình Deep Learning & Classical
├── predict.py                            # Script CLI suy luận dự đoán giá nhà thời gian thực
├── app.py                                # Web API Flask phục vụ dự đoán giá nhà đa mô hình
├── requirements.txt                      # Danh sách thư viện phụ thuộc (PyTorch, Scikit-Learn, Flask...)
├── .gitignore                            # Quy tắc bỏ qua cache và artifact tạm
├── data/                                 # Quản lý dữ liệu thô và dữ liệu sau xử lý
│   ├── raw/                              # Dữ liệu nhà ở gốc
│   │   └── Boston-house-price-data.csv   # 506 bản ghi nhà ở Boston (13 đặc trưng + MEDV)
│   ├── processed/                        # Dữ liệu sạch sau khi xử lý missing, duplicate và lệch
│   │   ├── Boston-clean.csv              # Toàn bộ dữ liệu sạch
│   │   ├── Boston-train-clean.csv        # Dữ liệu tập Train (70%)
│   │   ├── Boston-val-clean.csv          # Dữ liệu tập Validation (15%)
│   │   └── Boston-test-clean.csv         # Dữ liệu tập Test (15%)
│   └── features/                         # Ma trận đặc trưng số học định dạng NumPy (.npy)
│       ├── X_train_scaled.npy            # Ma trận đặc trưng Train đã chuẩn hóa Z-Score
│       ├── X_val_scaled.npy              # Ma trận đặc trưng Val đã chuẩn hóa
│       ├── X_test_scaled.npy             # Ma trận đặc trưng Test đã chuẩn hóa
│       ├── y_train.npy                   # Nhãn giá nhà tập Train
│       ├── y_val.npy                     # Nhãn giá nhà tập Val
│       └── y_test.npy                    # Nhãn giá nhà tập Test
├── models/                               # Lưu trữ trọng số mô hình đã huấn luyện
│   ├── best_mlp_boston.pth               # Trọng số tối ưu của mạng PyTorch MLP
│   ├── linear_model.pkl                  # Mô hình Linear Regression
│   ├── tree_model.pkl                    # Mô hình Decision Tree Regressor
│   ├── rf_model.pkl                      # Mô hình Random Forest Regressor
│   └── preprocessing_meta.json           # Tham số tiền xử lý phục vụ suy luận thời gian thực
├── notebooks/                            # Sổ tay nghiên cứu trực quan
│   └── 01_boston_housing_mlp.ipynb       # Thực nghiệm phân tích EDA, Skewness, Huấn luyện MLP
├── src/                                  # Thư mục mã nguồn và runs TensorBoard
│   ├── lab4.ipynb                        # Notebook thực nghiệm gốc
│   └── runs/                             # Nhật ký huấn luyện TensorBoard
│       └── boston_mlp/                   # Đồ thị hội tụ loss Train và Validation theo epoch
├── scripts/                              # Kịch bản thực thi chức năng độc lập
│   ├── data_preparation.py              # Đọc dữ liệu, chia 70/15/15, biến đổi Skew và lưu ma trận
│   ├── eval.py                           # Đánh giá độc lập 4 mô hình trên tập Test
│   └── demo.py                           # Chạy kịch bản demo 3 ngôi nhà tiêu biểu qua Test Client
├── reports/                              # Báo cáo kết quả và tài liệu trực quan
│   ├── BAO_CAO_LAB4.md                   # Báo cáo chi tiết kết quả huấn luyện nộp môn học
│   └── figures/                          # Biểu đồ phân tích và đồ thị dự báo
└── docs/                                 # Tài liệu kiến trúc và thiết kế
    ├── DESIGN.md                         # Thiết kế dự án theo cấu trúc 4W1H
    ├── PLAN.md                           # Kế hoạch phát triển chi tiết qua các giai đoạn
    ├── DATA_PIPELINE.md                  # Kiến trúc Pipeline tiền xử lý và chuẩn hóa dữ liệu
    ├── ALGORITHMS.md                     # Chi tiết công thức toán học các mô hình hồi quy
    ├── MODEL_CARD.md                     # Thẻ thông số kỹ thuật mô hình học máy
    ├── lab4_explanation.md               # Tài liệu phân tích chuyên sâu về quy trình và toán học
    └── flowcharts/                       # Sơ đồ luồng xử lý định dạng ảnh và Markdown
```

| Thành phần | Vai trò |
|---|---|
| `data/raw/Boston-house-price-data.csv` | File dữ liệu giá nhà Boston gốc. |
| `notebooks/` | Chứa sổ tay thực nghiệm trực quan hóa EDA và kiến trúc mạng MLP. |
| `train.py` | Pipeline huấn luyện tự động toàn bộ 4 mô hình Deep Learning và ML. |
| `models/` | Nơi lưu trữ trọng số PyTorch (.pth) và mô hình Scikit-Learn (.pkl). |
| `app.py` | Web API Deployment bằng Flask để phục vụ định giá nhà thời gian thực. |
| `scripts/` | Kiểm tra dữ liệu, đánh giá độc lập và chạy demo 3 kịch bản. |
| `reports/` | Báo cáo chi tiết và biểu đồ so sánh hiệu năng. |
| `docs/` | Thiết kế 4W1H, kế hoạch, pipeline, thuật toán và model card. |

---

## 4. Điểm nhấn Kỹ thuật (Technical Highlights)

Các thành phần cốt lõi được triển khai chặt chẽ theo chuẩn MLOps:

| Thành phần | Class / logic chính | Ghi chú |
|---|---|---|
| Chống Rò rỉ Dữ liệu | `Zero Data Leakage Split` | Chia tập 70/15/15 cố định seed=42; Mean, Std, Skewness chỉ học trên Train. |
| Xử lý Đuôi nặng | `np.log1p`, `reflect_log` | Tự động phát hiện \|skewness\| > 1.0 và làm mượt phân phối đặc trưng. |
| Deep Learning | `BostonMLP(nn.Module)` | Mạng nơ-ron đa tầng `13 -> 64 -> 32 -> 1` với hàm kích hoạt ReLU. |
| Tối ưu hóa & Regularization | `optim.Adam(lr=0.001)` | Tối ưu hóa gradient thích ứng với tỷ lệ học $10^{-3}$. |
| Ngắt sớm | `Early Stopping` | Giám sát `Validation Loss` với patience=20 epochs chống Overfitting. |
| Đối chuẩn Đa mô hình | `Model Benchmarking` | So sánh đa chiều (MSE, MAE, RMSE, $R^2$) giữa Deep Learning và Classical ML. |

---

## 5. Hướng dẫn Chạy Dự án (How to Run)

### A. Huấn luyện và Đánh giá tự động qua CLI

1. Cài đặt các thư viện cơ bản:
```bash
pip install -r requirements.txt
```

2. Chạy toàn bộ pipeline huấn luyện (Data Prep -> Split -> Skew Transform -> Train MLP & ML -> Benchmarking):
```bash
python train.py
```

3. Đánh giá độc lập 4 mô hình trên tập Test:
```bash
python scripts/eval.py
```

4. Chạy kịch bản demo dự đoán 3 ngôi nhà tiêu biểu:
```bash
python scripts/demo.py
```

### B. Chạy Khối lượng Nghiên cứu (Notebooks & TensorBoard)

1. Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

2. Khởi chạy Jupyter Notebook hoặc mở trên VS Code:
```bash
jupyter notebook notebooks/01_boston_housing_mlp.ipynb
```

3. Theo dõi đường cong huấn luyện Loss Real-time qua TensorBoard:
```bash
tensorboard --logdir src/runs
```

### C. Chạy Triển khai API (Model Deployment)

1. Tại thư mục gốc của dự án, mở Terminal và gõ:
```bash
python app.py
```

2. Máy chủ Flask sẽ khởi chạy tại `http://127.0.0.1:5001/`. Các Endpoints hỗ trợ bao gồm:
- **`GET /`**: Trang chủ, hiển thị thông tin chung và danh sách mô hình hỗ trợ.
- **`GET /health`**: Kiểm tra trạng thái hoạt động của dịch vụ (Health Check).
- **`POST /predict`**: Nhận dữ liệu JSON các thuộc tính nhà ở và trả về giá nhà dự đoán ($MEDV$).

3. Mở một Terminal khác (hoặc dùng Postman), gửi dữ liệu thuộc tính nhà ở vào API để lấy kết quả định giá:

*(Ví dụ dùng PowerShell trên Windows)*
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5001/predict" -Method Post -ContentType "application/json" -Body '{"model": "mlp", "CRIM": 0.02, "RM": 7.8, "LSTAT": 4.5, "TAX": 220}'
```

4. Kết quả phản hồi từ API:
```json
{
  "status": "success",
  "model_used": "mlp",
  "predicted_medv_k": 45.09,
  "predicted_price_usd": 45085.0,
  "unit": "$1,000s"
}
```
