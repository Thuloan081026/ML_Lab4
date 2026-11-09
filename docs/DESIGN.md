## LAB 4: DỰ ĐOÁN GIÁ NHÀ BOSTON (BOSTON HOUSING PRICE PREDICTION)

---

## 1. WHAT (Dự án này là gì? Giải quyết vấn đề gì?)
- **Mục tiêu bài toán:** Xây dựng hệ thống học máy hồi quy đa biến (*Multi-variable Regression*) kết hợp **Mạng Nơ-ron Học sâu (Deep Learning Multi-Layer Perceptron - MLP)** để dự đoán giá trị trung bình của một ngôi nhà tại khu vực Boston (`MEDV`, đơn vị: nghìn USD) từ 13 thuộc tính cấu trúc (số phòng, tuổi nhà), nhân khẩu học (tỷ lệ thu nhập thấp `LSTAT`, tỷ lệ da màu `B`) và môi trường xã hội (tỷ lệ tội phạm `CRIM`, nồng độ khí thải `NOX`, thuế bất động sản `TAX`).
- **Yêu cầu kỹ thuật cốt lõi:**
  - Tuyệt đối tuân thủ nguyên tắc **Zero Data Leakage**: Phân chia tập dữ liệu 70% Train, 15% Validation, 15% Test ngay từ đầu. Mọi thông số thống kê (Mean, Std, Skewness, Max) đều chỉ được học từ tập Train.
  - Xử lý các phân phối lệch nặng và ngoại lai bằng phép biến đổi phi tuyến **Logarithm (`np.log1p`)** và **Reflected Logarithm**.
  - Xây dựng kiến trúc mạng học sâu **PyTorch MLP** (`13 -> 64 -> 32 -> 1`, ReLU, Adam Optimizer, MSE Loss, Early Stopping).
  - Tích hợp cơ chế **Pure NumPy Fallback**: Cho phép giải nén trọng số từ checkpoint `.pth` bằng `zipfile` và suy luận bằng ma trận NumPy thuần túy mà không bắt buộc máy chủ phải cài đặt PyTorch.
  - So sánh đối chuẩn toàn diện với 3 mô hình học máy truyền thống: Linear Regression (OLS), Decision Tree Regressor và Random Forest Regressor (100 cây Bagging).
  - Đóng gói triển khai qua **Web REST API (Flask)** và công cụ dòng lệnh CLI.

---

## 2. WHERE (Các thành phần nằm ở đâu trong dự án?)
Dự án được phân chia thư mục rõ ràng theo từng nhiệm vụ:

| Thành phần | File / Thư mục | Nhiệm vụ chính |
|---|---|---|
| **Dữ liệu thô** | [`data/raw/Boston-house-price-data.csv`](../data/raw/Boston-house-price-data.csv) | Chứa 506 bản ghi gốc ban đầu về thị trường bất động sản Boston. |
| **Dữ liệu đã xử lý** | [`data/processed/`](../data/processed/) | Lưu dữ liệu sạch sau khi chia tập và biến đổi log: `Boston-train-clean.csv`, `val-clean.csv`, `test-clean.csv`. |
| **Ma trận đặc trưng số** | [`data/features/`](../data/features/) | Lưu các mảng NumPy đã chuẩn hóa: `X_train_scaled.npy`, `X_val_scaled.npy`, `X_test_scaled.npy`, `y_*.npy`. |
| **Tham số tiền xử lý** | [`models/preprocessing_meta.json`](../models/preprocessing_meta.json) | Lưu danh sách đặc trưng, cột biến đổi log, vector Mean và Std của tập Train. |
| **Mô hình Deep Learning** | [`models/best_mlp_boston.pth`](../models/best_mlp_boston.pth) | Checkpoint trọng số mạng nơ-ron PyTorch tốt nhất đạt được qua Early Stopping. |
| **Mô hình Machine Learning** | [`models/rf_model.pkl`](../models/rf_model.pkl), `linear_model.pkl`, `tree_model.pkl` | Các mô hình Random Forest (quán quân ML), Hồi quy tuyến tính và Cây quyết định. |
| **Script chạy chính** | [`train.py`](../train.py) | Huấn luyện tự động cả PyTorch MLP và 3 mô hình ML cổ điển, tự xuất báo cáo Markdown. |
| **Kiểm thử & Chạy thử** | [`scripts/eval.py`](../scripts/eval.py), [`scripts/demo.py`](../scripts/demo.py) | Đánh giá độc lập trên tập Test và chạy thử nghiệm 3 kịch bản nhà ở (Biệt thự, Đô thị, Bình dân). |
| **Sổ tay nghiên cứu** | [`notebooks/01_boston_housing_mlp.ipynb`](../notebooks/01_boston_housing_mlp.ipynb) | Jupyter Notebook thực hiện toàn bộ quy trình EDA, xử lý phân phối lệch và thực nghiệm Deep Learning. |
| **Ứng dụng API** | [`app.py`](../app.py), [`predict.py`](../predict.py) | Server Flask Web API và CLI tiếp nhận 13 thông số bất động sản và trả về giá nhà dự báo. |
| **Báo cáo & Biểu đồ** | [`reports/BAO_CAO_LAB4.md`](../reports/BAO_CAO_LAB4.md), [`reports/evaluation_results.md`](../reports/evaluation_results.md) | Báo cáo chi tiết đồ án và bảng xếp hạng hiệu năng mô hình trên Test Set. |
| **Tài liệu học thuật** | [`docs/ALGORITHMS.md`](ALGORITHMS.md), [`docs/lab4_explanation.md`](lab4_explanation.md) | Tài liệu cơ sở toán học chi tiết và hướng dẫn giải thích tường tận từng dòng code. |

---

## 3. WHEN (Quy trình thực hiện theo thứ tự nào?)
Quy trình thực hiện dự án diễn ra tuần tự từng bước nghiêm ngặt để **chống rò rỉ dữ liệu (Zero Data Leakage)**:

```text
[Bước 1: Nạp và kiểm tra dữ liệu gốc]
  └─ Đọc file Boston-house-price-data.csv (506 dòng, 14 cột), lọc bỏ ô trống hoặc giá trị null.
       │
[Bước 2: Phân chia tập dữ liệu 70/15/15 — THỰC HIỆN ĐẦU TIÊN]
  └─ Chia thành 3 tập độc lập: Train (354 mẫu), Validation (76 mẫu), Test (76 mẫu).
       │
[Bước 3: Phát hiện độ lệch Skewness & Biến đổi Phi tuyến]
  └─ Chỉ tính độ lệch trên tập Train: phát hiện các cột lệch nặng (|skew| > 1.0) như CRIM, ZN, B, LSTAT.
  └─ Áp dụng phép biến đổi Logarit cơ số tự nhiên np.log1p đồng bộ lên cả 3 tập Train/Val/Test.
       │
[Bước 4: Chuẩn hóa phân phối Z-Score]
  └─ Tính Mean và Std CHỈ TRÊN TẬP TRAIN, lưu cấu hình vào preprocessing_meta.json.
  └─ Áp dụng công thức (X - mean) / std cho cả 3 tập để bảo toàn tính khách quan của Test Set.
       │
[Bước 5: Huấn luyện Mạng Nơ-ron PyTorch MLP & Early Stopping]
  └─ Mạng truyền thẳng (13 -> 64 -> 32 -> 1) với hàm kích hoạt ReLU và bộ tối ưu hóa Adam (lr=0.001).
  └─ Theo dõi hàm mất mát MSE trên tập Validation; tự động ngắt sớm (patience=20 epochs) để chống Overfitting.
       │
[Bước 6: Huấn luyện các mô hình Baseline cổ điển]
  └─ Huấn luyện Random Forest (100 cây), Linear Regression và Decision Tree trên cùng tập Train.
       │
[Bước 7: Đánh giá độc lập trên tập Test (Final Evaluation)]
  └─ Đo lường 4 chỉ số MSE, MAE, RMSE, R2 trên 76 căn nhà chưa từng thấy của tập Test.
  └─ Kết quả: PyTorch MLP đạt R2 = 0.9005 (Quán quân Deep Learning), Random Forest đạt R2 = 0.8964.
       │
[Bước 8: Lưu trữ Model & Triển khai Web REST API]
  └─ Lưu file trọng số best_mlp_boston.pth và khởi chạy app.py (port 5003) phục vụ dự đoán.
```

---

## 4. WHY (Tại sao lại làm như vậy?)
Lý do đằng sau các quyết định kỹ thuật cốt lõi trong Lab 4:

1. **Tại sao phải chia tập Train/Val/Test TRƯỚC KHI xử lý Skewness và Chuẩn hóa?**
   - Nếu chuẩn hóa hoặc tính toán độ lệch trên toàn bộ bảng dữ liệu trước khi chia, các thông tin thống kê của tập Test (như giá trị trung bình, độ lệch chuẩn, giá trị lớn nhất) sẽ bị rò rỉ vào tập Train (**Data Leakage**). Điều này khiến mô hình "thấy trước tương lai" và cho ra điểm số kiểm thử cao giả tạo.
2. **Tại sao lại dùng biến đổi Logarithm (`np.log1p`) thay vì xóa bỏ Outliers?**
   - Biến tỉ lệ tội phạm `CRIM` có những thị trấn lên tới 80-90 (trong khi trung bình chỉ ~3.6). Nếu dùng phương pháp cắt bỏ ngoại lai (Clipping hoặc Drop), ta sẽ làm mất đi những mẫu dữ liệu thực tế cực kỳ quý giá về các khu vực an ninh phức tạp. Hàm $\ln(1 + x)$ giúp co ngắn khoảng cách khổng lồ này một cách mềm mại, đưa phân phối nhọn hoắt về gần với phân phối chuẩn Gauss hình chuông.
3. **Tại sao chọn hàm kích hoạt ReLU cho mạng MLP?**
   - ReLU ($f(z) = \max(0, z)$) có đạo hàm bằng 1 khi $z > 0$, giúp triệt tiêu hoàn toàn hiện tượng biến mất đạo hàm (Vanishing Gradient) thường gặp ở Sigmoid hay Tanh. Nhờ đó, các tầng ẩn học được nhanh và có khả năng bẻ cong không gian tuyến tính để nắm bắt quy luật định giá phi tuyến phức tạp của bất động sản.
4. **Tại sao cần cơ chế Dừng sớm (Early Stopping với patience = 20)?**
   - Mạng nơ-ron sâu có xu hướng học quá mức (Overfitting) thuộc lòng các chi tiết nhiễu của dữ liệu Train sau hàng trăm epoch. Early Stopping đóng vai trò như một trọng tài: liên tục chấm điểm trên bài thi thử Validation, nếu sau 20 vòng liên tiếp điểm thi không tiến bộ, hệ thống sẽ ngắt ngay lập tức để lấy lại bộ trọng số tại thời điểm phong độ cao nhất.
5. **Tại sao lại xây dựng thêm cơ chế Pure NumPy Fallback cho suy luận?**
   - File `.pth` của PyTorch thực chất là một file nén định dạng Zip chứa các mảng tensor nhị phân. Khi triển khai lên môi trường máy chủ sản xuất (Production Server), việc cài đặt thư viện PyTorch (nặng hàng trăm Megabytes, cần CUDA/C++) là rất cồng kềnh. Cơ chế đọc nhị phân bằng `zipfile` và tính toán bằng phép nhân ma trận NumPy thuần túy cho phép hệ thống suy luận siêu nhẹ, độc lập và chạy được trên mọi thiết bị.

---

## 5. HOW (Mô hình hoạt động và dự đoán như thế nào?)
Khi một người dùng gửi thông tin 13 thuộc tính của một ngôi nhà qua Web API hoặc CLI, quy trình tính toán diễn ra qua **5 bước**:

```text
[Dữ liệu 13 thông số: CRIM=0.03, RM=7.5 phòng, LSTAT=4.5%, TAX=220, NOX=0.4...]
       │
       ▼
1. Tiếp nhận & Kiểm tra: Đọc 13 giá trị số học từ JSON request, xác thực kiểu dữ liệu.
       │
       ▼
2. Biến đổi Skewness: Nạp cấu hình từ preprocessing_meta.json:
   - Với các cột lệch dương (CRIM, ZN, B, LSTAT): Tính x_new = ln(1 + x).
   - Với các cột lệch âm: Tính x_new = ln(1 + max_val - x).
       │
       ▼
3. Chuẩn hóa Z-Score: Áp dụng vector mean_ và std_ đã lưu từ tập Train:
   z = (x_transformed - mean_train) / std_train.
       │
       ▼
4. Lan truyền xuôi qua mạng nơ-ron MLP (Forward Pass):
   - Tầng 1: h1 = ReLU(z @ W1^T + b1)  --> Mảng 64 nơ-ron
   - Tầng 2: h2 = ReLU(h1 @ W2^T + b2) --> Mảng 32 nơ-ron
   - Tầng Ra: y_pred = h2 @ W3^T + b3  --> 1 giá trị số thực duy nhất
       │
       ▼
5. Xuất kết quả định giá:
   - Giá trị dự báo: 45.09 (nghìn USD).
   - Quy đổi tiền tệ: $45,085 USD (khoảng 1.1 tỷ VNĐ).
   - Đồng thời trả về kết quả đối chuẩn từ Random Forest ($43.56k) và Linear Regression ($36.78k).
```
