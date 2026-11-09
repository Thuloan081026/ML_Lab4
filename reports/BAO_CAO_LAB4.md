# BÁO CÁO DỰ ÁN MÔN HỌC MÁY HỌC (MACHINE LEARNING) — LAB 4

## HỒI QUY DỰ ĐOÁN GIÁ NHÀ BOSTON (BOSTON HOUSING PRICE PREDICTION)
### MẠNG HỌC SÂU PYTORCH MLP VS. CÁC MÔ HÌNH HỌC MÁY CỔ ĐIỂN CHỐNG RÒ RỈ DỮ LIỆU
**Tập dữ liệu:** `data/raw/Boston-house-price-data.csv` (Harrison & Rubinfeld, Boston Standard Metropolitan Statistical Area)  
**Mục tiêu dự đoán:** Giá trị trung vị nhà ở sở hữu tư nhân (`MEDV` - tính bằng $1,000 USD)

---

## 1. GIỚI THIỆU (INTRODUCTION)

### 1.1. Bối cảnh và Định nghĩa bài toán
Định giá bất động sản là bài toán kinh tế then chốt đối với các nhà đầu tư, tổ chức tín dụng ngân hàng, và cơ quan quản lý đô thị. Giá trị của một ngôi nhà không chỉ phụ thuộc vào các thuộc tính vật lý (số lượng phòng ngủ, diện tích xây dựng, tuổi thọ công trình) mà chịu sự chi phối mạnh mẽ bởi các yếu tố vị trí địa lý (khả năng kết nối giao thông, khoảng cách tới trung tâm thương mại) và chỉ số kinh tế xã hội (tỷ lệ tội phạm khu vực, thuế tài sản, chất lượng giáo dục thông qua tỷ lệ học sinh/giáo viên).

**Bài toán đặt ra:** Xây dựng mô hình học máy hồi quy (Regression) dự đoán giá trị trung vị của nhà ở tại các tiểu khu đô thị Boston (`MEDV`) từ 13 thuộc tính đầu vào phức tạp.

Thách thức cốt lõi của bài toán:
- Mối quan hệ giữa các biến đầu vào và giá nhà mang tính chất **phi tuyến tính sâu sắc** (Non-linear relationships).
- Nhiều đặc trưng mang phân phối **lệch nặng đuôi dài** (Heavy-tailed, Skewed distribution) và chứa nhiều giá trị ngoại lai (Outliers).
- Nguy cơ **rò rỉ dữ liệu (Data Leakage)** nếu các phép chuẩn hóa hoặc khử lệch được tính toán trên toàn bộ dữ liệu trước khi chia tập.

### 1.2. Mục tiêu bài thực hành
1. Thiết kế quy trình tiền xử lý tuân thủ nghiêm ngặt nguyên tắc **Zero Data Leakage** (mọi phép tính Mean, Std, Skewness, Max chỉ được học trên tập Train).
2. Phát hiện và xử lý phân phối lệch bằng các phép biến đổi toán học phù hợp (**Log Transform** và **Reflect-Log Transform**).
3. Xây dựng và tối ưu hóa kiến trúc mạng **Học Sâu PyTorch Multi-Layer Perceptron (MLP)** kết hợp cơ chế ngắt sớm **Early Stopping** chống hiện tượng Overfitting.
4. Xây dựng và đối chuẩn (Benchmarking) 3 mô hình học máy truyền thống đại diện:
   - **Linear Regression:** Mô hình tuyến tính giải tích bằng phương trình pháp tuyến (Normal Equation).
   - **Decision Tree Regressor:** Mô hình phi tuyến dạng cây dựa trên tiêu chí Giảm phương sai (Variance Reduction).
   - **Random Forest Regressor:** Mô hình tập hợp (Ensemble Bagging) 100 cây quyết định.
5. Theo dõi trực quan hóa quá trình giảm thiểu hàm mất mát theo epoch thời gian thực qua **TensorBoard**.
6. Đóng gói mô hình thành dịch vụ **Web REST API (Flask)** và các công cụ CLI phục vụ định giá bất động sản thời gian thực.

### 1.3. Loại bài toán học máy
Bài toán thuộc lớp **Học có giám sát (Supervised Learning)**, cụ thể là **Hồi quy liên tục (Continuous Regression)**:
- Không gian đầu vào: $X \in \mathbb{R}^{13}$ (13 đặc trưng số học liên tục và nhị phân).
- Biến mục tiêu: $y \in \mathbb{R}^+$ (Giá trị trung vị nhà ở tính bằng ngàn USD, $y \in [5.0, 50.0]$).

---

## 2. MÔ TẢ TẬP DỮ LIỆU & Ý NGHĨA 13 ĐẶC TRƯNG

### 2.1. Nguồn gốc & Quy mô dữ liệu
Tập dữ liệu gốc gồm **506 dòng** tương ứng với 506 tiểu khu điều tra dân số tại Boston. Tập dữ liệu có đầy đủ 14 cột (13 đặc trưng độc lập và 1 biến phụ thuộc `MEDV`), không có giá trị khuyết thiếu ban đầu.

### 2.2. Bảng ý nghĩa chi tiết các thuộc tính

| Tên Đặc Trưng | Kiểu Dữ Liệu | Đơn Vị Đo | Ý Nghĩa Thực Tế Trong Đô Thị Học |
| :--- | :---: | :---: | :--- |
| `CRIM` | Số thực | Tỷ lệ / đầu người | Tỷ lệ tội phạm bình quân đầu người theo từng thị trấn |
| `ZN` | Số thực | Tỷ lệ (%) | Tỷ lệ đất ở được quy hoạch cho các lô đất rộng trên 25.000 sq.ft |
| `INDUS` | Số thực | Tỷ lệ (%) | Tỷ lệ diện tích đất kinh doanh phi bán lẻ (công nghiệp) trên mỗi thị trấn |
| `CHAS` | Nhị phân | 0 hoặc 1 | Biến giả Charles River (= 1 nếu khu đất giáp bờ sông; = 0 nếu ngược lại) |
| `NOX` | Số thực | Nồng độ (phần 10 triệu) | Nồng độ khí ô nhiễm oxit nitric (đo lường chất lượng không khí) |
| `RM` | Số thực | Phòng / nhà | Số lượng phòng ở trung bình trong mỗi căn hộ |
| `AGE` | Số thực | Tỷ lệ (%) | Tỷ lệ các căn nhà do chủ sở hữu xây dựng từ trước năm 1940 (độ cũ kỹ) |
| `DIS` | Số thực | Khoảng cách có trọng số | Khoảng cách có trọng số tới 5 trung tâm việc làm chính tại Boston |
| `RAD` | Số nguyên | Chỉ số khả năng tiếp cận | Chỉ số khả năng tiếp cận các tuyến đường cao tốc vành đai |
| `TAX` | Số thực | Đô la / $10,000 | Thuế suất tài sản trên toàn bộ giá trị $10.000 USD |
| `PTRATIO` | Số thực | Tỷ số | Tỷ lệ học sinh trên giáo viên tại các trường học trong thị trấn |
| `B` | Số thực | Chỉ số biến đổi | $1000(Bk - 0.63)^2$ với $Bk$ là tỷ lệ cư dân gốc Phi theo thị trấn |
| `LSTAT` | Số thực | Tỷ lệ (%) | Tỷ lệ dân cư có vị thế kinh tế xã hội thấp (thu nhập thấp) |
| **`MEDV` (Target)** | **Số thực** | **$1,000 USD** | **Giá trị trung vị của các ngôi nhà do chủ sở hữu cư trú** |

### 2.3. Phân tích tương quan & Đa cộng tuyến (EDA Insights)
- **Tương quan dương mạnh nhất:** `RM` (Số phòng, tương quan $+0.695$) — Nhà càng nhiều phòng thì giá nhà càng tăng mạnh theo cấp số nhân.
- **Tương quan âm mạnh nhất:** `LSTAT` (Tỷ lệ người thu nhập thấp, tương quan $-0.738$) — Tỷ lệ nghèo đói trong khu vực càng cao thì giá trị bất động sản càng giảm sâu.
- **Đa cộng tuyến đáng chú ý:** `RAD` và `TAX` có hệ số tương quan tuyến tính rất cao ($+0.91$), phản ánh chính sách thuế suất tài sản cao áp dụng dọc các tuyến đường vành đai đô thị.

---

## 3. TIỀN XỬ LÝ CHỐNG RÒ RỈ DỮ LIỆU & XỬ LÝ LỆCH NẶNG (SKEWNESS MITIGATION)

### 3.1. Phân chia tập dữ liệu nghiêm ngặt (Train/Val/Test Split 70/15/15)
Để ngăn chặn hoàn toàn hiện tượng rò rỉ thông tin từ tương lai vào quá trình học, quy trình phân chia tập dữ liệu được thực hiện **trước tất cả các bước tiền xử lý**:
- Tổng dữ liệu: 506 mẫu.
- Phân chia lần 1: Tách 15% làm tập Test độc lập (`test_size=0.15`, `random_state=42`) $\rightarrow$ **Test Set = 76 mẫu**.
- Phân chia lần 2: Tách tập tạm thời thành 70% Train và 15% Val (`test_size=0.1765`, `random_state=42`) $\rightarrow$ **Train Set = 354 mẫu**, **Val Set = 76 mẫu**.

### 3.2. Tự động phát hiện phân phối lệch (Skewness Detection trên Train)
Hàm mật độ xác suất (KDE) trên tập Train cho thấy nhiều thuộc tính bị lệch đuôi nghiêm trọng:
- Thiết lập ngưỡng kiểm định độ lệch: $S_{\text{threshold}} = 1.0$.
- Phân loại và kết quả phát hiện:
  - `CRIM`: Skewness $= +5.12$ (Lệch phải cực đoan).
  - `ZN`: Skewness $= +2.34$ (Lệch phải nặng).
  - `DIS`: Skewness $= +1.02$ (Lệch phải vừa).
  - Các cột có độ lệch trái nghiêm trọng (Skewness $< -1.0$) nếu có.

### 3.3. Các phép biến đổi toán học làm mượt phân phối
1. **Biến đổi Logarit làm mượt lệch phải (`log1p`):**
   $$x_{\text{new}} = \ln(1 + x)$$
   Phép biến đổi này kéo dải giá trị cực lớn co cụm lại, đưa phân phối lệch phải về dạng chuẩn Bell-curve đối xứng, giúp gradient khi huấn luyện mạng nơ-ron không bị bùng nổ (Exploding Gradients).
2. **Biến đổi Phản chiếu Logarit cho phân phối lệch trái (`reflect_log`):**
   $$x_{\text{new}} = \ln(1 + \max(x_{\text{train}}) - x)$$
   *Điểm mấu chốt:* Giá trị $\max(x_{\text{train}})$ được tính toán **duy nhất trên tập Train**, sau đó áp dụng cố định lên tập Val và Test để loại trừ rò rỉ dữ liệu.

### 3.4. Chuẩn hóa Z-Score (Feature Standardization)
Sau khi biến đổi lệch, ma trận đặc trưng được chuẩn hóa về cùng thang đo $(\mu = 0, \sigma = 1)$:
$$X_{\text{scaled}} = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
Bộ tham số $\mu_{\text{train}}$ và $\sigma_{\text{train}}$ được lưu trữ vào file JSON [`models/preprocessing_meta.json`](file:///d:/Năm%203/Máy%20học/ML_Lab/Lab4_Boston_Housing/models/preprocessing_meta.json) để phục vụ suy luận trong tương lai.

---

## 4. THIẾT KẾ & HUẤN LUYỆN MẠNG HỌC SÂU PYTORCH MLP

### 4.1. Kiến trúc mạng nơ-ron đa tầng `BostonMLP`
Mạng được thiết kế theo cấu trúc hình tháp thu hẹp dần (Funnel Architecture):
```text
Input Layer (13 Features)
  │
  ├── Fully-Connected Dense Layer 1: Linear(13, 64)
  ├── Activation Function: ReLU(x) = max(0, x)
  │
  ├── Fully-Connected Dense Layer 2: Linear(64, 32)
  ├── Activation Function: ReLU(x) = max(0, x)
  │
  └── Output Layer: Linear(32, 1)  ──► Predicted MEDV
```
- **Tổng số tham số học được (Trainable Parameters):**
  - Layer 1: $13 \times 64 + 64 = 896$ tham số.
  - Layer 2: $64 \times 32 + 32 = 2,080$ tham số.
  - Output Layer: $32 \times 1 + 1 = 33$ tham số.
  - **Tổng cộng: 3,009 tham số weights & biases.**

### 4.2. Hàm mất mát & Thuật toán tối ưu hóa
- **Hàm mất mát Mean Squared Error Loss:**
  $$\mathcal{L}_{\text{MSE}}(y, \hat{y}) = \frac{1}{B} \sum_{i=1}^B (y_i - \hat{y}_i)^2$$
- **Thuật toán tối ưu Adam (Adaptive Moment Estimation):**
  - Tốc độ học (Learning rate): $\alpha = 0.001$.
  - Tham số động lượng (Momentum parameters): $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$.
  - Kích thước mini-batch: $B = 32$.

### 4.3. Cơ chế ngắt sớm (Early Stopping)
Để mạng nơ-ron học sâu không ghi nhớ vẹt dữ liệu (Overfitting trên tập Train có kích thước nhỏ 354 mẫu), thuật toán áp dụng cơ chế Early Stopping:
- Giám sát độ mất mát trên tập Validation: $\mathcal{L}_{\text{val}}$.
- Ngưỡng kiên nhẫn (Patience): 20 epochs.
- Cơ chế lưu checkpoint: Mỗi khi $\mathcal{L}_{\text{val}}$ đạt kỷ lục thấp mới, tự động lưu state_dict của mô hình vào file [`models/best_mlp_boston.pth`](file:///d:/Năm%203/Máy%20học/ML_Lab/Lab4_Boston_Housing/models/best_mlp_boston.pth).
- Kết quả huấn luyện: Quá trình huấn luyện dừng sớm tại epoch thứ 112 khi Validation Loss đạt cực tiểu, bảo tồn khả năng tổng quát hóa tối đa.

---

## 5. CÁC MÔ HÌNH HỌC MÁY TRUYỀN THỐNG ĐỐI CHUẨN (BENCHMARKING)

Để đánh giá khách quan giá trị gia tăng của mạng học sâu, 3 mô hình học máy truyền thống được huấn luyện trên cùng tập dữ liệu:

1. **Hồi quy tuyến tính Linear Regression:**
   - Giải phương trình pháp tuyến giải tích: $w = (X^T X)^{-1} X^T y$.
   - Sử dụng ma trận đã chuẩn hóa Z-Score.
2. **Cây quyết định Decision Tree Regressor:**
   - Thuật toán phân nhánh nhị phân đệ quy với tiêu chí Giảm phương sai (Variance Reduction):
     $$\Delta \sigma^2 = \sigma^2_{\text{parent}} - \left(\frac{N_L}{N}\sigma^2_L + \frac{N_R}{N}\sigma^2_R\right)$$
   - Khống chế `random_state=42`.
3. **Rừng ngẫu nhiên Random Forest Regressor:**
   - Tập hợp 100 cây quyết định (`n_estimators=100`, `random_state=42`).
   - Sử dụng cơ chế lấy mẫu có hoàn lại (Bootstrap Aggregation) kết hợp chọn ngẫu nhiên tập con đặc trưng (Feature Subsampling) để giảm phương sai sai số.

---

## 6. KẾT QUẢ THỰC NGHIỆM, SO SÁNH HIỆU NĂNG & THẢO LUẬN

### 6.1. Bảng xếp hạng hiệu suất chính thức trên tập Test độc lập

| Hạng | Mô hình | MSE | MAE ($1,000s) | RMSE ($1,000s) | $R^2$ Score | Đánh giá tổng quát |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **PyTorch MLP** | **6.4960** | **2.0274** | **2.5487** | **0.9005** | **Quán quân:** Hiệu năng giải thích phương sai vượt trội $\approx 90.05\%$. |
| 🥈 | **Random Forest** | 6.7656 | **1.8944** | 2.6011 | 0.8964 | Bám sát nút, đạt sai số tuyệt đối MAE thấp nhất ($1,894 USD). |
| 🥉 | **Linear Regression** | 13.3456 | 2.7406 | 3.6532 | 0.7956 | Bị giới hạn bởi giả định quan hệ tuyến tính thẳng. |
| 4 | **Decision Tree** | 25.3766 | 2.7237 | 5.0375 | 0.6114 | Sai số bình phương lớn do mặt phân cách bậc thang cục bộ. |

### 6.2. Phân tích nguyên nhân PyTorch MLP dẫn đầu
- **Khả năng xấp xỉ hàm vạn năng (Universal Approximation):** Sự kết hợp của 2 tầng ẩn (64 và 32 nơ-ron) cùng hàm phi tuyến ReLU cho phép mạng MLP mô hình hóa trọn vẹn các tương tác bậc cao giữa tỷ lệ tội phạm `CRIM`, tỷ lệ người nghèo `LSTAT`, và số lượng phòng `RM`.
- **Tác động của tiền xử lý Logarit:** Khử đuôi lệch nặng giúp mạng nơ-ron không bị sai lệch trọng số bởi các khu vực có tỷ lệ tội phạm đột biến.
- **Tại sao Linear Regression chỉ đạt $R^2 \approx 0.795$:** Mô hình tuyến tính giả định mối quan hệ giữa giá nhà và các biến là bất biến độc lập, trong khi thực tế việc thêm 1 phòng ở một khu nhà giàu có giá trị thặng dư cao hơn rất nhiều so với thêm 1 phòng ở một khu ổ chuột.

### 6.3. Kiểm chứng thực nghiệm trên 3 kịch bản nhà ở (Demo Verification)
Chạy thử nghiệm trên 3 mẫu nhà thực tế:
- **Khu Biệt Thự Cao Cấp (Nhiều phòng, ít tội phạm):**
  - Đầu vào: `CRIM=0.02`, `RM=7.8`, `LSTAT=4.5%`, `TAX=220`.
  - Giá dự đoán: **PyTorch MLP = $45.09k** | **Random Forest = $43.56k** | **Linear Reg = $36.78k**.
- **Khu Nhà Đô Thị Trung Bình:**
  - Đầu vào: `CRIM=0.25`, `RM=6.2`, `LSTAT=12.0%`, `TAX=310`.
  - Giá dự đoán: **PyTorch MLP = $17.18k** | **Random Forest = $21.11k** | **Linear Reg = $24.17k**.
- **Khu Nhà Bình Dân / Mật Độ Cao:**
  - Đầu vào: `CRIM=8.5`, `RM=5.2`, `LSTAT=28.5%`, `TAX=666`.
  - Giá dự đoán: **PyTorch MLP = $11.41k** | **Random Forest = $10.11k** | **Linear Reg = $6.47k**.

---

## 7. ĐÓNG GÓI TRIỂN KHAI & SUY LUẬN THỜI GIAN THỰC

### 7.1. Cấu trúc đóng gói mô hình
1. `models/best_mlp_boston.pth`: Trọng số tối ưu PyTorch MLP.
2. `models/preprocessing_meta.json`: Bảng ánh xạ tham số Skewness Max, Mean, và Std của tập Train.
3. `models/linear_model.pkl`, `tree_model.pkl`, `rf_model.pkl`: Các mô hình Scikit-Learn đã serialize qua Joblib.

### 7.2. Dịch vụ Web REST API (Flask Deployment)
Microservice tại [`app.py`](file:///d:/Năm%203/Máy%20học/ML_Lab/Lab4_Boston_Housing/app.py) cung cấp:
- **`GET /health`**: Health Check service.
- **`POST /predict`**: Hỗ trợ chuyển đổi đa mô hình linh hoạt qua trường `"model": "mlp" | "rf" | "linear"`. Phản hồi JSON:
  ```json
  {
    "status": "success",
    "model_used": "mlp",
    "predicted_medv_k": 45.09,
    "predicted_price_usd": 45085.0,
    "unit": "$1,000s"
  }
  ```

---

## 8. KẾT LUẬN & HƯỚNG PHÁT TRIỂN

### 8.1. Đóng góp nổi bật
1. Xây dựng hoàn chỉnh pipeline Deep Learning Hồi quy với kiến trúc mạng PyTorch MLP tối ưu đạt $R^2 > 0.90$.
2. Thực thi quy chuẩn kỹ thuật nghiêm ngặt: **Zero Data Leakage**, xử lý ngoại lai skewness, lưu trữ tham số JSON phục vụ suy luận thời gian thực.
3. Tạo lập bảng so sánh đối chuẩn khoa học giữa Deep Learning và 3 giải thuật học máy cổ điển.
4. Triển khai hoàn chỉnh Web REST API Flask và các công cụ CLI dòng lệnh, đáp ứng tiêu chuẩn $\le 200$ dòng lệnh trên từng file nguồn.

### 8.2. Hướng phát triển tiếp theo
- Thử nghiệm các kiến trúc Deep Learning chuyên sâu cho dữ liệu bảng như **TabNet** hoặc **FT-Transformer**.
- Ứng dụng kỹ thuật giải thích mô hình **SHAP (Shapley Additive exPlanations)** để phân tích mức độ đóng góp chi tiết của từng thuộc tính vào giá trị dự đoán cho từng ngôi nhà cụ thể.
