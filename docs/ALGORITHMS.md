# CÁC THUẬT TOÁN — LAB 4: BOSTON HOUSING PRICE PREDICTION
---

## 1. MẠNG NƠ-RON HỌC SÂU PYTORCH MULTI-LAYER PERCEPTRON (`BostonMLP`)

### 1.1. Kiến trúc Mạng Nơ-ron Truyền thẳng (Feedforward Neural Network)
Mô hình `BostonMLP` là một mạng nơ-ron sâu đa tầng gồm 13 đặc trưng đầu vào, 2 tầng ẩn (Hidden Layers) phi tuyến và 1 nơ-ron đầu ra liên tục:
$$\text{Input } (13) \xrightarrow{W_1, b_1} \text{Linear}(13 \to 64) \xrightarrow{\text{ReLU}} h_1 \xrightarrow{W_2, b_2} \text{Linear}(64 \to 32) \xrightarrow{\text{ReLU}} h_2 \xrightarrow{W_3, b_3} \text{Linear}(32 \to 1) \to \hat{y}$$

1. **Tầng ẩn 1 (Hidden Layer 1):**
   $$z_1 = x W_1^T + b_1 \quad (W_1 \in \mathbb{R}^{64 \times 13}, b_1 \in \mathbb{R}^{64})$$
   $$h_1 = \text{ReLU}(z_1) = \max(0, z_1)$$
2. **Tầng ẩn 2 (Hidden Layer 2):**
   $$z_2 = h_1 W_2^T + b_2 \quad (W_2 \in \mathbb{R}^{32 \times 64}, b_2 \in \mathbb{R}^{32})$$
   $$h_2 = \text{ReLU}(z_2) = \max(0, z_2)$$
3. **Tầng đầu ra (Output Layer):**
   $$\hat{y} = h_2 W_3^T + b_3 \quad (W_3 \in \mathbb{R}^{1 \times 32}, b_3 \in \mathbb{R}^{1})$$
   *(Không dùng hàm kích hoạt ở tầng cuối để cho phép dự báo giá trị số thực liên tục bất kỳ)*.

### 1.2. Hàm Kích hoạt ReLU (Rectified Linear Unit)
$$\text{ReLU}(z) = \max(0, z), \quad \text{Đạo hàm: } \text{ReLU}'(z) = \begin{cases} 1 & \text{khi } z > 0 \\ 0 & \text{khi } z \le 0 \end{cases}$$
- **Lợi ích:** Triệt tiêu hoàn toàn hiện tượng biến mất đạo hàm (Vanishing Gradient) vốn gặp phải ở hàm Sigmoid hoặc Tanh, giúp việc huấn luyện qua nhiều tầng diễn ra nhanh và ổn định.

### 1.3. Hàm Mất mát MSE Loss & Thuật toán Tối ưu Adam
- **Mean Squared Error Loss:**
  $$\mathcal{L}_{\text{MSE}}(y, \hat{y}) = \frac{1}{B} \sum_{i=1}^B (y_i - \hat{y}_i)^2$$
- **Thuật toán Adam (Adaptive Moment Estimation):**
  Adam kết hợp ưu điểm của Momentum (quán tính đạo hàm bậc 1) và RMSprop (thích ứng bước nhảy theo phương sai đạo hàm bậc 2):
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\text{Ước lượng Moment bậc 1})$$
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\text{Ước lượng Moment bậc 2})$$
  Hiệu chỉnh độ chệch (Bias Correction):
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
  Cập nhật trọng số tham số $\theta$:
  $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
  *(Với tốc độ học $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$)*.

### 1.4. Cơ chế Dừng sớm (Early Stopping)
Để chống học vẹt (Overfitting) trên tập huấn luyện, sau mỗi epoch, hệ thống đo lường MSE trên tập Validation độc lập:
- Nếu $\text{Loss}_{\text{val}}$ đạt kỷ lục mới thấp hơn, lưu trạng thái mô hình tốt nhất vào `best_mlp_boston.pth` và thiết lập lại bộ đếm kiên nhẫn `p_count = 0`.
- Nếu sau `patience = 20` epoch liên tiếp mà $\text{Loss}_{\text{val}}$ không giảm, quá trình huấn luyện lập tức dừng lại (`break`).

### 1.5. Cơ chế Suy luận Thuần túy Không phụ thuộc Thư viện (Pure NumPy Fallback)
Khi môi trường chưa cài đặt PyTorch, tệp `train.py` và `predict.py` sử dụng thư viện `zipfile` của Python để bóc tách trực tiếp các tensor nhị phân trong file checkpoint `.pth` và thực hiện lan truyền xuôi bằng ma trận NumPy:
```python
z1 = np.maximum(0, X_scaled @ w1.T + b1)
z2 = np.maximum(0, z1 @ w2.T + b2)
y_pred = (z2 @ w3.T + b3).flatten()
```
Đảm bảo khả năng triển khai mô hình nhẹ, độc lập và chạy được trên mọi server mà không cần GPU hay PyTorch cồng kềnh.

---

## 2. CÁC MÔ HÌNH HỌC MÁY CỔ ĐIỂN ĐỐI CHỨNG (BASELINE BENCHMARKS)

### 2.1. Hồi quy Tuyến tính Đa biến (Ordinary Least Squares - OLS)
Tìm siêu phẳng tham số cực tiểu hóa tổng bình phương phần dư bằng nghiệm giải tích:
$$\hat{\beta} = (X^T X)^{-1} X^T y$$
Dùng làm mốc tham chiếu cơ sở (Baseline) để chứng minh xem các quy luật phi tuyến phức tạp của thị trường nhà đất có thực sự cần tới Cây quyết định hay Mạng nơ-ron hay không.

### 2.2. Hồi quy Cây Quyết định (Decision Tree Regressor - CART)
Phân chia không gian 13 chiều bằng các mặt phẳng trực giao. Tại mỗi bước rẽ nhánh, thuật toán tìm biến $j$ và ngưỡng $s$ tối đa hóa mức độ giảm phương sai:
$$\Delta \sigma^2 = \sigma^2_{\text{parent}} - \left( \frac{N_L}{N}\sigma^2_L + \frac{N_R}{N}\sigma^2_R \right)$$
Giá trị dự báo tại nút lá là giá trị trung bình cộng giá nhà của các mẫu rơi vào lá đó.

### 2.3. Rừng Ngẫu nhiên Hồi quy (Random Forest Regressor - Quán quân Thực nghiệm 🏆)
Mô hình kết hợp (Ensemble Learning) gồm $B = 100$ cây quyết định độc lập:
1. **Bootstrap Aggregating (Bagging):** Mỗi cây $b$ được huấn luyện trên một tập dữ liệu con kích thước $N$ được rút mẫu ngẫu nhiên có hoàn lại (Sampling with Replacement) từ tập Train gốc.
2. **Feature Subsampling:** Tại mỗi nút rẽ nhánh của cây, chỉ có một tập con ngẫu nhiên các đặc trưng ($m \approx D/3$) được phép xem xét. Điều này giúp khử tương quan (Decorrelation) giữa các cây.
3. **Dự báo Tổng hợp (Aggregation):**
   $$\hat{y}_{\text{RF}}(x) = \frac{1}{B} \sum_{b=1}^B T_b(x)$$
   - *Tại sao Random Forest đạt $R^2 = 0.88$ cao nhất?* Việc lấy trung bình cộng của 100 cây giúp phương sai (Variance) của mô hình giảm xuống xấp xỉ $\frac{1}{B}$, triệt tiêu hoàn toàn tính bất ổn định của từng cây đơn lẻ mà vẫn giữ nguyên độ chệch (Bias) thấp.

---

## 3. THUẬT TOÁN TIỀN XỬ LÝ & BIẾN ĐỔI PHI TUYẾN DỮ LIỆU

### 3.1. Đo lường Hệ số Bất đối xứng (Skewness Detection)
Hệ số Skewness của một biến ngẫu nhiên được tính theo Moment bậc 3 chuẩn hóa:
$$\text{Skew}(X) = \frac{\frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^3}{\left( \frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^2 \right)^{3/2}}$$

### 3.2. Biến đổi Logarit Phi tuyến Khử Lệch (Log Transformation)
Nếu một biến có $|\text{Skew}| > 1.0$, phân phối của nó bị kéo dài đuôi sang một phía (nhiều ngoại lai lệch cực đoan):
1. **Lệch dương nặng ($\text{Skew} > +1.0$, ví dụ biến tội phạm `CRIM`, `ZN`, `LSTAT`):**
   Áp dụng hàm biến đổi Logarit cơ số tự nhiên:
   $$x' = \ln(1 + x) \quad (\text{sử dụng } \texttt{np.log1p})$$
   Giúp co ngắn khoảng cách khổng lồ giữa các thị trấn có tỉ lệ tội phạm 80-90 về gần với phân phối Gauss hình chuông.
2. **Lệch âm nặng ($\text{Skew} < -1.0$):**
   Áp dụng phép phản xạ kết hợp logarit (Reflected Log):
   $$x' = \ln\left(1 + (\max(X_{\text{train}}) - x)\right)$$

### 3.3. Chuẩn hóa Phân phối Z-Score (StandardScaler)
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
Toàn bộ giá trị trung bình $\mu_{\text{train}}$ và độ lệch chuẩn $\sigma_{\text{train}}$ được tính trên tập Train và lưu vào tệp `models/preprocessing_meta.json` để phục vụ quy trình chuẩn hóa đồng bộ cho Validation, Test và Web Demo.

---

## 4. HỆ THỐNG CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH HỒI QUY (EVALUATION METRICS)

1. **Mean Squared Error (MSE):**
   $$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
   *(Đơn vị: $(\text{nghìn USD})^2$ - Phạt rất nặng các sai số lớn)*.
2. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   *(Đơn vị: nghìn USD - Sai số bình quân thực tế)*.
3. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\text{MSE}}$$
4. **Hệ số Xác định (R-Squared Score):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
   *(Phản ánh tỷ lệ phần trăm sự biến động của giá nhà Boston được giải thích bởi 13 đặc trưng đầu vào)*.
