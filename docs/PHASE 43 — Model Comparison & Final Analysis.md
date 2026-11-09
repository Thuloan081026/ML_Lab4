# 43. PHASE 43 — MODEL COMPARISON & FINAL ANALYSIS

## 43.1. Mục tiêu

So sánh hiệu năng của 4 mô hình Regression trên cùng một bộ dữ liệu:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression
4. MLP Regression

Mục tiêu cuối cùng:

> Xác định mô hình nào có khả năng dự đoán giá nhà tốt nhất trên **Test Set** và phân tích nguyên nhân dẫn đến sự khác biệt về hiệu năng giữa các mô hình.

---

# 43.2. Các mô hình được sử dụng

```text
                    Regression Models
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
 Linear Regression   Decision Tree    Random Forest
                                          
                           +
                           
                         MLP
```

### Model 1 — Linear Regression

Mô hình tuyến tính:

```text
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Mục đích:

> Sử dụng làm baseline để đánh giá xem quan hệ tuyến tính có đủ để dự đoán giá nhà hay không.

---

### Model 2 — Decision Tree Regression

Decision Tree chia dữ liệu thành các vùng dựa trên các feature.

```text
              Feature
              /     \
             /       \
        Condition   Condition
          /             \
       Node             Node
       /  \             /  \
     ...   ...         ...  ...
```

Mục đích:

> Kiểm tra khả năng mô hình hóa các quan hệ phi tuyến mà Linear Regression không thể biểu diễn tốt.

---

### Model 3 — Random Forest Regression

Random Forest kết hợp nhiều Decision Tree.

```text
              Dataset
                 │
       ┌─────────┼─────────┐
       ↓         ↓         ↓
     Tree 1    Tree 2    Tree N
       ↓         ↓         ↓
     Pred 1    Pred 2    Pred N
       └─────────┼─────────┘
                 ↓
          Average Prediction
```

Mục đích:

> Tăng khả năng tổng quát hóa và giảm hiện tượng overfitting của một Decision Tree đơn lẻ.

---

### Model 4 — MLP Regression

MLP đã được huấn luyện ở các phase trước:

```text
Input
  ↓
Hidden Layer
  ↓
ReLU
  ↓
Hidden Layer
  ↓
ReLU
  ↓
Output
```

Mục đích:

> Kiểm tra khả năng của mạng neuron trong việc học các quan hệ phi tuyến phức tạp giữa các feature và giá nhà.

---

# 43.3. Chuẩn hóa dữ liệu đầu vào cho các model

Tất cả model phải được đánh giá trên **cùng một Test Set**.

Tuy nhiên, preprocessing có thể khác nhau giữa các model.

### Linear Regression

Có thể sử dụng dữ liệu đã scaling:

```text
X_train_scaled
X_test_scaled
```

### Decision Tree

Không bắt buộc scaling:

```text
X_train
X_test
```

### Random Forest

Không bắt buộc scaling:

```text
X_train
X_test
```

### MLP

Sử dụng dữ liệu đã scaling:

```text
X_train_scaled
X_test_scaled
```

Điểm quan trọng:

> Không được thay đổi Test Set giữa các model.

---

# 43.4. Train Linear Regression

Import:

```python
from sklearn.linear_model import LinearRegression
```

Khởi tạo:

```python
linear_model = LinearRegression()
```

Training:

```python
linear_model.fit(
    X_train_scaled,
    y_train
)
```

Prediction:

```python
y_pred_linear = linear_model.predict(
    X_test_scaled
)
```

---

# 43.5. Train Decision Tree Regression

Import:

```python
from sklearn.tree import DecisionTreeRegressor
```

Khởi tạo:

```python
tree_model = DecisionTreeRegressor(
    random_state=42
)
```

Training:

```python
tree_model.fit(
    X_train,
    y_train
)
```

Prediction:

```python
y_pred_tree = tree_model.predict(
    X_test
)
```

---

# 43.6. Train Random Forest Regression

Import:

```python
from sklearn.ensemble import RandomForestRegressor
```

Khởi tạo:

```python
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

Training:

```python
rf_model.fit(
    X_train,
    y_train
)
```

Prediction:

```python
y_pred_rf = rf_model.predict(
    X_test
)
```

---

# 43.7. Generate MLP Predictions

Load model tốt nhất đã lưu:

```python
model.load_state_dict(
    torch.load(
        "models/best_mlp.pth"
    )
)
```

Chuyển sang evaluation mode:

```python
model.eval()
```

Prediction:

```python
with torch.no_grad():
    y_pred_mlp = model(
        X_test_tensor
    )
```

Sau đó chuyển Tensor về NumPy:

```python
y_pred_mlp = y_pred_mlp.cpu().numpy()
```

Nếu cần:

```python
y_pred_mlp = y_pred_mlp.reshape(-1)
```

---

# 43.8. Đảm bảo các prediction có cùng test samples

Đây là bước rất quan trọng.

Cần đảm bảo:

```text
y_test
   │
   ├── Linear Regression Prediction
   ├── Decision Tree Prediction
   ├── Random Forest Prediction
   └── MLP Prediction
```

Tất cả prediction phải tương ứng với:

> **Cùng một sample trong Test Set.**

Không được đánh giá các model trên các Test Set khác nhau.

---

# 43.9. Calculate MSE

MSE:

```text
MSE = Mean Squared Error
```

Sử dụng:

```python
from sklearn.metrics import mean_squared_error

mse_linear = mean_squared_error(
    y_test,
    y_pred_linear
)

mse_tree = mean_squared_error(
    y_test,
    y_pred_tree
)

mse_rf = mean_squared_error(
    y_test,
    y_pred_rf
)

mse_mlp = mean_squared_error(
    y_test,
    y_pred_mlp
)
```

Nguyên tắc:

```text
MSE càng nhỏ → Model càng tốt
```

---

# 43.10. Calculate R² Score

R² cho biết model giải thích được bao nhiêu biến thiên của target.

```python
from sklearn.metrics import r2_score
```

Tính:

```python
r2_linear = r2_score(
    y_test,
    y_pred_linear
)

r2_tree = r2_score(
    y_test,
    y_pred_tree
)

r2_rf = r2_score(
    y_test,
    y_pred_rf
)

r2_mlp = r2_score(
    y_test,
    y_pred_mlp
)
```

Nguyên tắc:

```text
R² càng cao → Model càng tốt
```

---

# 43.11. Tạo bảng Performance Comparison

Tạo bảng tổng hợp:

| Model | MSE ↓ | R² ↑ |
|---|---:|---:|
| Linear Regression | ... | ... |
| Decision Tree | ... | ... |
| Random Forest | ... | ... |
| MLP | ... | ... |

Trong đó:

```text
↓ = càng thấp càng tốt
↑ = càng cao càng tốt
```

Đây là bảng quan trọng nhất của Phase 43.

---

# 43.12. So sánh Prediction

Không chỉ so sánh metric.

Cần lấy một số sample:

| Actual | Linear | Tree | Random Forest | MLP |
|---:|---:|---:|---:|---:|
| ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |

Mục tiêu:

> Quan sát trực tiếp cách từng model dự đoán cùng một căn nhà.

---

# 43.13. Actual vs Predicted Plot

Vẽ Actual và Prediction của từng model.

Có thể thực hiện riêng:

```text
Actual vs Linear Regression
Actual vs Decision Tree
Actual vs Random Forest
Actual vs MLP
```

Mục tiêu:

> Quan sát mức độ prediction bám sát giá trị thực tế.

Model tốt sẽ có các điểm prediction nằm gần đường:

```text
y = x
```

---

# 43.14. Residual Comparison

Tính residual:

```text
Residual = Actual - Predicted
```

Cho từng model:

```python
residual_linear = y_test - y_pred_linear
residual_tree = y_test - y_pred_tree
residual_rf = y_test - y_pred_rf
residual_mlp = y_test - y_pred_mlp
```

Phân tích:

- Residual trung bình.
- Residual lớn nhất.
- Phân bố residual.
- Các prediction có error lớn.

Mục tiêu:

> Xác định model nào có sai số ổn định hơn.

---

# 43.15. Phân tích Model tốt nhất

Dựa trên:

```text
MSE
R²
Prediction
Residual
```

Xác định:

```text
Best Model
```

Tiêu chí chính:

```text
MSE thấp nhất
+
R² cao nhất
```

Nếu hai metric không cùng chỉ ra một model, cần phân tích thêm thay vì chỉ dựa vào một metric.

---

# 43.16. Phân tích Linear Regression

Cần trả lời:

### Câu hỏi 1

Linear Regression hoạt động tốt hay không?

### Câu hỏi 2

Nếu kết quả thấp hơn các model phi tuyến:

> Có phải quan hệ giữa feature và house price không hoàn toàn tuyến tính?

### Câu hỏi 3

Nếu Linear Regression vẫn đạt kết quả tốt:

> Điều đó cho thấy quan hệ tuyến tính đã giải thích được một phần đáng kể biến thiên của target.

---

# 43.17. Phân tích Decision Tree

Kiểm tra:

> Decision Tree có tốt hơn Linear Regression không?

Nếu có:

```text
Decision Tree > Linear Regression
```

Có thể cho thấy:

> Dataset chứa các quan hệ phi tuyến mà Linear Regression khó mô hình hóa.

Tuy nhiên Decision Tree đơn có thể:

- Overfit.
- Nhạy với dữ liệu.
- Có prediction không ổn định.

---

# 43.18. Phân tích Random Forest

So sánh:

```text
Random Forest
vs
Decision Tree
```

Nếu:

```text
Random Forest MSE < Decision Tree MSE
```

và:

```text
Random Forest R² > Decision Tree R²
```

thì có thể kết luận:

> Việc kết hợp nhiều Decision Tree giúp cải thiện khả năng tổng quát hóa so với một Decision Tree đơn lẻ.

---

# 43.19. Phân tích MLP

So sánh:

```text
MLP
vs
Linear Regression
```

Nếu MLP tốt hơn:

> MLP có khả năng học các quan hệ phi tuyến phức tạp hơn mô hình tuyến tính.

So sánh:

```text
MLP
vs
Decision Tree
vs
Random Forest
```

Nếu MLP tốt nhất:

> MLP có khả năng biểu diễn các quan hệ phi tuyến phù hợp với dataset.

Nếu MLP không tốt nhất:

> Không có nghĩa MLP là mô hình kém. Dataset có thể nhỏ, trong khi Tree-based models thường hoạt động rất tốt trên dữ liệu dạng bảng; MLP cũng nhạy với kiến trúc, scaling và hyperparameter.

Đây là điểm rất quan trọng khi viết phần phân tích.

---

# 43.20. So sánh Training và Testing Performance

Nếu có validation results của MLP và các model khác, có thể xây dựng:

| Model | Train | Validation | Test |
|---|---:|---:|---:|
| Linear Regression | ... | ... | ... |
| Decision Tree | ... | ... | ... |
| Random Forest | ... | ... | ... |
| MLP | ... | ... | ... |

Mục tiêu:

> Kiểm tra khả năng tổng quát hóa.

Đặc biệt cần chú ý Decision Tree và MLP có dấu hiệu overfitting hay không.

---

# 43.21. Phân tích độ chênh lệch Performance

Ví dụ:

```text
Random Forest
MSE = 20
R² = 0.85

MLP
MSE = 24
R² = 0.82
```

Không chỉ kết luận:

> Random Forest tốt hơn MLP.

Mà cần phân tích:

```text
Random Forest tốt hơn
        ↓
Có khả năng xử lý quan hệ phi tuyến
        ↓
Hoạt động tốt trên dữ liệu dạng tabular
        ↓
Ensemble nhiều cây giúp giảm variance
```

Đây mới là phần **Analysis**.

---

# 43.22. Xếp hạng các Model

Có thể xếp hạng:

```text
1. Model có MSE thấp nhất
2. Model có MSE thấp thứ hai
3. ...
```

và:

```text
1. Model có R² cao nhất
2. Model có R² cao thứ hai
3. ...
```

Ví dụ:

| Rank | Model | MSE | R² |
|---:|---|---:|---:|
| 1 | Random Forest | ... | ... |
| 2 | MLP | ... | ... |
| 3 | Decision Tree | ... | ... |
| 4 | Linear Regression | ... | ... |

Thứ tự thực tế phải dựa trên kết quả chạy lab, không được giả định trước.

---

# 43.23. Final Conclusion

Phần kết luận cần trả lời 5 câu hỏi:

### 1. Model nào có MSE thấp nhất?

```text
→ ...
```

### 2. Model nào có R² cao nhất?

```text
→ ...
```

### 3. MLP có tốt hơn các traditional regression models không?

```text
→ Có / Không
```

### 4. Tại sao model đó tốt hơn?

Phân tích dựa trên:

- Linear vs nonlinear relationship.
- Model capacity.
- Overfitting.
- Generalization.
- Dataset characteristics.

### 5. Model nào nên được lựa chọn?

Kết luận dựa trên:

```text
Performance
+
Generalization
+
Error Analysis
```

Không chỉ dựa vào một metric.

---

# 43.24. Final Comparison Pipeline

```text
                 TEST SET
                    │
          ┌─────────┼─────────┐
          │         │         │
          ▼         ▼         ▼
       Linear     Tree       Random
     Regression  Regression   Forest
          │         │         │
          │         │         │
          └─────────┼─────────┘
                    │
                    │
                    ▼
                  MLP
                    │
                    ▼
             Predictions
                    │
                    ▼
        ┌──────────────────────┐
        │   Performance        │
        │   Evaluation         │
        ├──────────────────────┤
        │ MSE                  │
        │ R²                   │
        │ Actual vs Predicted  │
        │ Residual Analysis    │
        └──────────┬───────────┘
                   ↓
          Model Comparison
                   ↓
          Best Model Selection
                   ↓
          Final Conclusion
```

---

# 43.25. Final Checklist

```text
[ ] Load trained MLP
[ ] Load Test Set

[ ] Train Linear Regression
[ ] Predict Test Set

[ ] Train Decision Tree Regression
[ ] Predict Test Set

[ ] Train Random Forest Regression
[ ] Predict Test Set

[ ] Generate MLP predictions

[ ] Calculate MSE
[ ] Calculate R²

[ ] Create comparison table

[ ] Compare actual vs predicted

[ ] Plot Actual vs Predicted

[ ] Calculate residuals

[ ] Analyze residuals

[ ] Compare Linear Regression vs MLP

[ ] Compare Decision Tree vs MLP

[ ] Compare Random Forest vs MLP

[ ] Determine best model

[ ] Explain why the best model performs better

[ ] Discuss MLP performance

[ ] Discuss overfitting/generalization

[ ] Write final conclusion
```

---

# 43.26. Expected Final Output

Sau Phase 43, lab phải có một kết quả tổng hợp như:

```text
================ MODEL COMPARISON ================

Model                  MSE          R²
----------------------------------------------------
Linear Regression      ...          ...
Decision Tree          ...          ...
Random Forest          ...          ...
MLP                    ...          ...

----------------------------------------------------
Best Model: ...
====================================================
```

Và phần kết luận:

```text
Based on the MSE and R² scores on the testing set,
the best-performing model is ______.

Compared with Linear Regression, ______ performs
better/worse because ______.

Compared with Decision Tree Regression, ______
performs better/worse because ______.

Compared with Random Forest Regression, ______
performs better/worse because ______.

Overall, the results indicate that ______ is the
most suitable model for this Boston Housing dataset.
```

---

# 43.27. Nguyên tắc cuối cùng của toàn bộ Lab

```text
EDA
 ↓
Preprocessing
 ↓
Train / Validation / Test
 ↓
        ┌───────────────┐
        │               │
        ▼               ▼
      MLP         Traditional Models
        │               │
        │       ┌───────┼────────┐
        │       ↓       ↓        ↓
        │    Linear    Tree   Random Forest
        │               │
        └───────┬───────┘
                ↓
          Same Test Set
                ↓
          MSE + R²
                ↓
       Prediction Comparison
                ↓
         Error Analysis
                ↓
        Final Model Analysis
                ↓
           Conclusion
```

**Điểm quan trọng nhất:** Phase cuối này không chỉ là "chạy thêm 3 model". Mục tiêu của nó là tạo một **fair comparison**: các model cùng giải một bài toán, được đánh giá trên **cùng Test Set**, bằng **cùng MSE và R²**, sau đó mới phân tích tại sao kết quả khác nhau.