# END-TO-END MLP TRAINING PIPELINE

## 1. Tổng quan

### 1.1. Mục tiêu

Xây dựng một pipeline Machine Learning hoàn chỉnh sử dụng **Multilayer Perceptron (MLP)** để giải quyết bài toán **Regression** trên bộ dữ liệu **Boston Housing**.

Pipeline bao gồm toàn bộ quá trình:

```text
Raw Dataset
    ↓
Data Loading
    ↓
Data Validation
    ↓
Data Cleaning
    ↓
Exploratory Data Analysis (EDA)
    ↓
Data Splitting
    ↓
Data Preprocessing
    ↓
Dataset / DataLoader
    ↓
Build MLP Model
    ↓
Training
    ↓
Validation
    ↓
Hyperparameter Tuning
    ↓
Final Training
    ↓
Test Evaluation
    ↓
Model Analysis
    ↓
Save Model
    ↓
Inference / Prediction
```

### 1.2. Bài toán

Boston Housing là bài toán **Regression**.

Mục tiêu của mô hình:

> Dựa trên các đặc trưng của khu vực nhà ở, dự đoán giá trị trung vị của nhà.

Input:

```text
Các đặc trưng của khu vực
```

Output:

```text
Giá trị dự đoán của nhà
```

Mô hình sử dụng:

```text
Multilayer Perceptron (MLP)
```

Framework:

```text
PyTorch
```

---

# 2. Cấu trúc thư mục dự kiến

```text
PR4/
│
├── .agents/            # Cấu hình agents/AI
│
├── .venv/              # Virtual environment
│
├── data/
│   └── raw/
│       └── Boston-house-price-data.csv
│
├── docs/               # Tài liệu dự án
│
├── models/             # Lưu model sau khi train
│
├── src/
│   ├── add_eda_cells.py
│   ├── lab4.ipynb          # Notebook thực hiện EDA và Train
│   ├── notebook_code.py    # Code trích xuất từ notebook
│   └── plan_description.md # Kế hoạch triển khai
│
└── README.md
```

---

# 3. PHASE 01 — Chuẩn bị dữ liệu

## 3.1. Thu thập dataset

Sử dụng file:

```text
Boston-house-price-data.csv
```

Dataset được đặt tại:

```text
data/raw/
```

Không thực hiện preprocessing trực tiếp trên file raw.

Mục đích:

* Giữ nguyên dữ liệu ban đầu.
* Có thể quay lại kiểm tra dữ liệu gốc.
* Tránh mất dữ liệu khi preprocessing sai.

---

# 4. PHASE 02 — Load Dataset

## 4.1. Đọc dữ liệu

Sử dụng Pandas và xử lý sẵn các giá trị chuỗi rỗng/dấu cách:

```python
import pandas as pd
import numpy as np

df = pd.read_csv(
    "../data/raw/Boston-house-price-data.csv",
    na_values=["", " ", "nan", "NaN", "null", "NULL"]
)
```

## 4.2. Khắc phục chuỗi rỗng

```python
df = df.replace(r"^\s*$", np.nan, regex=True)
```

## 4.3. Kiểm tra thông tin dataset

```python
df.info()
display(df.head())
```

---

# 5. PHASE 03 — Data Validation (Missing Values)

Trước khi phân tích dữ liệu, cần xác định dataset có missing values hay không.

## 5.1. Kiểm tra Missing Values

```python
missing_counts = df.isna().sum()
print("Total missing:", missing_counts.sum())
```

## 5.2. Visualise Missing Values

Sử dụng **Heatmap** và **Bar chart** để có cái nhìn trực quan:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Vẽ Heatmap missing values
sns.heatmap(missing_counts.to_frame().T, annot=True, cmap='YlOrRd')

# Vẽ Bar chart số lượng missing mỗi cột
# (Chi tiết xem code trong notebook lab4.ipynb)
```

Các phương án:
* Phân tích nguyên nhân thiếu.
* Đánh giá xem có nên xoá dòng (Drop NA) hay thay thế (Imputation).

---

# 6. PHASE 04 — Kiểm tra Duplicate

Kiểm tra:

```python
dup_count = df.duplicated().sum()
print(dup_count)
```

## 6.1. Visualize Duplicated Rows

Sử dụng **Pie chart** và **Bar chart** để phân phối Unique vs Duplicated:

```python
# Phân chia số lượng Unique và Duplicated
unique_count = len(df) - dup_count
# Vẽ Pie chart tỷ lệ
# (Chi tiết xem code trong notebook lab4.ipynb)
```

Mục tiêu:
> Tránh một sample xuất hiện nhiều lần làm ảnh hưởng đến quá trình training.

---

# 7. PHASE 05 — Kiểm tra dữ liệu bất thường (IQR Method)

## 7.1. Phân tích thống kê

```python
description = df.describe()
display(description)
```

## 7.2. Tính IQR và phân định Outlier cho các Numeric Columns

Xác định ngưỡng dưới (Lower Bound) và ngưỡng trên (Upper Bound):

```text
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

In ra thống kê Outliers chi tiết cho từng cột:

```python
# Loop qua các cột số để tính và in số lượng Outlier
# (Chi tiết xem code trong notebook lab4.ipynb)
```

---

# 8. PHASE 06 — Exploratory Data Analysis (EDA)

Đây là giai đoạn tìm hiểu bản chất dataset bằng cách vẽ các phân phối và mối quan hệ.

## 8.1. Distribution of Numerical Features (Histogram)

Vẽ biểu đồ tần suất (Histogram) cho tất cả các đặc trưng số:

```python
numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols].hist(figsize=(16, 12), bins=30)
plt.show()
```

## 8.2. Scatter Plot: Features vs MEDV

Vẽ phân tán đồ (Scatter Plot) xem xét quan hệ giữa các Input Features và Target (MEDV):

```python
# Loop để vẽ scatter plot từng feature với MEDV
# Giúp phát hiện mối quan hệ tuyến tính / phi tuyến tính
```

Mục tiêu:
* Hiểu phân phối (skewness) của dữ liệu.
* Đánh giá sơ bộ mối quan hệ với giá nhà.

---

# 9. PHASE 07 — Q-Q Plot

Q-Q Plot được sử dụng để kiểm tra mức độ gần với phân phối chuẩn cho Target (MEDV).

```python
import scipy.stats as stats

stats.probplot(
    df["MEDV"].dropna(),
    dist="norm",
    plot=plt
)

plt.show()
```

Phân tích:
```text
Điểm gần đường thẳng  → Gần phân phối chuẩn
Điểm lệch mạnh       → Có khả năng không chuẩn
```

---

# 10. PHASE 08 — Outlier Analysis (Boxplot)

Trực quan hóa Outliers bằng biểu đồ Hộp (Boxplot) sử dụng giá trị ngắt IQR đã tính ở Phase 5.

```python
# Vẽ Boxplot dạng grid cho tất cả các numeric columns
# Hiển thị đường Lower/Upper bằng đường đứt nét
# Highlight các điểm flier bằng màu đỏ
```

Mục đích:
* Quan sát trực quan các giá trị cực đoan.
* Đánh giá có nên giữ hay xử lý các điểm này không (không phải tự động xoá toàn bộ outlier).

---

# 11. PHASE 09 — Correlation Analysis

Kiểm tra mối quan hệ đa cộng tuyến giữa các feature.

```python
correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.show()
```

Mục tiêu:
* Xác định feature có tương quan mạnh với target.
* Phát hiện các feature tương quan cao với nhau.

---

# 12. PHASE 10 — Feature / Target Definition

Xác định:

```text
X = Input Features
y = Target
```

Ví dụ:

```python
X = df.drop(columns=["MEDV"])
y = df["MEDV"]
```

Trong đó:
```text
X: Các đặc trưng đầu vào
y: Giá trị cần dự đoán
```

Kiểm tra:
```python
X.shape
y.shape
```

---

# 13. PHASE 11 — Train / Validation / Test Split

Không được train trực tiếp trên toàn bộ dataset.

Chia:

```text
Dataset
   │
   ├── Training Set
   │
   ├── Validation Set
   │
   └── Test Set
```

Ví dụ:

```text
70% Train
15% Validation
15% Test
```

Ý nghĩa:

### Training Set

Dùng để:

> Học trọng số của MLP.

### Validation Set

Dùng để:

> Theo dõi quá trình training và lựa chọn hyperparameter.

### Test Set

Dùng để:

> Đánh giá cuối cùng trên dữ liệu chưa từng được sử dụng để lựa chọn model.

---

# 14. PHASE 12 — Data Leakage Prevention

Đây là bước rất quan trọng.

Không được:

```text
Fit scaler trên toàn bộ dataset
```

Mà phải:

```text
Train data
    ↓
Fit scaler

Validation data
    ↓
Transform

Test data
    ↓
Transform
```

Ví dụ:

```python
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```

Lý do:

> Validation và Test phải mô phỏng dữ liệu thực tế mà model chưa nhìn thấy.

---

# 15. PHASE 13 — Feature Scaling

MLP rất nhạy với scale của input.

Các feature có thể có range rất khác nhau:

```text
Feature A: 0 → 1
Feature B: 0 → 100
Feature C: 0 → 1000
```

Nếu không scaling:

> Một số feature có giá trị lớn có thể ảnh hưởng mạnh đến quá trình tối ưu.

Có thể sử dụng Standardization:

```text
z = (x - mean) / standard_deviation
```

Sau scaling:

```text
mean ≈ 0
std ≈ 1
```

---

# 16. PHASE 14 — Target Preparation

Target `y` cũng cần được xem xét.

Có hai phương án:

### Phương án 1

Không scale target.

```text
X → StandardScaler
y → giữ nguyên
```

### Phương án 2

Scale target.

```text
X → StandardScaler
y → StandardScaler
```

Nếu scale target, khi prediction phải:

```text
Scaled Prediction
       ↓
Inverse Transform
       ↓
Original Prediction
```

Đối với lab cơ bản, có thể giữ target ở scale gốc để việc giải thích RMSE/MAE dễ hơn.

---

# 17. PHASE 15 — Convert sang Tensor

PyTorch sử dụng Tensor.

```python
import torch

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
)
```

Target regression cần shape phù hợp:

```text
[N]
```

hoặc:

```text
[N, 1]
```

Nên thống nhất thành:

```text
[N, 1]
```

---

# 18. PHASE 16 — PyTorch Dataset

Tạo custom Dataset hoặc sử dụng TensorDataset.

```python
from torch.utils.data import TensorDataset

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)
```

Tương tự:

```text
train_dataset
val_dataset
test_dataset
```

Mục tiêu:

> Đóng gói dữ liệu để PyTorch DataLoader có thể xử lý.

---

# 19. PHASE 17 — DataLoader

Tạo DataLoader:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

Validation:

```python
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)
```

Test:

```python
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
```

### Tại sao Train shuffle?

Để:

> Tránh model học theo thứ tự cố định của dữ liệu.

### Tại sao Validation/Test không shuffle?

Không cần thiết vì không cập nhật trọng số trong hai giai đoạn này.

---

# 20. PHASE 18 — Thiết kế kiến trúc MLP

MLP gồm:

```text
Input Layer
     ↓
Hidden Layer 1
     ↓
Activation
     ↓
Hidden Layer 2
     ↓
Activation
     ↓
Output Layer
```

Ví dụ:

```text
Input
  13
   ↓
Linear(13 → 64)
   ↓
ReLU
   ↓
Linear(64 → 32)
   ↓
ReLU
   ↓
Linear(32 → 1)
   ↓
Output
```

Đối với Regression:

```text
Output = 1 neuron
```

Không sử dụng activation ở output layer trong trường hợp target có thể nhận giá trị liên tục không bị giới hạn.

---

# 21. PHASE 19 — Activation Function

Có thể sử dụng:

```text
ReLU
```

Công thức:

```text
ReLU(x) = max(0, x)
```

Lý do sử dụng:

* Đơn giản.
* Tính toán nhanh.
* Giúp MLP học quan hệ phi tuyến.

---

# 22. PHASE 20 — Loss Function

Đây là bài toán Regression.

Có thể sử dụng:

```text
MSELoss
```

Công thức:

```text
MSE = 1/n Σ(y_pred - y)^2
```

Trong PyTorch:

```python
criterion = torch.nn.MSELoss()
```

Loss được sử dụng để:

> Đo mức độ sai khác giữa prediction và target.

---

# 23. PHASE 21 — Optimizer

Sử dụng:

```text
Adam
```

Ví dụ:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Optimizer có nhiệm vụ:

> Cập nhật weights và bias dựa trên gradient.

---

# 24. PHASE 22 — Training Loop

Training loop:

```text
For each epoch:

    For each batch:

        1. Forward Pass
        2. Calculate Loss
        3. Backward Pass
        4. Update Parameters
        5. Clear Gradients
```

Pseudo-code:

```text
FOR epoch = 1 → N:

    FOR each batch:

        prediction = model(X)

        loss = criterion(
            prediction,
            y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
```

---

# 25. PHASE 23 — Forward Propagation

Input:

```text
X
```

đi qua các layer:

```text
X
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
Prediction
```

Mục tiêu:

> Tạo ra giá trị dự đoán.

---

# 26. PHASE 24 — Backpropagation

Sau khi tính loss:

```text
Loss
 ↓
Backward()
 ↓
Gradient
 ↓
Optimizer
 ↓
Update Weight
```

PyTorch:

```python
loss.backward()
```

Gradient cho biết:

> Mỗi weight cần thay đổi theo hướng nào để giảm loss.

---

# 27. PHASE 25 — Validation

Sau mỗi epoch:

```text
Training
    ↓
Validation
```

Validation phải:

```python
model.eval()
```

và:

```python
with torch.no_grad():
```

Không cập nhật weight.

Theo dõi:

```text
Train Loss
Validation Loss
```

---

# 28. PHASE 26 — Training Curves

Vẽ:

```text
Epoch vs Training Loss
Epoch vs Validation Loss
```

Ví dụ:

```text
Loss
 │\
 │ \
 │  \
 │   \____
 │        \____
 │
 └──────────────── Epoch
```

Mục tiêu:

> Quan sát quá trình hội tụ của model.

---

# 29. PHASE 27 — Detect Overfitting

So sánh:

```text
Training Loss
Validation Loss
```

### Underfitting

```text
Train Loss cao
Validation Loss cao
```

### Good Fit

```text
Train Loss thấp
Validation Loss thấp
```

### Overfitting

```text
Train Loss tiếp tục giảm

Validation Loss:
        ↓
      giảm
        ↓
      tăng
```

Khi đó model đang học quá tốt training data nhưng khả năng tổng quát hóa giảm.

---

# 30. PHASE 28 — Early Stopping

Có thể sử dụng Early Stopping.

Ví dụ:

```text
patience = 20
```

Nếu validation loss không cải thiện sau 20 epochs:

```text
Stop Training
```

Mục đích:

> Tránh tiếp tục training khi model bắt đầu overfit.

---

# 31. PHASE 29 — Model Checkpoint

Không chỉ lưu model cuối cùng.

Nên lưu:

> Model có Validation Loss tốt nhất.

Logic:

```text
IF val_loss < best_val_loss:

    best_val_loss = val_loss

    save model
```

Ví dụ:

```python
torch.save(
    model.state_dict(),
    "models/best_mlp.pth"
)
```

---

# 32. PHASE 30 — Hyperparameter Tuning

Các hyperparameter cần thử:

```text
Learning Rate
Batch Size
Number of Epochs
Number of Hidden Layers
Number of Neurons
Dropout
Weight Decay
```

Ví dụ:

| Parameter     | Candidates            |
| ------------- | --------------------- |
| Learning Rate | 0.001, 0.0005, 0.0001 |
| Batch Size    | 16, 32, 64            |
| Hidden Units  | 32, 64, 128           |
| Hidden Layers | 1, 2, 3               |
| Dropout       | 0, 0.1, 0.2           |
| Weight Decay  | 0, 0.0001             |

Không chọn hyperparameter dựa trên Test Set.

Quy trình:

```text
Train
  ↓
Validation
  ↓
Select Best Hyperparameters
  ↓
Final Model
  ↓
Test
```

---

# 33. PHASE 31 — Final Training

Sau khi tìm được hyperparameter tốt nhất:

```text
Best Hyperparameters
        ↓
Train Final Model
        ↓
Best Validation Checkpoint
```

Không tiếp tục dùng Test Set để điều chỉnh hyperparameter.

---

# 34. PHASE 32 — Test Evaluation

Chỉ sử dụng Test Set sau khi model đã được lựa chọn.

Các metrics:

## MAE

```text
MAE = 1/n Σ|y - y_pred|
```

Ý nghĩa:

> Sai số tuyệt đối trung bình.

---

## MSE

```text
MSE = 1/n Σ(y - y_pred)^2
```

Ý nghĩa:

> Phạt mạnh các sai số lớn.

---

## RMSE

```text
RMSE = √MSE
```

Ý nghĩa:

> Sai số ở cùng đơn vị với target.

---

## R² Score

```text
R² = 1 - SS_res / SS_tot
```

Ý nghĩa:

> Đánh giá mức độ model giải thích được biến thiên của target.

---

# 35. PHASE 33 — Compare Actual vs Predicted

Tạo bảng:

```text
Actual    Predicted    Error
--------------------------------
20.5      21.2         0.7
15.2      14.8         0.4
30.1      29.4         0.7
...
```

Mục đích:

> Quan sát trực tiếp prediction của model.

---

# 36. PHASE 34 — Prediction Plot

Vẽ:

```text
Actual vs Predicted
```

Nếu model tốt:

```text
Predicted
   │       •
   │     •
   │   •
   │ •
   │•
   └──────────── Actual
```

Các điểm càng gần đường:

```text
y = x
```

thì prediction càng tốt.

---

# 37. PHASE 35 — Residual Analysis

Residual:

```text
Residual = Actual - Predicted
```

Phân tích:

```text
Residual vs Predicted
```

Model tốt thường có residual:

```text
phân bố tương đối ngẫu nhiên
quanh 0
```

Nếu residual có pattern rõ ràng:

> Có thể model chưa học được một số quan hệ trong dữ liệu.

---

# 38. PHASE 36 — Error Analysis

Tìm những sample có sai số lớn nhất.

```text
Sample
Actual
Prediction
Absolute Error
```

Phân tích:

```text
Tại sao model dự đoán sai?
```

Có thể do:

* Outlier.
* Dữ liệu đặc biệt.
* Feature chưa đủ thông tin.
* Model quá đơn giản.
* Noise.

---

# 39. PHASE 37 — Model Saving

Lưu:

```text
Model weights
Scaler
Feature names
Hyperparameters
```

Model:

```python
torch.save(
    model.state_dict(),
    "models/mlp_boston_housing.pth"
)
```

Scaler cũng phải được lưu để inference sử dụng đúng preprocessing.

---

# 40. PHASE 38 — Inference Pipeline

Khi có dữ liệu mới:

```text
New Data
    ↓
Validate
    ↓
Same Preprocessing
    ↓
Same Scaler
    ↓
Convert Tensor
    ↓
Load MLP
    ↓
Prediction
    ↓
Return Result
```

Điều quan trọng:

> Dữ liệu inference phải được preprocessing giống dữ liệu training.

Không được fit scaler lại trên dữ liệu mới.

---

# 41. PHASE 39 — Final Pipeline

Toàn bộ hệ thống:

```text
                    ┌─────────────────────┐
                    │   Raw CSV Dataset   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │   Data Validation   │
                    │ Missing / Duplicate │
                    │ Data Types / Errors │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │        EDA          │
                    │ Distribution        │
                    │ Q-Q Plot            │
                    │ Outlier             │
                    │ Correlation         │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Feature / Target    │
                    │     Selection       │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Train / Val / Test  │
                    │     Split           │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │    Preprocessing    │
                    │     Scaling         │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Tensor + Dataset    │
                    │     DataLoader      │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │      MLP Model      │
                    │ Input → Hidden →    │
                    │ Hidden → Output     │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │      Training       │
                    │ Forward             │
                    │ Loss                │
                    │ Backpropagation     │
                    │ Optimizer            │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │     Validation      │
                    │ Loss / Metrics      │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Hyperparameter      │
                    │ Tuning              │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │   Best Model        │
                    │   Checkpoint        │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │   Test Evaluation   │
                    │ MAE / MSE / RMSE    │
                    │ R²                  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │   Error Analysis    │
                    │ Actual vs Predicted │
                    │ Residual Analysis   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │    Save Model       │
                    │ + Scaler + Config   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │     Inference       │
                    │     New Data        │
                    └─────────────────────┘
```

---

# 42. PHASE 40 — Các kết quả cần lưu

Sau khi hoàn thành lab, cần có:

## Dataset Analysis

```text
Dataset information
Missing values
Duplicate values
Descriptive statistics
```

## EDA

```text
Histogram
Boxplot
Q-Q Plot
Correlation Heatmap
```

## Training

```text
Training Loss Curve
Validation Loss Curve
Best Epoch
Best Validation Loss
```

## Evaluation

```text
MAE
MSE
RMSE
R²
```

## Visualization

```text
Actual vs Predicted
Residual Plot
Prediction Error Distribution
```

## Model

```text
best_mlp.pth
scaler.pkl
config.json
```

---

# 43. PHASE 41 — Tiêu chí đánh giá cuối cùng

Model được đánh giá dựa trên:

### 1. Prediction Performance

```text
MAE ↓
MSE ↓
RMSE ↓
R² ↑
```

### 2. Generalization

So sánh:

```text
Train Performance
Validation Performance
Test Performance
```

Nếu chênh lệch quá lớn:

> Có khả năng overfitting.

### 3. Training Stability

Kiểm tra:

```text
Loss có giảm không?
Loss có dao động mạnh không?
Validation Loss có tăng không?
```

### 4. Error Distribution

Kiểm tra:

```text
Residual
Outlier prediction
Large errors
```

---

# 44. PHASE 42 — Báo cáo kết quả

Báo cáo cuối cùng nên trình bày:

## 1. Dataset

* Nguồn dữ liệu.
* Số lượng samples.
* Số lượng features.
* Target.
* Kiểu dữ liệu.

## 2. Data Cleaning

* Missing values.
* Duplicate.
* Outlier.
* Các vấn đề dữ liệu.

## 3. EDA

* Distribution.
* Q-Q Plot.
* Correlation.
* Outlier analysis.

## 4. Preprocessing

* Train/Validation/Test split.
* Scaling.
* Tensor conversion.

## 5. MLP Architecture

Ví dụ:

```text
13
 ↓
64
 ↓
ReLU
 ↓
32
 ↓
ReLU
 ↓
1
```

## 6. Training

* Loss function.
* Optimizer.
* Learning rate.
* Batch size.
* Epoch.
* Early stopping.

## 7. Evaluation

Bảng:

| Metric | Value |
| ------ | ----: |
| MAE    |   ... |
| MSE    |   ... |
| RMSE   |   ... |
| R²     |   ... |

## 8. Analysis

Trả lời:

* Model có overfit không?
* Model có underfit không?
* Feature scaling có cần thiết không?
* Hyperparameter nào tốt nhất?
* Prediction có chính xác không?
* Những trường hợp nào model dự đoán sai nhiều?
* Có thể cải thiện model như thế nào?

---

# 45. FINAL END-TO-END CHECKLIST

```text
[ ] Load raw dataset
[ ] Check shape
[ ] Check columns
[ ] Check data types
[ ] Check missing values
[ ] Check duplicates
[ ] Check invalid values
[ ] Descriptive statistics

[ ] Histogram
[ ] KDE
[ ] Boxplot
[ ] Q-Q Plot
[ ] Correlation heatmap
[ ] Outlier analysis

[ ] Define X
[ ] Define y
[ ] Train/Validation/Test split
[ ] Prevent data leakage
[ ] Fit scaler on train only
[ ] Transform validation/test
[ ] Convert to Tensor
[ ] Create Dataset
[ ] Create DataLoader

[ ] Design MLP
[ ] Define activation
[ ] Define loss function
[ ] Define optimizer

[ ] Training loop
[ ] Forward propagation
[ ] Calculate loss
[ ] Backpropagation
[ ] Update parameters
[ ] Validation

[ ] Training curves
[ ] Detect overfitting
[ ] Early stopping
[ ] Save best checkpoint

[ ] Hyperparameter tuning
[ ] Select best configuration
[ ] Final training

[ ] Test evaluation
[ ] MAE
[ ] MSE
[ ] RMSE
[ ] R²
[ ] Actual vs Predicted
[ ] Residual analysis
[ ] Error analysis

[ ] Save model
[ ] Save scaler
[ ] Save configuration

[ ] Build inference pipeline
[ ] Test prediction on new data

[ ] Complete report
```

---

# 46. Nguyên tắc quan trọng nhất

Pipeline phải đảm bảo nguyên tắc:

```text
                 DATA
                   │
                   ▼
              EDA / CHECK
                   │
                   ▼
             SPLIT DATASET
          ┌────────┼────────┐
          ▼        ▼        ▼
        TRAIN      VAL      TEST
          │        │        │
          ▼        │        │
       FIT SCALER  │        │
          │        │        │
          ▼        ▼        ▼
       TRANSFORM TRANSFORM TRANSFORM
          │        │        │
          └────────┼────────┘
                   ▼
                 MLP
                   │
                   ▼
               TRAINING
                   │
                   ▼
              VALIDATION
                   │
                   ▼
          HYPERPARAMETER TUNING
                   │
                   ▼
              BEST MODEL
                   │
                   ▼
                 TEST
                   │
                   ▼
              FINAL RESULT
```

**Quy tắc vàng:**

> **Không để Test Set tham gia vào quá trình lựa chọn hoặc tuning model.**

Test Set chỉ được sử dụng **một cách cuối cùng để đánh giá khả năng tổng quát hóa của model**.
