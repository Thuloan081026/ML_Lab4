# Data Pipeline Architecture — Boston Housing Price

```text
Raw Data (Boston-house-price-data.csv: 506 rows x 14 cols)
  │
  ├── 1. Train/Val/Test Split (70% / 15% / 15%):
  │      Cố định random_state=42, đảm bảo tương lai không rò rỉ vào quá khứ.
  ├── 2. Data Cleaning:
  │      Loại bỏ missing values và duplicates độc lập trên từng tập.
  ├── 3. Skewness Detection & Transformation:
  │      Phát hiện cột lệch trên Train (|skewness| > 1.0).
  │      Lệch phải -> Log1p(x); Lệch trái -> Log1p(max_val_train - x).
  ├── 4. Feature Scaling (Z-Score):
  │      Học Mean và Std trên Train.
  │      Áp dụng (X - Mean_train) / Std_train cho cả 3 tập.
  └── 5. Tensors / Arrays Export:
         Lưu trữ .npy và .csv phục vụ huấn luyện và đánh giá.
```

