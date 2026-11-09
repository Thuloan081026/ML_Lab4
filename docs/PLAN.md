# Kế hoạch Thực hiện Lab 4 — Boston Housing Price

```text
Giai đoạn 1: Khám phá và Phân tích Dữ liệu (EDA)
  - Phân tích thống kê mô tả, kiểm tra missing, vẽ phân phối và tương quan.
Giai đoạn 2: Tiền xử lý & Kỹ thuật đặc trưng (Feature Engineering)
  - Chia tập 70/15/15 chống rò rỉ dữ liệu.
  - Xử lý Outlier bằng kiểm tra Skewness và biến đổi Log/Reflect-Log.
  - Chuẩn hóa Z-score dựa trên Mean/Std của tập Train.
Giai đoạn 3: Huấn luyện Mạng Nơ-ron Sâu (PyTorch MLP)
  - Xây dựng kiến trúc 13 -> 64 -> 32 -> 1, Adam optimizer, MSELoss.
  - Áp dụng Early Stopping và lưu trọng số best_mlp_boston.pth.
Giai đoạn 4: Đánh giá Đối chuẩn Mô hình (Benchmarking)
  - Huấn luyện Linear Regression, Decision Tree, Random Forest.
  - So sánh MSE, MAE, RMSE, R2 Score trên Test set.
Giai đoạn 5: Đóng gói Triển khai (Deployment)
  - Xây dựng CLI predict.py và Web API Flask (app.py).
```

