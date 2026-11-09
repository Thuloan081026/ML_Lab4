# Model Card — Boston Housing Price Predictors

- **Tên mô hình chính:** BostonMLP (Champion Deep Learning Model)
- **Mô hình so sánh:** Random Forest, Linear Regression, Decision Tree
- **Phiên bản:** 1.0.0
- **Nhiệm vụ:** Hồi quy giá trị trung vị bất động sản (biến `MEDV`)
- **Đặc trưng đầu vào (13 đặc trưng):**
  `CRIM`, `ZN`, `INDUS`, `CHAS`, `NOX`, `RM`, `AGE`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`, `LSTAT`.
- **Hiệu năng trên Test Set:**
  - PyTorch MLP: $R^2 = 0.9005$, MSE = 6.4960, MAE = 2.0274
  - Random Forest: $R^2 = 0.8964$, MSE = 6.7656, MAE = 1.8944
  - Linear Regression: $R^2 = 0.7956$, MSE = 13.3456, MAE = 2.7406
  - Decision Tree: $R^2 = 0.6114$, MSE = 25.3766, MAE = 2.7237
- **Định dạng trọng số:** `models/best_mlp_boston.pth`, `models/*.pkl`

