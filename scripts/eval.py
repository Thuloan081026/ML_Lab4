import os, sys, zipfile, joblib
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def evaluate_all():
    feat_dir = os.path.join(ROOT, "data", "features")
    models_dir = os.path.join(ROOT, "models")
    proc_dir = os.path.join(ROOT, "data", "processed")

    X_te_sc = np.load(os.path.join(feat_dir, "X_test_scaled.npy"))
    y_test = np.load(os.path.join(feat_dir, "y_test.npy"))
    df_te = pd.read_csv(os.path.join(proc_dir, "Boston-test-clean.csv"))
    features = [c for c in df_te.columns if c != "MEDV"]
    X_test_raw = df_te[features].values

    # 1. PyTorch MLP
    mlp_pth = os.path.join(models_dir, "best_mlp_boston.pth")
    with zipfile.ZipFile(mlp_pth, "r") as z:
        w1 = np.frombuffer(z.read("best_mlp_boston/data/0"), dtype=np.float32).reshape(64, 13)
        b1 = np.frombuffer(z.read("best_mlp_boston/data/1"), dtype=np.float32)
        w2 = np.frombuffer(z.read("best_mlp_boston/data/2"), dtype=np.float32).reshape(32, 64)
        b2 = np.frombuffer(z.read("best_mlp_boston/data/3"), dtype=np.float32)
        w3 = np.frombuffer(z.read("best_mlp_boston/data/4"), dtype=np.float32).reshape(1, 32)
        b3 = np.frombuffer(z.read("best_mlp_boston/data/5"), dtype=np.float32)
    z1 = np.maximum(0, X_te_sc @ w1.T + b1)
    z2 = np.maximum(0, z1 @ w2.T + b2)
    p_mlp = (z2 @ w3.T + b3).flatten()

    # 2. Classical Models
    lr_m = joblib.load(os.path.join(models_dir, "linear_model.pkl"))
    dt_m = joblib.load(os.path.join(models_dir, "tree_model.pkl"))
    rf_m = joblib.load(os.path.join(models_dir, "rf_model.pkl"))

    all_preds = {
        "PyTorch MLP": p_mlp,
        "Random Forest": rf_m.predict(X_test_raw),
        "Linear Regression": lr_m.predict(X_te_sc),
        "Decision Tree": dt_m.predict(X_test_raw)
    }

    print("\n" + "=" * 62)
    print("BẢNG ĐÁNH GIÁ ĐỘC LẬP CÁC MÔ HÌNH TRÊN TEST SET")
    print("=" * 62)
    print(f"{'Mô hình':<20} | {'MSE':<10} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}")
    print("-" * 62)
    for name, p in sorted(all_preds.items(), key=lambda x: -r2_score(y_test, x[1])):
        mse = mean_squared_error(y_test, p)
        mae = mean_absolute_error(y_test, p)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, p)
        print(f"{name:<20} | {mse:<10.4f} | {mae:<10.4f} | {rmse:<10.4f} | {r2:<10.4f}")
    print("=" * 62)

if __name__ == "__main__":
    evaluate_all()

