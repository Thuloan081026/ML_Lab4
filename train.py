import os, sys, json, zipfile, importlib
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Nạp động PyTorch an toàn tránh lỗi linter Pylance khi chưa có torch
torch, nn, optim, DataLoader, TensorDataset = None, None, None, None, None
HAS_TORCH = False
try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    optim = importlib.import_module("torch.optim")
    _tdata = importlib.import_module("torch.utils.data")
    DataLoader, TensorDataset = _tdata.DataLoader, _tdata.TensorDataset
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

def run_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, "data", "raw", "Boston-house-price-data.csv")
    models_dir = os.path.join(base_dir, "models")
    feat_dir = os.path.join(base_dir, "data", "features")
    proc_dir = os.path.join(base_dir, "data", "processed")
    rep_dir = os.path.join(base_dir, "reports")
    for d in [models_dir, feat_dir, proc_dir, rep_dir]: os.makedirs(d, exist_ok=True)

    print("=" * 65)
    print("LAB 4: BOSTON HOUSING PRICE PREDICTION PIPELINE")
    print("=" * 65)

    df = pd.read_csv(raw_path, na_values=["", " ", "nan", "NaN", "null", "NULL"]).replace(r"^\s*$", np.nan, regex=True)
    features = [c for c in df.select_dtypes(include="number").columns if c != "MEDV"]

    X, y = df[features].copy(), df["MEDV"].copy()
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.15, random_state=42)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.1765, random_state=42)

    df_tr = pd.concat([X_tr, y_tr], axis=1).dropna().drop_duplicates().reset_index(drop=True)
    df_va = pd.concat([X_va, y_va], axis=1).dropna().drop_duplicates().reset_index(drop=True)
    df_te = pd.concat([X_te, y_te], axis=1).dropna().drop_duplicates().reset_index(drop=True)

    skew_info = {}
    for col in features:
        sk = float(df_tr[col].skew())
        if abs(sk) > 1.0:
            ttype = "reflect_log" if sk < -1.0 else "log"
            skew_info[col] = {"skewness": round(sk, 4), "transform": ttype}
            if ttype == "reflect_log": skew_info[col]["max_val"] = float(df_tr[col].max())

    for col, info in skew_info.items():
        for d in [df_tr, df_va, df_te]:
            d[col] = np.log1p(info["max_val"] - d[col]) if info["transform"] == "reflect_log" else np.log1p(d[col])

    df_tr.to_csv(os.path.join(proc_dir, "Boston-train-clean.csv"), index=False)
    df_va.to_csv(os.path.join(proc_dir, "Boston-val-clean.csv"), index=False)
    df_te.to_csv(os.path.join(proc_dir, "Boston-test-clean.csv"), index=False)

    X_train, y_train = df_tr[features], df_tr["MEDV"].values
    X_val, y_val = df_va[features], df_va["MEDV"].values
    X_test, y_test = df_te[features], df_te["MEDV"].values

    mu, sigma = X_train.mean(), X_train.std()
    X_tr_sc = ((X_train - mu) / sigma).values
    X_va_sc = ((X_val - mu) / sigma).values
    X_te_sc = ((X_test - mu) / sigma).values

    prep_meta = {"features": features, "skew_info": skew_info, "mean": mu.to_dict(), "std": sigma.to_dict()}
    with open(os.path.join(models_dir, "preprocessing_meta.json"), "w", encoding="utf-8") as f:
        json.dump(prep_meta, f, indent=2)

    for name, arr in [("X_train_scaled", X_tr_sc), ("X_val_scaled", X_va_sc), ("X_test_scaled", X_te_sc),
                      ("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        np.save(os.path.join(feat_dir, f"{name}.npy"), arr)

    mlp_pth = os.path.join(models_dir, "best_mlp_boston.pth")

    if HAS_TORCH and nn is not None and torch is not None:
        print("[1/2] Huấn luyện Deep Learning MLP bằng PyTorch...")
        class BostonMLP(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
            def forward(self, x): return self.net(x)

        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tr_ds = TensorDataset(torch.tensor(X_tr_sc, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        va_ds = TensorDataset(torch.tensor(X_va_sc, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
        tr_loader, va_loader = DataLoader(tr_ds, batch_size=32, shuffle=True), DataLoader(va_ds, batch_size=32, shuffle=False)

        mlp = BostonMLP(len(features)).to(device)
        crit, opt = nn.MSELoss(), optim.Adam(mlp.parameters(), lr=0.001)
        best_loss, pat, p_count = float("inf"), 20, 0

        for ep in range(1000):
            mlp.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                crit(mlp(bx), by).backward()
                opt.step()
            mlp.eval()
            v_loss = 0.0
            with torch.no_grad():
                for bx, by in va_loader:
                    bx, by = bx.to(device), by.to(device)
                    v_loss += crit(mlp(bx), by).item() * bx.size(0)
            v_loss /= len(va_ds)
            if v_loss < best_loss:
                best_loss, p_count = v_loss, 0
                torch.save(mlp.state_dict(), mlp_pth)
            else:
                p_count += 1
                if p_count >= pat:
                    print(f"  -> Early stopping tai epoch {ep+1} | Best Val MSE: {best_loss:.4f}")
                    break

        mlp_best = BostonMLP(len(features))
        mlp_best.load_state_dict(torch.load(mlp_pth, weights_only=True))
        mlp_best.eval()
        with torch.no_grad():
            mlp_preds = mlp_best(torch.tensor(X_te_sc, dtype=torch.float32)).numpy().flatten()
    else:
        print("[1/2] Nạp mô hình BostonMLP (Trọng số đã huấn luyện từ best_mlp_boston.pth)...")
        with zipfile.ZipFile(mlp_pth, "r") as z:
            w1 = np.frombuffer(z.read("best_mlp_boston/data/0"), dtype=np.float32).reshape(64, 13)
            b1 = np.frombuffer(z.read("best_mlp_boston/data/1"), dtype=np.float32)
            w2 = np.frombuffer(z.read("best_mlp_boston/data/2"), dtype=np.float32).reshape(32, 64)
            b2 = np.frombuffer(z.read("best_mlp_boston/data/3"), dtype=np.float32)
            w3 = np.frombuffer(z.read("best_mlp_boston/data/4"), dtype=np.float32).reshape(1, 32)
            b3 = np.frombuffer(z.read("best_mlp_boston/data/5"), dtype=np.float32)
        z1 = np.maximum(0, X_te_sc @ w1.T + b1)
        z2 = np.maximum(0, z1 @ w2.T + b2)
        mlp_preds = (z2 @ w3.T + b3).flatten()

    print("[2/2] Huấn luyện các mô hình Machine Learning cổ điển...")
    lr_m = LinearRegression().fit(X_tr_sc, y_train)
    dt_m = DecisionTreeRegressor(random_state=42).fit(X_train.values, y_train)
    rf_m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train.values, y_train)

    joblib.dump(lr_m, os.path.join(models_dir, "linear_model.pkl"))
    joblib.dump(dt_m, os.path.join(models_dir, "tree_model.pkl"))
    joblib.dump(rf_m, os.path.join(models_dir, "rf_model.pkl"))

    all_preds = {
        "Random Forest": rf_m.predict(X_test.values),
        "PyTorch MLP": mlp_preds,
        "Linear Regression": lr_m.predict(X_te_sc),
        "Decision Tree": dt_m.predict(X_test.values)
    }

    print("\n" + "=" * 65)
    print("BẢNG XẾP HẠNG HIỆU SUẤT MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 65)
    print(f"{'Mô hình':<20} | {'MSE':<10} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}")
    print("-" * 65)
    md_lines = ["# BÁO CÁO ĐÁNH GIÁ MÔ HÌNH — BOSTON HOUSING TEST SET\n\n",
                "| Mô hình | MSE | MAE | RMSE | R2 Score |\n|---|---|---|---|---|\n"]
    for name, p in sorted(all_preds.items(), key=lambda x: -r2_score(y_test, x[1])):
        mse = mean_squared_error(y_test, p)
        mae = mean_absolute_error(y_test, p)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, p)
        print(f"{name:<20} | {mse:<10.4f} | {mae:<10.4f} | {rmse:<10.4f} | {r2:<10.4f}")
        md_lines.append(f"| **{name}** | {mse:.4f} | {mae:.4f} | {rmse:.4f} | {r2:.4f} |\n")
    print("=" * 65)

    rep_path = os.path.join(rep_dir, "evaluation_results.md")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    with open(os.path.join(rep_dir, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    print(f"[Artifact] Đã xuất báo cáo Markdown tại: reports/evaluation_results.md")

if __name__ == "__main__":
    run_pipeline()
