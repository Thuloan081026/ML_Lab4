import os, sys, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def prepare_data():
    raw_path = os.path.join(ROOT, "data", "raw", "Boston-house-price-data.csv")
    proc_dir = os.path.join(ROOT, "data", "processed")
    feat_dir = os.path.join(ROOT, "data", "features")
    models_dir = os.path.join(ROOT, "models")
    for d in [proc_dir, feat_dir, models_dir]: os.makedirs(d, exist_ok=True)

    print("[Data Preparation] Dang nap du lieu tu:", raw_path)
    df = pd.read_csv(raw_path, na_values=["", " ", "nan", "NaN", "null", "NULL"])
    df = df.replace(r"^\s*$", np.nan, regex=True)
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
            if info["transform"] == "reflect_log":
                d[col] = np.log1p(info["max_val"] - d[col])
            else:
                d[col] = np.log1p(d[col])

    df_tr.to_csv(os.path.join(proc_dir, "Boston-train-clean.csv"), index=False)
    df_va.to_csv(os.path.join(proc_dir, "Boston-val-clean.csv"), index=False)
    df_te.to_csv(os.path.join(proc_dir, "Boston-test-clean.csv"), index=False)

    mu, sigma = df_tr[features].mean(), df_tr[features].std()
    X_tr_sc = ((df_tr[features] - mu) / sigma).values
    X_va_sc = ((df_va[features] - mu) / sigma).values
    X_te_sc = ((df_te[features] - mu) / sigma).values

    prep_meta = {
        "features": features, "skew_info": skew_info,
        "mean": mu.to_dict(), "std": sigma.to_dict()
    }
    with open(os.path.join(models_dir, "preprocessing_meta.json"), "w", encoding="utf-8") as f:
        json.dump(prep_meta, f, indent=2)

    np.save(os.path.join(feat_dir, "X_train_scaled.npy"), X_tr_sc)
    np.save(os.path.join(feat_dir, "X_val_scaled.npy"), X_va_sc)
    np.save(os.path.join(feat_dir, "X_test_scaled.npy"), X_te_sc)
    np.save(os.path.join(feat_dir, "y_train.npy"), df_tr["MEDV"].values)
    np.save(os.path.join(feat_dir, "y_val.npy"), df_va["MEDV"].values)
    np.save(os.path.join(feat_dir, "y_test.npy"), df_te["MEDV"].values)
    print(f"[Data Preparation] Thanh cong! Da luu tap Train ({len(df_tr)}), Val ({len(df_va)}), Test ({len(df_te)})")

if __name__ == "__main__":
    prepare_data()

