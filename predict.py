import os, sys, json, argparse, zipfile
import numpy as np, pandas as pd, joblib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_meta():
    meta_path = os.path.join(MODELS_DIR, "preprocessing_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess_sample(raw_dict, meta):
    features = meta["features"]
    sample_df = pd.DataFrame([{col: float(raw_dict.get(col, 0.0)) for col in features}])
    for col, info in meta["skew_info"].items():
        if info["transform"] == "reflect_log":
            sample_df[col] = np.log1p(info["max_val"] - sample_df[col])
        else:
            sample_df[col] = np.log1p(sample_df[col])
    unscaled = sample_df.values
    scaled = np.zeros_like(unscaled, dtype=np.float32)
    for i, col in enumerate(features):
        mu = meta["mean"][col]
        sigma = meta["std"][col]
        scaled[0, i] = (unscaled[0, i] - mu) / sigma
    return unscaled, scaled

def predict_mlp(scaled_feat):
    mlp_pth = os.path.join(MODELS_DIR, "best_mlp_boston.pth")
    with zipfile.ZipFile(mlp_pth, "r") as z:
        w1 = np.frombuffer(z.read("best_mlp_boston/data/0"), dtype=np.float32).reshape(64, 13)
        b1 = np.frombuffer(z.read("best_mlp_boston/data/1"), dtype=np.float32)
        w2 = np.frombuffer(z.read("best_mlp_boston/data/2"), dtype=np.float32).reshape(32, 64)
        b2 = np.frombuffer(z.read("best_mlp_boston/data/3"), dtype=np.float32)
        w3 = np.frombuffer(z.read("best_mlp_boston/data/4"), dtype=np.float32).reshape(1, 32)
        b3 = np.frombuffer(z.read("best_mlp_boston/data/5"), dtype=np.float32)
    z1 = np.maximum(0, scaled_feat @ w1.T + b1)
    z2 = np.maximum(0, z1 @ w2.T + b2)
    return float((z2 @ w3.T + b3)[0, 0])

def predict_house(raw_dict, model_type="mlp"):
    meta = load_meta()
    unscaled, scaled = preprocess_sample(raw_dict, meta)
    if model_type == "mlp":
        pred = predict_mlp(scaled)
    elif model_type in ["rf", "tree"]:
        m_file = "rf_model.pkl" if model_type == "rf" else "tree_model.pkl"
        model = joblib.load(os.path.join(MODELS_DIR, m_file))
        pred = float(model.predict(unscaled)[0])
    elif model_type == "linear":
        model = joblib.load(os.path.join(MODELS_DIR, "linear_model.pkl"))
        pred = float(model.predict(scaled)[0])
    else:
        raise ValueError(f"Khong ho tro model: {model_type}")
    return max(0.0, pred)

def run_demo():
    print("\n" + "=" * 68)
    print("DEMO SUY LUẬN DỰ ĐOÁN GIÁ NHÀ BOSTON (MEDV - $1,000s)")
    print("=" * 68)
    samples = [
        ("Khu Biệt Thự Cao Cấp (Ngoại Ô)", {
            "CRIM": 0.02, "ZN": 80.0, "INDUS": 2.5, "CHAS": 0.0, "NOX": 0.40,
            "RM": 7.8, "AGE": 25.0, "DIS": 6.5, "RAD": 2.0, "TAX": 220.0,
            "PTRATIO": 14.5, "B": 396.9, "LSTAT": 4.5
        }),
        ("Khu Nhà Đô Thị Trung Bình", {
            "CRIM": 0.25, "ZN": 12.5, "INDUS": 8.0, "CHAS": 0.0, "NOX": 0.52,
            "RM": 6.2, "AGE": 60.0, "DIS": 3.8, "RAD": 5.0, "TAX": 310.0,
            "PTRATIO": 18.0, "B": 385.0, "LSTAT": 12.0
        }),
        ("Khu Nhà Bình Dân / Mật Độ Cao", {
            "CRIM": 8.5, "ZN": 0.0, "INDUS": 18.1, "CHAS": 0.0, "NOX": 0.69,
            "RM": 5.2, "AGE": 95.0, "DIS": 1.8, "RAD": 24.0, "TAX": 666.0,
            "PTRATIO": 20.2, "B": 280.0, "LSTAT": 28.5
        })
    ]
    for name, data in samples:
        print(f"\n[Ngôi nhà]: {name}")
        print(f"  Đặc trưng chính: CRIM={data['CRIM']}, RM={data['RM']} phòng, LSTAT={data['LSTAT']}%, TAX={data['TAX']}")
        p_mlp = predict_house(data, "mlp")
        p_rf = predict_house(data, "rf")
        p_lr = predict_house(data, "linear")
        print(f"  -> PyTorch MLP  : ${p_mlp:.2f}k (${p_mlp * 1000:,.0f} USD)")
        print(f"  -> Random Forest: ${p_rf:.2f}k (${p_rf * 1000:,.0f} USD)")
        print(f"  -> Linear Reg   : ${p_lr:.2f}k (${p_lr * 1000:,.0f} USD)")
    print("\n" + "=" * 68)

def main():
    parser = argparse.ArgumentParser(description="Du doan gia nha Boston")
    parser.add_argument("--demo", action="store_true", help="Chay demo 3 mau ngoi nha tieu bieu")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "rf", "linear", "tree"])
    parser.add_argument("--crim", type=float, default=0.2)
    parser.add_argument("--rm", type=float, default=6.0)
    parser.add_argument("--lstat", type=float, default=12.0)
    parser.add_argument("--tax", type=float, default=300.0)
    args = parser.parse_args()

    if args.demo or len(sys.argv) == 1:
        run_demo()
    else:
        sample = {"CRIM": args.crim, "RM": args.rm, "LSTAT": args.lstat, "TAX": args.tax}
        pred = predict_house(sample, args.model)
        print(f"Gia nha du doan ({args.model}): ${pred:.2f}k (${pred * 1000:,.0f} USD)")

if __name__ == "__main__":
    main()
