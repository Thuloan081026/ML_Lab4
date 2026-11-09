import os, sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import app

def run_demo():
    print("\n" + "=" * 68)
    print("DEMO DỰ ĐOÁN GIÁ NHÀ BOSTON QUA FLASK TEST CLIENT")
    print("=" * 68)

    client = app.test_client()
    scenarios = [
        ("Kịch bản 1: Khu Biệt Thự Ngoại Ô (Nhiều phòng, ít tội phạm)", {
            "CRIM": 0.02, "ZN": 80.0, "INDUS": 2.5, "CHAS": 0.0, "NOX": 0.40,
            "RM": 7.8, "AGE": 25.0, "DIS": 6.5, "RAD": 2.0, "TAX": 220.0,
            "PTRATIO": 14.5, "B": 396.9, "LSTAT": 4.5
        }),
        ("Kịch bản 2: Khu Nhà Đô Thị Trung Bình", {
            "CRIM": 0.25, "ZN": 12.5, "INDUS": 8.0, "CHAS": 0.0, "NOX": 0.52,
            "RM": 6.2, "AGE": 60.0, "DIS": 3.8, "RAD": 5.0, "TAX": 310.0,
            "PTRATIO": 18.0, "B": 385.0, "LSTAT": 12.0
        }),
        ("Kịch bản 3: Khu Bình Dân / Mật Độ Cao", {
            "CRIM": 8.5, "ZN": 0.0, "INDUS": 18.1, "CHAS": 0.0, "NOX": 0.69,
            "RM": 5.2, "AGE": 95.0, "DIS": 1.8, "RAD": 24.0, "TAX": 666.0,
            "PTRATIO": 20.2, "B": 280.0, "LSTAT": 28.5
        })
    ]

    md_lines = ["# KẾT QUẢ THỰC NGHIỆM DEMO — LAB 4 (BOSTON HOUSING)\n\n"]
    for name, payload in scenarios:
        print(f"\n[Kịch bản]: {name}")
        md_lines.append(f"### {name}\n- **Đặc trưng:** CRIM={payload['CRIM']}, RM={payload['RM']} phòng, LSTAT={payload['LSTAT']}%, TAX={payload['TAX']}\n")
        for m in ["mlp", "rf", "linear"]:
            body = dict(payload)
            body["model"] = m
            res = client.post("/predict", json=body)
            data = res.get_json()
            print(f"  -> {m.upper():<6}: ${data['predicted_medv_k']:.2f}k (${data['predicted_price_usd']:,.0f} USD)")
            md_lines.append(f"  - **{m.upper()}**: `${data['predicted_medv_k']:.2f}k` (${data['predicted_price_usd']:,.0f} USD)\n")
        md_lines.append("\n")

    print("\n" + "=" * 68)
    rep_dir = os.path.join(ROOT, "reports")
    os.makedirs(rep_dir, exist_ok=True)
    with open(os.path.join(rep_dir, "demo_results.md"), "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    print(f"[+] Đã xuất kết quả demo tại: reports/demo_results.md")

if __name__ == "__main__":
    run_demo()

