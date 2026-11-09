import os, sys
from flask import Flask, request, jsonify

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from predict import predict_house, load_meta

app = Flask(__name__)
_meta = None

def get_meta():
    global _meta
    if _meta is None:
        _meta = load_meta()
    return _meta

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "project": "Boston Housing Price Prediction API",
        "models": ["mlp (PyTorch)", "rf (Random Forest)", "linear (Linear Regression)", "tree (Decision Tree)"],
        "endpoints": {
            "GET /": "API Information",
            "GET /health": "Health Check",
            "POST /predict": "Predict House Median Value (MEDV)"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    try:
        get_meta()
        return jsonify({"status": "healthy", "service": "Boston Housing Price API"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body phai la JSON"}), 400

    model_type = data.pop("model", "mlp").lower()
    if model_type not in ["mlp", "rf", "linear", "tree"]:
        return jsonify({"error": f"Model khong hop le: {model_type}. Chon: mlp, rf, linear, tree"}), 400

    try:
        medv_pred = predict_house(data, model_type=model_type)
        return jsonify({
            "status": "success",
            "model_used": model_type,
            "predicted_medv_k": round(medv_pred, 2),
            "predicted_price_usd": round(medv_pred * 1000, 2),
            "unit": "$1,000s"
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)

