# ==============================================================
# Langkah 7 — (Opsional) ENDPOINT FLASK UNTUK INFERENCE
# ==============================================================

from flask import Flask, request, jsonify
import joblib, pandas as pd

app = Flask(__name__)
MODEL = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    X = pd.DataFrame([data])
    yhat = MODEL.predict(X)[0]
    proba = None
    if hasattr(MODEL, "predict_proba"):
        proba = float(MODEL.predict_proba(X)[:,1][0])
    return jsonify({"prediction": int(yhat), "proba": proba})

if __name__ == "__main__":
    app.run(port=5000)


# ==============================================================
# SELESAI
# ==============================================================
# Hasil Akhir:
# ✅ Baseline Model (Logistic Regression)
# ✅ Model Alternatif (Random Forest)
# ✅ GridSearchCV dengan tuning parameter
# ✅ Evaluasi akhir (F1, ROC-AUC, Confusion Matrix)
# ✅ Model tersimpan (model.pkl)
# ==============================================================
