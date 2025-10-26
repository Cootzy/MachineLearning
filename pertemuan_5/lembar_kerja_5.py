# ==============================================================
# LEMBAR KERJA PERTEMUAN 5 — MODELING
# Mata Kuliah : Machine Learning
# Topik       : Selection • Training • Validation • Testing
# ==============================================================
# Catatan: Dataset kecil otomatis diperbanyak agar tidak error saat stratify split
# ==============================================================

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import joblib

# ==============================================================
# Langkah 1 — MUAT DATA
# ==============================================================
# Membaca dataset hasil pertemuan 4
df = pd.read_csv("processed_kelulusan.csv")

# Periksa distribusi kelas
print("\nDistribusi awal kelas:")
print(df["Lulus"].value_counts())

# Jika jumlah data < 50 baris, gandakan dataset agar cukup untuk stratify split
if len(df) < 50:
    print("\nDataset terlalu kecil, menduplikasi data agar tidak error saat stratify...")
    df = pd.concat([df] * 10, ignore_index=True)

# Periksa ulang distribusi kelas setelah duplikasi
print("\nDistribusi setelah duplikasi:")
print(df["Lulus"].value_counts())

# Memisahkan fitur (X) dan target (y)
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Membagi data menjadi Train (70%), Validation (15%), dan Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("\n=== Ukuran Dataset ===")
print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)

# ==============================================================
# Langkah 2 — BASELINE MODEL (Logistic Regression)
# ==============================================================
# Model dasar: Logistic Regression untuk klasifikasi biner

# Pilih kolom numerik (semua kolom fitur)
num_cols = X_train.select_dtypes(include="number").columns

# Preprocessing pipeline:
#  - SimpleImputer isi nilai kosong (jika ada) dengan median
#  - StandardScaler untuk normalisasi skala fitur numerik
pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols),
], remainder="drop")

# Model baseline Logistic Regression
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

# Gabungkan preprocessing + model ke dalam pipeline
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)])

# Latih model baseline
pipe_lr.fit(X_train, y_train)

# Prediksi data validation
y_val_pred = pipe_lr.predict(X_val)

# Evaluasi model baseline
print("\n=== BASELINE MODEL (Logistic Regression) ===")
print("F1 Score (Validation):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# ==============================================================
# Langkah 3 — MODEL ALTERNATIF (Random Forest)
# ==============================================================
# Random Forest cenderung lebih kuat untuk data kecil dan non-linear
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42
)
pipe_rf = Pipeline([("pre", pre), ("clf", rf)])

# Latih model Random Forest
pipe_rf.fit(X_train, y_train)

# Prediksi pada data validation
y_val_rf = pipe_rf.predict(X_val)

# Evaluasi performa awal Random Forest
print("\n=== RANDOM FOREST MODEL ===")
print("F1 Score (Validation):", f1_score(y_val, y_val_rf, average="macro"))

# ==============================================================
# Langkah 4 — VALIDASI SILANG & TUNING HIPERPARAMETER
# ==============================================================
print("\n=== GRID SEARCH RANDOM FOREST ===")

# Gunakan StratifiedKFold agar pembagian data mempertahankan proporsi kelas
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Parameter yang diuji dalam GridSearchCV
param = {
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}

# GridSearchCV mencari kombinasi parameter terbaik berdasarkan F1 makro
gs = GridSearchCV(
    pipe_rf,
    param_grid=param,
    cv=skf,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)

# Jalankan tuning hyperparameter
gs.fit(X_train, y_train)
print("Best Params:", gs.best_params_)
print("Best CV F1 Score:", gs.best_score_)

# Ambil model terbaik hasil tuning
best_rf = gs.best_estimator_

# Evaluasi model terbaik di data validation
y_val_best = best_rf.predict(X_val)
print("Best RF F1 (Validation):", f1_score(y_val, y_val_best, average="macro"))

# ==============================================================
# Langkah 5 — EVALUASI AKHIR (TEST SET)
# ==============================================================
final_model = best_rf  # gunakan model terbaik hasil tuning
y_test_pred = final_model.predict(X_test)

print("\n=== EVALUASI AKHIR (TEST SET) ===")
print("F1 Score (Test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))

# Jika model mendukung probabilitas, tampilkan ROC curve
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]

    # Hitung ROC-AUC Score
    try:
        print("ROC-AUC (Test):", roc_auc_score(y_test, y_test_proba))
    except:
        pass

    # Plot kurva ROC
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_test.png", dpi=120)
    plt.show()

# ==============================================================
# Langkah 6 — SIMPAN MODEL (Opsional)
# ==============================================================
# Simpan model terbaik ke file agar bisa digunakan untuk prediksi baru
joblib.dump(final_model, "model.pkl")
print("\nModel terbaik disimpan ke file: model.pkl")

# ==============================================================
# Langkah 7 — (Opsional) API FLASK UNTUK INFERENCE
# ==============================================================
"""
Contoh file Flask (save sebagai app.py):

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
"""
# ==============================================================
# SELESAI ✅
# ==============================================================
# Hasil Akhir:
# ✅ Baseline Model (Logistic Regression)
# ✅ Model Alternatif (Random Forest)
# ✅ GridSearchCV untuk tuning parameter
# ✅ Evaluasi akhir (F1, ROC-AUC, Confusion Matrix)
# ✅ Model tersimpan (model.pkl)
# ==============================================================
