# lembar_kerja_6_random_forest.py
# ==============================================================
# LEMBAR KERJA PERTEMUAN 6 — RANDOM FOREST UNTUK KLASIFIKASI
# Tujuan: bangun, tune, dan evaluasi Random Forest; simpan model siap pakai
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==============================================================
# Langkah 1 — Muat Data
# ==============================================================
df_path = "processed_kelulusan.csv"
if not os.path.exists(df_path):
    raise FileNotFoundError(f"{df_path} tidak ditemukan. Jalankan Lembar Kerja 4 terlebih dahulu.")

df = pd.read_csv(df_path)
print("Data awal shape:", df.shape)
print(df.head())

# Jika dataset kecil, gandakan (duplicate) secara proporsional agar stratify split tidak error.
# Ini hanya untuk keperluan eksperimen/tugas — cara terbaik adalah mengumpulkan lebih banyak data.
if len(df) < 50:
    print("\nDataset kecil (<50). Menggandakan seluruh dataset secara ringan (x10) untuk stabilitas split.")
    df = pd.concat([df] * 10, ignore_index=True)
    print("Shape setelah duplikasi:", df.shape)

# Periksa distribusi kelas
print("\nDistribusi kelas (Lulus):")
print(df["Lulus"].value_counts())

# ==============================================================
# Langkah 1b — Siapkan X dan y; split 70/15/15 stratified
# ==============================================================
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

print("\nUkuran dataset setelah split:")
print("X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)

# ==============================================================
# Langkah 2 — Pipeline & Baseline Random Forest
# ==============================================================
# Tentukan kolom numerik (di dataset ini seluruh fitur kecuali target seharusnya numerik)
num_cols = X_train.select_dtypes(include="number").columns.tolist()
print("\nKolom numerik:", num_cols)

# Preprocessing: imputasi median + standard scaling
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols)
], remainder="drop")

# Baseline RandomForest
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)

pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", rf)
])

# Latih baseline
print("\nMelatih baseline RandomForest...")
pipe.fit(X_train, y_train)

# Evaluasi pada validation set
y_val_pred = pipe.predict(X_val)
print("\n=== Baseline Random Forest (Validation) ===")
print("F1 (macro):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# ==============================================================
# Langkah 3 — Validasi Silang (Cross-Validation)
# ==============================================================
print("\n=== Cross-Validation (5-fold stratified) di data train ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
print("CV F1-macro (train): %.4f ± %.4f" % (scores.mean(), scores.std()))

# ==============================================================
# Langkah 4 — Tuning Ringkas (GridSearch)
# ==============================================================
print("\n=== GridSearchCV untuk tuning hyperparameter ===")
param_grid = {
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(pipe, param_grid=param_grid, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)

gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
print("Best CV F1 (train):", gs.best_score_)

best_model = gs.best_estimator_

# Evaluasi model terbaik pada validation set
y_val_best = best_model.predict(X_val)
print("\n=== Best RF (Validation) ===")
print("F1 (macro):", f1_score(y_val, y_val_best, average="macro"))
print(classification_report(y_val, y_val_best, digits=3))

# ==============================================================
# Langkah 5 — Evaluasi Akhir (Test Set)
# ==============================================================
final_model = best_model  # pilih model terbaik hasil GridSearch
y_test_pred = final_model.predict(X_test)

print("\n=== Evaluasi Akhir (Test Set) ===")
print("F1 (test, macro):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# ROC-AUC dan plot ROC/PR (jika predict_proba tersedia)
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, y_test_proba)
        print("ROC-AUC (test):", auc)
    except Exception as e:
        print("Tidak dapat menghitung ROC-AUC:", e)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})" if 'auc' in locals() else "ROC")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_test.png", dpi=120)
    print("ROC curve disimpan -> roc_test.png")

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test Set)")
    plt.tight_layout()
    plt.savefig("pr_test.png", dpi=120)
    print("PR curve disimpan -> pr_test.png")

# ==============================================================
# Langkah 6 — Pentingnya Fitur (Feature Importance)
# ==============================================================
print("\n=== Feature Importance (native RF) ===")
try:
    # Ekstrak feature names setelah preprocessing
    # Cara umum: jika preprocessor punya get_feature_names_out
    try:
        fn = final_model.named_steps["pre"].get_feature_names_out()
    except Exception:
        # fallback: gunakan num_cols langsung (preprocessing hanya numerical)
        fn = num_cols

    importances = final_model.named_steps["clf"].feature_importances_
    feat_imp = sorted(zip(fn, importances), key=lambda x: x[1], reverse=True)
    for name, val in feat_imp[:10]:
        print(f"{name}: {val:.4f}")

    # Visualisasi top features
    names = [n for n, _ in feat_imp]
    vals = [v for _, v in feat_imp]
    plt.figure(figsize=(8, max(4, len(names)*0.3)))
    sns.barplot(x=vals, y=names)
    plt.title("Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=120)
    print("Feature importances disimpan -> feature_importances.png")
except Exception as e:
    print("Feature importance tidak tersedia:", e)

# (Opsional) Permutation importance: uncomment jika ingin jalankan (itu lebih mahal komputasi)
# from sklearn.inspection import permutation_importance
# r = permutation_importance(final_model, X_val, y_val, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
# perm_imp = sorted(zip(X_val.columns, r.importances_mean), key=lambda x: x[1], reverse=True)
# print("Top permutation importances:", perm_imp[:10])

# ==============================================================
# Langkah 7 — Simpan Model
# ==============================================================
rf_path = "rf_model.pkl"
joblib.dump(final_model, rf_path)
print(f"\nModel Random Forest disimpan -> {rf_path}")

# ==============================================================
# Langkah 8 — Cek Inference Lokal (contoh)
# ==============================================================
print("\nContoh inference lokal (sample):")
sample = pd.DataFrame([{
    "IPK": 3.4,
    "Jumlah_Absensi": 4,
    "Waktu_Belajar_Jam": 7,
    "Rasio_Absensi": 4/14,
    "IPK_x_Study": 3.4*7
}])

# Pastikan kolom sample sesuai urutan/kolom X
sample_pred = final_model.predict(sample)[0]
print("Sample prediction:", int(sample_pred))

# ==============================================================
# SELESAI
# ==============================================================
print("\nSelesai. File output yang dibuat (jika tersedia):")
print(" - rf_model.pkl")
print(" - roc_test.png (jika ada predict_proba)")
print(" - pr_test.png (jika ada predict_proba)")
print(" - feature_importances.png")
