# ==============================================================
# Lembar Kerja 7 — Artificial Neural Network (ANN) untuk Klasifikasi
# ==============================================================
# Tujuan:
#  - Melatih model klasifikasi biner menggunakan ANN
#  - Menggunakan dataset processed_kelulusan.csv
#  - Menampilkan evaluasi dan learning curve
# ==============================================================

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Langkah 1 — Siapkan Data
# --------------------------------------------------------------
# Baca dataset hasil Lembar Kerja 4
df = pd.read_csv("processed_kelulusan.csv")

# Pisahkan fitur dan target
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Standarisasi fitur agar ANN lebih cepat konvergen
sc = StandardScaler()
Xs = sc.fit_transform(X)

# Split data menjadi train, validation, dan test
# (Non-stratified karena dataset kecil)
X_train, X_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Data shape (train, val, test):")
print(X_train.shape, X_val.shape, X_test.shape)

# --------------------------------------------------------------
# Langkah 2 — Bangun Model ANN
# --------------------------------------------------------------
tf.random.set_seed(42)  # Seed untuk reproducibility

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # jumlah fitur
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # Output biner (0/1)
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", "AUC"]
)

model.summary()

# --------------------------------------------------------------
# Langkah 3 — Training dengan Early Stopping
# --------------------------------------------------------------
es = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=4,
    callbacks=[es],
    verbose=1
)

# --------------------------------------------------------------
# Langkah 4 — Evaluasi di Test Set
# --------------------------------------------------------------
loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print("\nEvaluasi Model di Test Set")
print("===========================")
print(f"Test Accuracy : {acc:.3f}")
print(f"Test AUC      : {auc:.3f}")

# Prediksi probabilitas dan ubah ke kelas (0/1)
y_proba = model.predict(X_test).ravel()
y_pred = (y_proba >= 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

# --------------------------------------------------------------
# Langkah 5 — Visualisasi Learning Curve
# --------------------------------------------------------------
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Learning Curve")
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=120)
plt.show()

print("\n✅ Training selesai. Learning curve disimpan sebagai 'learning_curve.png'.")
