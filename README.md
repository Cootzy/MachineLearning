# 🧠 Machine Learning Projects — Revrico Ramadhino

**Nama:** Revrico Ramadhino  
**NIM:** 231011403294  
**Kelas:** 05 TPLE 013  

---

## 📘 Deskripsi Proyek (ID)
Repositori ini berisi kumpulan tugas dan implementasi *Machine Learning* dari Pertemuan 4 hingga Pertemuan 7.  
Setiap folder berisi kode Python (.py) yang dapat dijalankan di **VSCode** dengan lingkungan virtual (`ml_env`) dan dataset terkait.

**Tujuan utama proyek ini** adalah mempelajari tahapan *data preparation*, *modeling*, dan *evaluasi* model *Machine Learning*, mulai dari model klasik hingga *Artificial Neural Network (ANN)*.

---

## 📘 Project Description (EN)
This repository contains a collection of **Machine Learning assignments and implementations** from Week 4 to Week 7.  
Each folder includes executable Python scripts (.py) for **VSCode** using a virtual environment (`ml_env`) and relevant datasets.

**Main objective:** to learn key stages of *data preparation*, *model building*, and *evaluation*, ranging from classical models to *Artificial Neural Networks (ANN)*.

---

## 📂 Struktur Folder / Folder Structure

MachineLearning/
├── pertemuan_4/ # Data Preparation & Feature Engineering
├── pertemuan_5/ # Model Baseline & Evaluation
├── pertemuan_6/ # Model Deployment (Flask API)
└── pertemuan_7/ # Artificial Neural Network (ANN)

yaml
Copy code

---

## 🧩 Ringkasan Tiap Pertemuan / Summary of Each Meeting

### 📍 **Lembar Kerja 4 — Data Preparation**
- Pengumpulan & pembersihan data (CSV)
- Pembuatan fitur baru (*feature engineering*)
- Pembagian data (train, validation, test)
- Standardisasi menggunakan `StandardScaler`

### 📍 **Lembar Kerja 5 — Model Klasifikasi Dasar**
- Model: Logistic Regression, Random Forest  
- Evaluasi: Confusion Matrix, Accuracy, F1-Score, ROC-AUC  
- Penyimpanan model ke file `model.pkl`

### 📍 **Lembar Kerja 6 — Model Deployment (Flask API)**
- Implementasi endpoint `/predict` menggunakan Flask  
- Input data via JSON  
- Menjalankan server lokal dengan `app.py`

### 📍 **Lembar Kerja 7 — Artificial Neural Network (ANN)**
- Model ANN untuk klasifikasi biner  
- Layer: Dense, Dropout, Aktivasi ReLU & Sigmoid  
- Callback: EarlyStopping  
- Evaluasi: Accuracy, AUC, Confusion Matrix, Learning Curve

---

## ⚙️ Cara Menjalankan / How to Run

### 1️⃣ Buat virtual environment
```bash
python -m venv ml_env
2️⃣ Aktifkan environment
Windows:

bash
ml_env\Scripts\activate
Mac/Linux:

bash
source ml_env/bin/activate
3️⃣ Instal dependensi
bash
pip install -r requirements.txt
4️⃣ Jalankan file Python
bash
python pertemuan_7/lembar_kerja_7.py
📦 Dependencies
Python 3.10+

pandas

numpy

scikit-learn

tensorflow / keras

flask

matplotlib

Semua paket dapat diinstal otomatis melalui:

bash
pip install -r requirements.txt
🧠 Author
Revrico Ramadhino
💼 NIM: 231011403294
🎓 Kelas: 05 TPLE 013
📍 Universitas Pamulang

🏷️ License
This project is licensed under the MIT License — feel free to use, modify, and distribute.
