# ==============================================================
# LEMBAR KERJA PERTEMUAN 4 — DATA PREPARATION
# Topik       : Collection, Cleaning, EDA, Feature Engineering, Splitting
# ==============================================================

# Langkah 1 — Import Library
# Library yang digunakan untuk data handling, visualisasi, dan splitting
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==============================================================
# Langkah 2 — COLLECTION
# ==============================================================
# Membaca dataset CSV yang sudah dibuat
df = pd.read_csv("kelulusan_mahasiswa.csv")

# Menampilkan informasi awal dataset
print("=== Informasi Dataset ===")
print(df.info())  # menunjukkan tipe data dan jumlah data non-null
print("\n=== Lima Data Pertama ===")
print(df.head())  # menampilkan 5 baris pertama

# ==============================================================
# Langkah 3 — CLEANING
# ==============================================================
print("\n=== Pengecekan Missing Value ===")
print(df.isnull().sum())  # memastikan tidak ada nilai kosong

# Menghapus data duplikat jika ada
df = df.drop_duplicates()

# Visualisasi boxplot untuk mendeteksi outlier pada kolom IPK
# Outlier adalah data yang terlalu jauh dari rata-rata
sns.boxplot(x=df['IPK'])
plt.title("Boxplot IPK untuk Deteksi Outlier")
plt.show()

# ==============================================================
# Langkah 4 — EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================
print("\n=== Statistik Deskriptif Dataset ===")
print(df.describe())  # menampilkan ringkasan statistik (mean, std, min, max, dll.)

# Histogram distribusi nilai IPK
sns.histplot(df['IPK'], bins=10, kde=True)
plt.title("Distribusi Nilai IPK")
plt.xlabel("IPK")
plt.ylabel("Frekuensi")
plt.show()

# Scatterplot antara IPK dan Waktu Belajar
# Warna (hue) menunjukkan status kelulusan (1 = Lulus, 0 = Tidak)
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus', palette='coolwarm')
plt.title("Hubungan IPK dan Waktu Belajar terhadap Kelulusan")
plt.show()

# Heatmap korelasi antar fitur numerik
# Korelasi menunjukkan hubungan antar variabel (semakin mendekati 1, semakin kuat)
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap Korelasi Antar Variabel")
plt.show()

# ==============================================================
# Langkah 5 — FEATURE ENGINEERING
# ==============================================================
# Membuat fitur baru dari kombinasi data yang sudah ada

# Rasio_Absensi = perbandingan absensi terhadap total 14 pertemuan
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14

# IPK_x_Study = hasil kali antara IPK dan waktu belajar
# Fitur ini bisa menunjukkan seberapa besar "usaha" mahasiswa terhadap prestasi
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

# Menyimpan dataset hasil olahan ke file baru
df.to_csv("processed_kelulusan.csv", index=False)
print("\nDataset hasil olahan disimpan sebagai 'processed_kelulusan.csv'")

# ==============================================================
# Langkah 6 — SPLITTING DATASET
# ==============================================================
# Tujuan: membagi data menjadi Train, Validation, dan Test

# Pisahkan fitur (X) dan target (y)
X = df.drop('Lulus', axis=1)  # semua kolom kecuali 'Lulus'
y = df['Lulus']               # kolom target

# Stratified split memastikan proporsi kelas tetap seimbang di setiap subset
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Dari 30% sisa, setengah untuk validation dan setengah untuk test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Menampilkan ukuran masing-masing subset data
print("\n=== Ukuran Dataset Setelah Splitting ===")
print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)

# ==============================================================
# SELESAI
# ==============================================================
# Hasil akhir:
# 1. File processed_kelulusan.csv berisi data bersih dan fitur baru
# 2. Statistik dan visualisasi EDA tampil di jendela plot
# 3. Data terbagi menjadi train, validation, dan test
# ==============================================================

