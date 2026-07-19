# Klasifikasi Motif Batik

Sistem machine learning untuk mengklasifikasikan motif batik menggunakan ekstraksi fitur tekstur Gray Level Co-occurrence Matrix (GLCM) dan algoritma Support Vector Machine (SVM). Proyek ini diimplementasikan sebagai aplikasi web interaktif berbasis Flask.

## Deskripsi

Proyek ini dikembangkan sebagai Tugas Akhir Ujian Akhir Praktikum (UAP) mata kuliah Pengolahan Citra Digital. Sistem mampu mengklasifikasikan tiga motif batik Indonesia secara otomatis berdasarkan analisis tekstur citra.

**Motif yang didukung:**
- **Insang** — Pola susunan sisik ikan berulang dengan tekstur diagonal halus.
- **Kawung** — Pola lingkaran geometris simetris beraturan.
- **Parang** — Pola garis diagonal tegas berulang.

## Metodologi

### Pipeline
1. **Input Citra:** Citra batik
2. **Preprocessing:** Resize ke 128×128 piksel, konversi ke grayscale
3. **Ekstraksi Fitur:** GLCM (5 properti × mean & std × 3 jarak × 4 sudut → 10 fitur)
4. **Normalisasi:** StandardScaler
5. **Klasifikasi:** SVM (kernel RBF, C=10)
6. **Output:** Label Motif Prediksi + Confidence Score

### Fitur GLCM yang Diekstrak

| Fitur | Deskripsi |
|---|---|
| Contrast | Perbedaan intensitas antar piksel tetangga |
| Dissimilarity | Variasi lokal tekstur |
| Homogeneity | Keseragaman distribusi intensitas |
| Energy | Keseragaman tekstur (ASM) |
| Correlation | Hubungan linear antar piksel |

Fitur dihitung pada 4 sudut (0°, 45°, 90°, 135°) dan 3 jarak (1, 2, 3 piksel). Rata-rata dan standar deviasi kemudian diekstrak, menghasilkan total 10 fitur.

## Hasil Evaluasi

| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Insang | 0.81 | 0.79 | 0.80 | 28 |
| Kawung | 0.76 | 0.72 | 0.74 | 57 |
| Parang | 0.64 | 0.69 | 0.66 | 54 |
| **Accuracy** | | | **0.72** | **139** |

**Cross-validation (5-fold): 71.51% ± 2.64%**

## Struktur Proyek

```text
batik-motif-classification/
├── app.py                  # Aplikasi web Flask
├── train_model.py          # Skrip pelatihan model
├── requirements.txt        # Dependensi Python
├── README.md               # Dokumentasi
├── templates/
│   └── index.html          # Template antarmuka web
├── model/                  # Dibuat otomatis setelah pelatihan
│   ├── batik_svm.pkl
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   └── class_names.pkl
└── dataset/                # Direktori dataset (tidak masuk version control)
    ├── Insang/
    ├── Kawung/
    └── Parang/
```

## Dataset

Proyek ini menggunakan **BatikSnap Dataset** dari Kaggle:
> [syahdanputra/batiksnap-dataset](https://www.kaggle.com/datasets/syahdanputra/batiksnap-dataset)

| Kelas | Total | Train (80%) | Test (20%) |
|---|---|---|---|
| Insang | 138 | 110 | 28 |
| Kawung | 288 | 231 | 57 |
| Parang | 269 | 215 | 54 |
| **Total** | **695** | **556** | **139** |

## Panduan Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/ReyyIchiro/batik-motif-classification.git
cd batik-motif-classification
```

### 2. Setup Environment

```bash
# Buat virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependensi
pip install -r requirements.txt
```

### 3. Persiapan Dataset

Unduh dataset dari Kaggle dan susun struktur direktori sebagai berikut:

```text
dataset/
├── Insang/    ← citra motif insang (.jpg/.png)
├── Kawung/    ← citra motif kawung
└── Parang/    ← citra motif parang
```

### 4. Pelatihan Model

```bash
python train_model.py
```

### 5. Menjalankan Aplikasi (Development)

```bash
python app.py
```
Akses aplikasi melalui peramban web di `http://127.0.0.1:5000`.

### 6. Deployment Production

Untuk lingkungan production, disarankan menggunakan server WSGI standar produksi seperti Gunicorn daripada menggunakan server pengembangan bawaan Flask.

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Tech Stack

| Komponen | Teknologi |
|---|---|
| Web Framework | Flask |
| Pemrosesan Citra | Pillow, scikit-image |
| Machine Learning | scikit-learn (SVM) |
| Ekstraksi Fitur | GLCM (skimage.feature) |
| Serialisasi Model | joblib |

## Petunjuk Penggunaan

1. Buka aplikasi di peramban web.
2. Unggah citra motif batik (JPG/PNG, maksimal 5MB).
3. Klik tombol "Analisis Citra".
4. Sistem akan menampilkan prediksi motif, nilai confidence score, dan fitur GLCM yang diekstrak.

## Catatan Tambahan
- Model dilatih dengan rasio pemisahan (split) 80/20 (stratified).
- Preprocessing mengubah ukuran citra menjadi 128×128 piksel dan mengkonversinya ke skala abu-abu (grayscale).
- Fitur dinormalisasi menggunakan StandardScaler sebelum tahap klasifikasi.
- SVM dengan kernel RBF digunakan karena fitur tekstur GLCM tidak selalu terpisah secara linear.
