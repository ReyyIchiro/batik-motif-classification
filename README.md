# 🎨 Batik Motif Classification

Sistem klasifikasi motif batik menggunakan ekstraksi fitur tekstur **Gray Level Co-occurrence Matrix (GLCM)** dan algoritma **Support Vector Machine (SVM)**, diimplementasikan sebagai aplikasi web interaktif berbasis Flask.

## 📋 Deskripsi

Project ini merupakan Tugas Akhir UAP (Ujian Akhir Praktikum) mata kuliah **Pengolahan Citra Digital**. Sistem mampu mengklasifikasikan tiga motif batik Indonesia secara otomatis berdasarkan analisis tekstur citra.

**Motif yang didukung:**
- **Insang** — Pola susunan sisik ikan berulang dengan tekstur diagonal halus
- **Kawung** — Pola lingkaran geometris simetris beraturan
- **Parang** — Pola garis diagonal tegas berulang

## 🔬 Metode

```
Input Citra Batik
      ↓
Preprocessing (resize 128×128, grayscale)
      ↓
Ekstraksi Fitur GLCM
(5 properti × mean+std × 3 jarak × 4 sudut → 10 fitur)
      ↓
Normalisasi (StandardScaler)
      ↓
Klasifikasi SVM (kernel RBF, C=10)
      ↓
Output: Label Motif + Confidence Score
```

### Fitur GLCM yang Diekstrak

| Fitur | Deskripsi |
|||
| Contrast | Perbedaan intensitas antar piksel tetangga |
| Dissimilarity | Variasi lokal tekstur |
| Homogeneity | Keseragaman distribusi intensitas |
| Energy | Keseragaman tekstur (ASM) |
| Correlation | Hubungan linear antar piksel |

Dihitung pada **4 sudut** (0°, 45°, 90°, 135°) dan **3 jarak** (1, 2, 3 piksel), diambil rata-rata dan standar deviasi → **10 fitur total**.



## 📊 Hasil Evaluasi

| Kelas | Precision | Recall | F1-Score | Support |
||||||
| Insang | 0.81 | 0.79 | 0.80 | 28 |
| Kawung | 0.76 | 0.72 | 0.74 | 57 |
| Parang | 0.64 | 0.69 | 0.66 | 54 |
| **Accuracy** | | | **0.72** | **139** |

**Cross-validation (5-fold): 71.51% ± 2.64%**

## 🗂️ Struktur Project

```
batik-motif-classification/
├── app.py                  # Flask web application
├── train_model.py          # Script training model
├── requirements.txt        # Python dependencies
├── README.md
├── templates/
│   └── index.html          # UI aplikasi web
├── model/                  # Dibuat otomatis setelah training
│   ├── batik_svm.pkl
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   └── class_names.pkl
└── dataset/                # Isi dengan dataset (tidak di-upload)
    ├── Insang/
    ├── Kawung/
    └── Parang/
```

## 📦 Dataset

Menggunakan **BatikSnap Dataset** dari Kaggle:
> [syahdanputra/batiksnap-dataset](https://www.kaggle.com/datasets/syahdanputra/batiksnap-dataset)

| Kelas | Total | Train (80%) | Test (20%) |
|||||
| Insang | 138 | 110 | 28 |
| Kawung | 288 | 231 | 57 |
| Parang | 269 | 215 | 54 |
| **Total** | **695** | **556** | **139** |

## 🚀 Cara Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/ReyyIchiro/batik-motif-classification.git
cd batik-motif-classification
```

### 2. Setup environment

```bash
# Buat virtual environment
python -m venv .venv --system-site-packages
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Siapkan dataset

Download dataset dari Kaggle, lalu susun folder seperti berikut:

```
dataset/
├── Insang/    ← foto motif insang (.jpg/.png)
├── Kawung/    ← foto motif kawung
└── Parang/    ← foto motif parang
```

### 4. Training model

```bash
python train_model.py
```

Output yang diharapkan:
```
[✓] Total data: 695 dari 3 kelas
Akurasi pada data uji : 71.94%
Cross-validation (5-fold): 71.51% ± 2.64%
[✓] Model disimpan ke model/batik_svm.pkl
```

### 5. Jalankan aplikasi

```bash
python app.py
```

Buka browser: **http://127.0.0.1:5000**



## 🛠️ Tech Stack

| Komponen | Library |
|||
| Web Framework | Flask |
| Image Processing | Pillow, scikit-image |
| Machine Learning | scikit-learn (SVM) |
| Feature Extraction | GLCM (skimage.feature) |
| Model Serialization | joblib |

## 💡 Cara Penggunaan Aplikasi

1. Buka `http://127.0.0.1:5000` di browser
2. Upload foto motif batik (JPG/PNG, maks. 5MB)
3. Klik **Analisis Citra**
4. Sistem akan menampilkan:
   - Label motif yang terklasifikasi
   - Confidence score (%)
   - Nilai 5 fitur GLCM hasil ekstraksi

## 📝 Catatan

- Model dilatih dengan rasio split **80% train / 20% test** (stratified)
- Preprocessing: resize ke **128×128 piksel**, konversi ke **grayscale**
- Fitur dinormalisasi menggunakan **StandardScaler** sebelum klasifikasi
- SVM dengan kernel RBF dipilih karena fitur GLCM tidak selalu terpisah secara linear
