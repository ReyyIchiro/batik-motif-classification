"""
train_model.py
--------------
Training model klasifikasi batik GLCM + SVM.
SVM lebih baik dari KNN untuk fitur tekstur yang overlap.

Cara pakai:
    python train_model.py
"""

import os
import glob
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

ACTIVE_CLASSES = ['Insang', 'Kawung', 'Parang']


def extract_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))

    gray = rgb2gray(np.array(img))
    gray_uint8 = img_as_ubyte(gray)

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray_uint8, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)

    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = []
    for prop in props:
        vals = graycoprops(glcm, prop).flatten()
        features.append(vals.mean())
        features.append(vals.std())

    return features


def load_dataset(dataset_path):
    print(f"\n[*] Memuat dataset dari: {dataset_path}")
    X, y_labels = [], []
    class_names = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d)) and d in ACTIVE_CLASSES
    ])

    if not class_names:
        raise ValueError("Tidak ada subfolder kelas ditemukan!")

    for cls in class_names:
        folder = os.path.join(dataset_path, cls)
        images = (glob.glob(os.path.join(folder, '*.jpg')) +
                  glob.glob(os.path.join(folder, '*.jpeg')) +
                  glob.glob(os.path.join(folder, '*.png')))

        print(f"    [{cls}]: {len(images)} gambar", end='', flush=True)
        ok = 0
        for img_path in images:
            try:
                feats = extract_features(img_path)
                X.append(feats)
                y_labels.append(cls)
                ok += 1
            except:
                pass

        print(f" → {ok} berhasil diproses")

    print(f"\n[✓] Total data: {len(X)} dari {len(class_names)} kelas\n")
    return np.array(X), np.array(y_labels), class_names


def main():
    X, y, class_names = load_dataset('dataset')

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"Data latih : {len(X_train)} sampel")
    print(f"Data uji   : {len(X_test)} sampel")
    print(f"Fitur      : {X.shape[1]} fitur GLCM\n")

    # SVM dengan kernel RBF — lebih baik untuk fitur tekstur yang tidak linear
    print("[*] Melatih SVM (kernel=rbf)...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*45}")
    print(f"  Akurasi pada data uji  : {acc*100:.2f}%")
    print(f"{'='*45}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cv_scores = cross_val_score(svm, X_scaled, y_enc, cv=5)
    print(f"Cross-validation (5-fold): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    os.makedirs('model', exist_ok=True)
    joblib.dump(svm,         'model/batik_svm.pkl')
    joblib.dump(le,          'model/label_encoder.pkl')
    joblib.dump(scaler,      'model/scaler.pkl')
    joblib.dump(class_names, 'model/class_names.pkl')

    print(f"\n[✓] Model SVM disimpan ke model/batik_svm.pkl")
    print("[✓] Sekarang jalankan: python app.py\n")


if __name__ == '__main__':
    main()