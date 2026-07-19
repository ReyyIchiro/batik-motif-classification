from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import joblib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

CLASSES = ['Insang', 'Kawung', 'Parang']


def extract_glcm_features(image_path):
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
    display = {}
    for prop in props:
        vals = graycoprops(glcm, prop).flatten()
        features.append(vals.mean())
        features.append(vals.std())
        display[prop.capitalize()] = round(float(vals.mean()), 4)

    return features, display


def load_artifacts():
    model, scaler, le = None, None, None
    # Coba load SVM dulu, fallback ke KNN kalau tidak ada
    for model_path in ['model/batik_svm.pkl', 'model/batik_knn.pkl']:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            break
    if os.path.exists('model/scaler.pkl'):
        scaler = joblib.load('model/scaler.pkl')
    if os.path.exists('model/label_encoder.pkl'):
        le = joblib.load('model/label_encoder.pkl')
    return model, scaler, le

model, scaler, le = load_artifacts()


@app.route('/')
def index():
    return render_template('index.html', classes=CLASSES)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file yang dikirim'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'File kosong'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        features, feature_dict = extract_glcm_features(filepath)

        if model is not None and scaler is not None and le is not None:
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]
            confidence = round(float(max(proba)) * 100, 2)
            label = le.inverse_transform([int(prediction)])[0]
        else:
            label = "Model belum dilatih"
            confidence = 0.0

        return jsonify({
            'success': True,
            'label': label,
            'confidence': confidence,
            'features': feature_dict,
            'demo_mode': model is None
        })

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': 'An internal error occurred during processing.'}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)