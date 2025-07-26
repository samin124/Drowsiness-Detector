import os
from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense
from tensorflow.keras import layers
from PIL import Image
import base64
from io import BytesIO
import requests

# Constants
MODEL_URL = 'https://drive.google.com/uc?id=1i6XhpxCL47hvDWBD35r4sJispzVJogqR'
MODEL_PATH = 'MobileNet3_UP.h5'
IMG_SIZE = 224
CLASS_LABELS = {0: 'Closed Eyes', 1: 'Open Eyes', 2: 'No Yawn', 3: 'Yawn'}

app = Flask(__name__)

# ✅ Correct SE block (must match training-time definition)
def se_block(input_tensor, reduction=8):
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // reduction, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = layers.Multiply()([input_tensor, se])
    return x

# Download model from Google Drive (render-friendly version)
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        gdown_url = MODEL_URL
        response = requests.get(gdown_url, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model downloaded.")
    else:
        print("Model already exists.")

# Call download and load model
download_model()
model = load_model(MODEL_PATH, custom_objects={'se_block': se_block})

# Preprocess incoming image
def preprocess_image(image_data):
    img = Image.open(BytesIO(image_data)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction interpreters
def interpret_eye_prediction(pred):
    eye_probs = pred[0][:2]
    return 'Closed Eyes' if np.argmax(eye_probs) == 0 else 'Open Eyes'

def interpret_mouth_prediction(pred):
    mouth_probs = pred[0][2:]
    return 'No Yawn' if np.argmax(mouth_probs) == 0 else 'Yawn'

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract and decode base64 images
        left_eye_b64 = data['left_eye'].split(',')[1]
        right_eye_b64 = data['right_eye'].split(',')[1]
        mouth_b64 = data['mouth'].split(',')[1]

        left_eye_img = base64.b64decode(left_eye_b64)
        right_eye_img = base64.b64decode(right_eye_b64)
        mouth_img = base64.b64decode(mouth_b64)

        # Preprocess
        left_eye_input = preprocess_image(left_eye_img)
        right_eye_input = preprocess_image(right_eye_img)
        mouth_input = preprocess_image(mouth_img)

        # Predict
        left_pred = model.predict(left_eye_input)
        right_pred = model.predict(right_eye_input)
        mouth_pred = model.predict(mouth_input)

        # Interpret predictions
        left_eye_status = interpret_eye_prediction(left_pred)
        right_eye_status = interpret_eye_prediction(right_pred)
        mouth_status = interpret_mouth_prediction(mouth_pred)

        # Drowsiness logic
        if (left_eye_status == 'Closed Eyes' and right_eye_status == 'Open Eyes' and mouth_status == 'No Yawn') or \
           (left_eye_status == 'Open Eyes' and right_eye_status == 'Closed Eyes' and mouth_status == 'No Yawn') or \
           (left_eye_status == 'Open Eyes' and right_eye_status == 'Open Eyes' and mouth_status == 'No Yawn'):
            status = "Alert"
        else:
            status = "Drowsy"

        return jsonify({
            'status': status,
            'left_eye': left_eye_status,
            'right_eye': right_eye_status,
            'mouth': mouth_status
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Start Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
