import os
from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import base64
from io import BytesIO


# load model from drive

import os
import requests

MODEL_URL = 'https://drive.google.com/file/d/1i6XhpxCL47hvDWBD35r4sJispzVJogqR/view?usp=sharing'
MODEL_PATH = 'MobileNet3_UP.h5'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model downloaded.")
    else:
        print("Model already exists.")

# Download model before loading it
download_model()

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
# other imports...

# Now load your model
model = load_model(MODEL_PATH, custom_objects={'se_block': se_block})


# rest of your flask app code...

app = Flask(__name__)

# Load model - update path if needed
MODEL_PATH = 'MobileNet3_UP.h5'

def se_block(input_tensor, reduction=8):
    # Dummy function for loading model with custom objects
    from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Multiply
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // reduction, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = Multiply()([input_tensor, se])
    return x

model = load_model(MODEL_PATH, custom_objects={'se_block': se_block})

# Class labels
CLASS_LABELS = {0: 'Closed Eyes', 1: 'Open Eyes', 2: 'No Yawn', 3: 'Yawn'}

IMG_SIZE = 224

def preprocess_image(image_data):
    img = Image.open(BytesIO(image_data)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def interpret_eye_prediction(pred):
    # Use only first two classes for eyes
    eye_probs = pred[0][:2]
    eye_index = np.argmax(eye_probs)
    return 'Closed Eyes' if eye_index == 0 else 'Open Eyes'

def interpret_mouth_prediction(pred):
    # Use last two classes for mouth
    mouth_probs = pred[0][2:]
    mouth_index = np.argmax(mouth_probs)
    return 'No Yawn' if mouth_index == 0 else 'Yawn'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract base64 encoded images from JSON
        left_eye_b64 = data['left_eye'].split(',')[1]
        right_eye_b64 = data['right_eye'].split(',')[1]
        mouth_b64 = data['mouth'].split(',')[1]

        # Decode images
        left_eye_img = base64.b64decode(left_eye_b64)
        right_eye_img = base64.b64decode(right_eye_b64)
        mouth_img = base64.b64decode(mouth_b64)

        # Preprocess for model input
        left_eye_input = preprocess_image(left_eye_img)
        right_eye_input = preprocess_image(right_eye_img)
        mouth_input = preprocess_image(mouth_img)

        # Predict
        left_pred = model.predict(left_eye_input)
        right_pred = model.predict(right_eye_input)
        mouth_pred = model.predict(mouth_input)

        # Interpret
        left_eye_status = interpret_eye_prediction(left_pred)
        right_eye_status = interpret_eye_prediction(right_pred)
        mouth_status = interpret_mouth_prediction(mouth_pred)

        # Logic for drowsiness
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

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

