import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Load the model
model_path = os.path.join('model', 'autism_binary_handwritin_modelv.h5')
model = None

try:
    model = load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(img_array):
    if model is None:
        return "Model not loaded"
    prediction = model.predict(img_array)
    return 0 if prediction[0][0] > 0.5 else 1

@app.route('/handWriting', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("No image provided")
        return jsonify({"error": "No image provided"}), 400

    img_file = request.files['image']
    img_path = os.path.join('uploads', img_file.filename)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    img_file.save(img_path)

    try:
        img_array = preprocess_image(img_path)
        prediction = predict_image(img_array)
        print(f"Prediction: {prediction}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=False)