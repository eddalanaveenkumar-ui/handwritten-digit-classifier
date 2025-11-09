#!/usr/bin/env python3
"""
Flask Web Application for Handwritten Digit Classification
"""

from flask import Flask, render_template, request, jsonify
import base64
import io
import os
import json
from PIL import Image
import numpy as np
import joblib
import tensorflow as tf
from src.predictor import Predictor
import yaml

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


class WebPredictor:
    def __init__(self):
        self.config = self.load_config()
        self.predictor = Predictor(self.config)
        self.model_stats = self.load_model_stats()

    def load_config(self):
        """Load configuration from YAML file"""
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)

    def load_model_stats(self):
        """Load model performance statistics"""
        stats_file = os.path.join(self.config['model']['save_path'], 'model_performance.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                return json.load(f)
        return {}

    def preprocess_web_image(self, image_data):
        """Preprocess image from web canvas"""
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')

            # Ensure 28x28 size
            image = image.resize((28, 28))

            # Convert to numpy array and normalize
            img_array = np.array(image) / 255.0

            # Invert colors if background is white
            if np.mean(img_array) > 0.5:
                img_array = 1 - img_array

            return img_array.flatten()

        except Exception as e:
            print(f"Image preprocessing error: {e}")
            raise

    def predict_digit_web(self, image_data, model_type='ensemble'):
        """Predict digit from web image"""
        try:
            # Preprocess image
            processed_image = self.preprocess_web_image(image_data)

            # Make prediction
            if model_type == 'ensemble':
                # Get predictions from both models
                rf_pred, rf_conf = self.predict_with_model('rf', processed_image)
                nn_pred, nn_conf = self.predict_with_model('nn', processed_image)

                # Use the prediction with higher confidence
                if nn_conf > rf_conf:
                    prediction, confidence = nn_pred, nn_conf
                else:
                    prediction, confidence = rf_pred, rf_conf

                # Get all predictions for confidence bars
                all_predictions = self.get_all_predictions(processed_image)

            else:
                prediction, confidence = self.predict_with_model(model_type, processed_image)
                all_predictions = self.get_all_predictions(processed_image, model_type)

            return {
                'success': True,
                'prediction': int(prediction),
                'confidence': float(confidence),
                'all_predictions': all_predictions,
                'model_used': model_type
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def predict_with_model(self, model_type, image_array):
        """Predict using specific model"""
        if model_type == 'rf':
            model_path = os.path.join(self.config['model']['save_path'], 'random_forest_model.pkl')
            model = joblib.load(model_path)
            prediction = model.predict([image_array])[0]
            confidence = np.max(model.predict_proba([image_array]))
        else:  # nn
            model_path = os.path.join(self.config['model']['save_path'], 'neural_network_model.h5')
            model = tf.keras.models.load_model(model_path)
            probabilities = model.predict(np.array([image_array]), verbose=0)
            prediction = np.argmax(probabilities)
            confidence = np.max(probabilities)

        return prediction, confidence

    def get_all_predictions(self, image_array, model_type='nn'):
        """Get confidence scores for all digits 0-9"""
        try:
            if model_type == 'rf':
                model_path = os.path.join(self.config['model']['save_path'], 'random_forest_model.pkl')
                model = joblib.load(model_path)
                probabilities = model.predict_proba([image_array])[0]
            else:  # nn
                model_path = os.path.join(self.config['model']['save_path'], 'neural_network_model.h5')
                model = tf.keras.models.load_model(model_path)
                probabilities = model.predict(np.array([image_array]), verbose=0)[0]

            # Convert to list and ensure 10 classes
            all_probs = probabilities.tolist()
            if len(all_probs) < 10:
                all_probs.extend([0.0] * (10 - len(all_probs)))
            elif len(all_probs) > 10:
                all_probs = all_probs[:10]

            return all_probs

        except Exception as e:
            print(f"Error getting all predictions: {e}")
            return [0.0] * 10


# Initialize predictor
web_predictor = WebPredictor()


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle digit prediction requests"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})

        image_file = request.files['image']
        model_type = request.form.get('model_type', 'ensemble')

        # Read image data
        image_data = image_file.read()

        # Convert to base64 for processing
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        image_data_url = f"data:image/png;base64,{image_b64}"

        # Make prediction
        result = web_predictor.predict_digit_web(image_data_url, model_type)

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/model_stats')
def model_stats():
    """Get model performance statistics"""
    stats = {}

    if 'rf' in web_predictor.model_stats:
        stats['rf_accuracy'] = web_predictor.model_stats['rf']['accuracy']

    if 'nn' in web_predictor.model_stats:
        stats['nn_accuracy'] = web_predictor.model_stats['nn']['accuracy']

    return jsonify(stats)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'models_loaded': True})


if __name__ == '__main__':
    print("🚀 Starting Handwritten Digit Classifier Web App...")
    print("📧 Access the application at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")

    # Create necessary directories
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

    app.run(debug=True, host='0.0.0.0', port=5000)