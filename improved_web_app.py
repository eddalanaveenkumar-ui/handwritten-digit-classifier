#!/usr/bin/env python3
"""
Improved Web App with Better Models
"""

from flask import Flask, render_template, request, jsonify
import base64
import io
import os
import json
import numpy as np
import joblib
from PIL import Image
import yaml
from fine_tune_model import fine_tuner

app = Flask(__name__)


class ImprovedPredictor:
    def __init__(self):
        self.config = self.load_config()
        self.models = self.load_models()
        self.model_stats = self.load_model_stats()

    def load_config(self):
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)

    def load_models(self):
        """Load all improved models"""
        models = {}

        # Try to load improved Random Forest
        rf_path = 'models/improved_models/improved_rf_model.pkl'
        if os.path.exists(rf_path):
            models['rf'] = joblib.load(rf_path)
            print("✅ Loaded Improved Random Forest")
        else:
            print("⚠️ Improved RF not found, using original")
            rf_original = 'models/random_forest_model.pkl'
            if os.path.exists(rf_original):
                models['rf'] = joblib.load(rf_original)

        # Try to load improved Neural Network
        nn_path = 'models/improved_models/improved_nn_model.pkl'
        if os.path.exists(nn_path):
            models['nn'] = joblib.load(nn_path)
            print("✅ Loaded Improved Neural Network")

        # Try to load CNN model
        cnn_path = 'models/cnn_model/final_cnn_model.h5'
        if os.path.exists(cnn_path):
            try:
                import tensorflow as tf
                models['cnn'] = tf.keras.models.load_model(cnn_path)
                print("✅ Loaded CNN Model")
            except:
                print("⚠️ Could not load CNN model")

        if not models:
            raise Exception("❌ No models found! Please train models first.")

        return models

    def load_model_stats(self):
        stats = {}

        # Improved models stats
        improved_stats = 'models/improved_models/performance_metrics.json'
        if os.path.exists(improved_stats):
            with open(improved_stats, 'r') as f:
                stats.update(json.load(f))

        # Original models stats
        original_stats = 'models/model_performance.json'
        if os.path.exists(original_stats):
            with open(original_stats, 'r') as f:
                stats.update(json.load(f))

        return stats

    def preprocess_image(self, image_data):
        """Improved preprocessing"""
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')

            # Resize to 28x28
            image = image.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(image)

            # Invert colors if needed (black background, white digits)
            if np.mean(img_array) > 128:
                img_array = 255 - img_array

            # Normalize
            img_array = img_array.astype('float32') / 255.0

            return img_array.flatten()

        except Exception as e:
            print(f"Preprocessing error: {e}")
            raise

    def predict(self, image_data, model_type='ensemble'):
        """Improved prediction"""
        try:
            processed_image = self.preprocess_image(image_data)

            if model_type == 'ensemble':
                # Use all available models and take best prediction
                predictions = []
                confidences = []

                for name, model in self.models.items():
                    if name == 'cnn':
                        # CNN expects different input shape
                        cnn_input = processed_image.reshape(1, 28, 28, 1)
                        proba = model.predict(cnn_input, verbose=0)[0]
                        pred = np.argmax(proba)
                        conf = np.max(proba)
                    else:
                        # Traditional models
                        proba = model.predict_proba([processed_image])[0]
                        pred = model.predict([processed_image])[0]
                        conf = np.max(proba)

                    predictions.append(pred)
                    confidences.append(conf)

                # Choose prediction with highest confidence
                best_idx = np.argmax(confidences)
                prediction = predictions[best_idx]
                confidence = confidences[best_idx]
                all_predictions = self.models['rf'].predict_proba([processed_image])[0].tolist()

            else:
                # Use specific model
                if model_type not in self.models:
                    return {'success': False, 'error': f'Model {model_type} not available'}

                model = self.models[model_type]

                if model_type == 'cnn':
                    cnn_input = processed_image.reshape(1, 28, 28, 1)
                    proba = model.predict(cnn_input, verbose=0)[0]
                    prediction = np.argmax(proba)
                    confidence = np.max(proba)
                    all_predictions = proba.tolist()
                else:
                    prediction = model.predict([processed_image])[0]
                    confidence = np.max(model.predict_proba([processed_image]))
                    all_predictions = model.predict_proba([processed_image])[0].tolist()

            # Ensure 10 classes
            if len(all_predictions) < 10:
                all_predictions.extend([0.0] * (10 - len(all_predictions)))

            return {
                'success': True,
                'prediction': int(prediction),
                'confidence': float(confidence),
                'all_predictions': all_predictions,
                'model_used': model_type
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}


# Initialize predictor
predictor = ImprovedPredictor()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})

        image_file = request.files['image']
        model_type = request.form.get('model_type', 'ensemble')

        image_data = image_file.read()
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        image_data_url = f"data:image/png;base64,{image_b64}"

        result = predictor.predict(image_data_url, model_type)
        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/model_stats')
def model_stats():
    stats = {}

    for model_name, model_info in predictor.model_stats.items():
        if 'accuracy' in model_info:
            stats[f"{model_name}_accuracy"] = model_info['accuracy']

    return jsonify(stats)


@app.route('/correct_prediction', methods=['POST'])
def correct_prediction():
    """Handle correction feedback from user"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        correct_label = data.get('correct_label')

        if not image_data or correct_label is None:
            return jsonify({'success': False, 'error': 'Missing image or label'})

        # Add to fine-tuning dataset
        success = fine_tuner.collect_training_sample(image_data, correct_label)

        # Auto fine-tune after collecting enough samples
        if len(fine_tuner.training_data) >= 10:
            fine_tuner.fine_tune_model()

        return jsonify({'success': success, 'samples_count': len(fine_tuner.training_data)})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# Modify the predict route to use fine-tuned model
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})

        image_file = request.files['image']
        model_type = request.form.get('model_type', 'ensemble')

        image_data = image_file.read()
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        image_data_url = f"data:image/png;basebase64,{image_b64}"

        # First try fine-tuned model if available
        fine_tuned_result = fine_tuner.predict_with_fine_tuning(image_data_url)

        if fine_tuned_result and fine_tuned_result['confidence'] > 0.6:
            result = fine_tuned_result
            result['model_used'] = 'fine_tuned'
            result['success'] = True
        else:
            # Fall back to regular model
            result = predictor.predict(image_data_url, model_type)

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting IMPROVED Handwritten Digit Classifier Web App...")
    print("📧 Access at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop")

    app.run(debug=True, host='0.0.0.0', port=5000)