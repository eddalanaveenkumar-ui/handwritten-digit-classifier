#!/usr/bin/env python3
"""
FIXED Web App with JSON Serializable CNN Support
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

app = Flask(__name__)

# Import the predictors
from fine_tune_model import fine_tuner
from cnn_predictor import cnn_predictor


class JSONSerializablePredictor:
    def __init__(self):
        self.config = self.load_config()
        self.models = self.load_models()
        self.model_stats = self.load_model_stats()

        # Load CNN predictor
        self.cnn_predictor = cnn_predictor

        print("🤖 Available Models:")
        for model_name in self.models.keys():
            print(f"  ✅ {model_name}")
        print(f"  ✅ CNN: {self.cnn_predictor.is_available()}")
        print(f"  ✅ Fine-tuned: {fine_tuner.fine_tuned}")

    def load_config(self):
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)

    def load_models(self):
        """Load all available models"""
        models = {}

        # Try to load improved Random Forest
        rf_path = 'models/improved_models/improved_rf_model.pkl'
        if os.path.exists(rf_path):
            models['rf'] = joblib.load(rf_path)
            print("✅ Loaded Improved Random Forest")
        else:
            # Try original RF
            rf_original = 'models/random_forest_model.pkl'
            if os.path.exists(rf_original):
                models['rf'] = joblib.load(rf_original)
                print("✅ Loaded Original Random Forest")

        # Try to load improved Neural Network
        nn_path = 'models/improved_models/improved_nn_model.pkl'
        if os.path.exists(nn_path):
            models['nn'] = joblib.load(nn_path)
            print("✅ Loaded Improved Neural Network")

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

        # CNN model stats
        cnn_stats = 'models/cnn_model/model_info.json'
        if os.path.exists(cnn_stats):
            with open(cnn_stats, 'r') as f:
                cnn_info = json.load(f)
                stats['cnn'] = {
                    'accuracy': float(cnn_info['accuracy']),  # Ensure float
                    'model_type': 'cnn'
                }

        return stats

    def preprocess_image(self, image_data):
        """Improved preprocessing for all models"""
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

            return img_array

        except Exception as e:
            print(f"Preprocessing error: {e}")
            raise

    def predict_with_cnn(self, image_array):
        """Predict using CNN model"""
        if not self.cnn_predictor.is_available():
            return None

        result = self.cnn_predictor.predict(image_array)
        return result

    def predict_with_traditional(self, image_array, model_type):
        """Predict using traditional models with JSON serializable output"""
        if model_type not in self.models:
            return None

        try:
            # Flatten image for traditional models
            flattened = image_array.astype('float32').flatten() / 255.0

            model = self.models[model_type]
            prediction = model.predict([flattened])[0]
            confidence = np.max(model.predict_proba([flattened]))
            all_predictions = model.predict_proba([flattened])[0].tolist()

            # Ensure 10 classes and convert to Python native types
            if len(all_predictions) < 10:
                all_predictions.extend([0.0] * (10 - len(all_predictions)))

            # Convert to JSON serializable types
            all_predictions = [float(prob) for prob in all_predictions]

            return {
                'prediction': int(prediction),  # Convert to Python int
                'confidence': float(confidence),  # Convert to Python float
                'all_predictions': all_predictions,  # Already converted
                'model_used': model_type
            }

        except Exception as e:
            print(f"Traditional model prediction error: {e}")
            return None

    def predict(self, image_data, model_type='ensemble'):
        """Enhanced prediction with multiple model support and JSON serialization"""
        try:
            image_array = self.preprocess_image(image_data)

            # Strategy 1: Try fine-tuned model first (if available and confident)
            fine_tuned_result = fine_tuner.predict_with_fine_tuning(image_data)
            if fine_tuned_result and fine_tuned_result['confidence'] > 0.7:
                result = fine_tuned_result
                result['model_used'] = 'fine_tuned'
                result['success'] = True

                # Ensure JSON serializable
                result['prediction'] = int(result['prediction'])
                result['confidence'] = float(result['confidence'])
                if 'all_predictions' in result:
                    result['all_predictions'] = [float(p) for p in result['all_predictions']]

                print("🎯 Using fine-tuned model (high confidence)")
                return result

            # Strategy 2: Use specified model or ensemble
            if model_type == 'cnn':
                result = self.predict_with_cnn(image_array)
            elif model_type == 'ensemble':
                # Try CNN first, fall back to best traditional model
                result = self.predict_with_cnn(image_array)
                if not result or result['confidence'] < 0.8:
                    rf_result = self.predict_with_traditional(image_array, 'rf')
                    if rf_result and (not result or rf_result['confidence'] > result['confidence']):
                        result = rf_result
            else:
                result = self.predict_with_traditional(image_array, model_type)

            if result:
                result['success'] = True
                return result
            else:
                return {'success': False, 'error': 'No model available for prediction'}

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {'success': False, 'error': str(e)}


# Initialize predictor
predictor = JSONSerializablePredictor()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle digit prediction with enhanced model selection"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})

        image_file = request.files['image']
        model_type = request.form.get('model_type', 'ensemble')

        image_data = image_file.read()
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        image_data_url = f"data:image/png;base64,{image_b64}"

        # Make prediction
        result = predictor.predict(image_data_url, model_type)

        # Final JSON serialization check
        if result['success']:
            # Ensure all numeric values are JSON serializable
            result['prediction'] = int(result['prediction'])
            result['confidence'] = float(result['confidence'])
            if 'all_predictions' in result:
                result['all_predictions'] = [float(p) for p in result['all_predictions']]

        return jsonify(result)

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Prediction endpoint error: {error_msg}")
        return jsonify({'success': False, 'error': error_msg})


@app.route('/correct_prediction', methods=['POST'])
def correct_prediction():
    """Handle correction feedback from user"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        correct_label = data.get('correct_label')

        if not image_data or correct_label is None:
            return jsonify({'success': False, 'error': 'Missing image or label'})

        print(f"🔄 Processing correction for digit {correct_label}...")

        # Add to fine-tuning dataset
        success = fine_tuner.collect_training_sample(image_data, correct_label)

        # Auto fine-tune after collecting enough samples
        if len(fine_tuner.training_data) >= 5:
            print("🔄 Auto fine-tuning with user samples...")
            fine_tuner.fine_tune_model()

        return jsonify({
            'success': bool(success),  # Ensure boolean
            'samples_count': int(len(fine_tuner.training_data)),  # Ensure int
            'fine_tuned': bool(fine_tuner.fine_tuned)  # Ensure boolean
        })

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Correction endpoint error: {error_msg}")
        return jsonify({'success': False, 'error': error_msg})


@app.route('/model_stats')
def model_stats():
    stats = {}

    for model_name, model_info in predictor.model_stats.items():
        if 'accuracy' in model_info:
            # Ensure all values are JSON serializable
            stats[f"{model_name}_accuracy"] = float(model_info['accuracy'])

    # Add fine-tuning info with JSON serializable types
    stats['fine_tuned_samples'] = int(len(fine_tuner.training_data))
    stats['fine_tuned'] = bool(fine_tuner.fine_tuned)
    stats['cnn_available'] = bool(predictor.cnn_predictor.is_available())

    return jsonify(stats)


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': bool(len(predictor.models) > 0),
        'cnn_available': bool(predictor.cnn_predictor.is_available()),
        'fine_tuner_ready': True,
        'fine_tuned_samples': int(len(fine_tuner.training_data))
    })


def ensure_json_serializable(obj):
    """Recursively ensure all values in object are JSON serializable"""
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray)):
        return obj.tolist()
    else:
        return obj


if __name__ == '__main__':
    print("🚀 Starting JSON-SERIALIZABLE Handwritten Digit Classifier Web App...")
    print("📧 Access at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop")
    print("🎯 Features: CNN + JSON Fix + Multiple Models")

    # Create necessary directories
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    app.run(debug=True, host='0.0.0.0', port=5000)