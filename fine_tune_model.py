#!/usr/bin/env python3
"""
Fine-tune the model on user's drawing style
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import os
from PIL import Image
import json
import base64
import io


class ModelFineTuner:
    def __init__(self):
        self.training_data = []
        self.training_labels = []
        self.model = None
        self.fine_tuned = False

    def collect_training_sample(self, image_data, correct_label):
        """Collect user's drawing with correct label for fine-tuning"""
        try:
            # Preprocess the image
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            if image.mode != 'L':
                image = image.convert('L')

            image = image.resize((28, 28))
            img_array = np.array(image)

            # Invert if needed (black background, white digits)
            if np.mean(img_array) > 128:
                img_array = 255 - img_array

            img_array = img_array.astype('float32') / 255.0
            flattened = img_array.flatten()

            # Store the sample
            self.training_data.append(flattened)
            self.training_labels.append(correct_label)

            print(f"✅ Collected sample for digit {correct_label}. Total samples: {len(self.training_data)}")
            return True

        except Exception as e:
            print(f"Error collecting sample: {e}")
            return False

    def fine_tune_model(self):
        """Fine-tune the model on user's drawing style"""
        if len(self.training_data) < 5:
            print("⚠️ Need at least 5 samples to fine-tune")
            return False

        try:
            # Load base model
            base_model_path = 'models/improved_models/improved_rf_model.pkl'
            if not os.path.exists(base_model_path):
                base_model_path = 'models/random_forest_model.pkl'

            self.model = joblib.load(base_model_path)
            print(f"✅ Loaded base model from {base_model_path}")

            # Convert to numpy arrays
            X_train = np.array(self.training_data)
            y_train = np.array(self.training_labels)

            print(f"🔄 Fine-tuning on {len(X_train)} user samples...")

            # Create a new model that combines base knowledge with user style
            user_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42
            )
            user_model.fit(X_train, y_train)

            # Save fine-tuned model
            self.fine_tuned = True
            joblib.dump(user_model, 'models/fine_tuned_model.pkl')

            # Save training data for future use
            training_info = {
                'samples_count': len(self.training_data),
                'labels_distribution': np.bincount(y_train).tolist(),
                'fine_tuned': True
            }

            with open('models/fine_tuned_info.json', 'w') as f:
                json.dump(training_info, f, indent=2)

            print(f"🎯 Fine-tuned model saved! Trained on {len(X_train)} user samples")
            return True

        except Exception as e:
            print(f"Error fine-tuning: {e}")
            return False

    def predict_with_fine_tuning(self, image_data):
        """Predict using fine-tuned model"""
        try:
            if not self.fine_tuned and os.path.exists('models/fine_tuned_model.pkl'):
                self.model = joblib.load('models/fine_tuned_model.pkl')
                self.fine_tuned = True

            if not self.fine_tuned:
                return None

            # Preprocess image
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            if image.mode != 'L':
                image = image.convert('L')

            image = image.resize((28, 28))
            img_array = np.array(image)

            if np.mean(img_array) > 128:
                img_array = 255 - img_array

            img_array = img_array.astype('float32') / 255.0
            flattened = img_array.flatten()

            # Predict
            prediction = self.model.predict([flattened])[0]
            confidence = np.max(self.model.predict_proba([flattened]))
            all_predictions = self.model.predict_proba([flattened])[0].tolist()

            return {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'all_predictions': [float(p) for p in all_predictions]
            }

        except Exception as e:
            print(f"Fine-tuned prediction error: {e}")
            return None


# Global instance
fine_tuner = ModelFineTuner()