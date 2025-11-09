import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt


class Predictor:
    def __init__(self, config):
        self.config = config
        self.model_path = config['model']['save_path']

    def predict_custom_image(self, model_type, image_path):
        """Predict digit from custom image"""
        # Load and preprocess image
        image = self._preprocess_image(image_path)

        # Load model
        if model_type == 'rf':
            model = joblib.load(os.path.join(self.model_path, 'random_forest_model.pkl'))
            prediction = model.predict([image])[0]
            confidence = np.max(model.predict_proba([image]))
        else:
            model = tf.keras.models.load_model(os.path.join(self.model_path, 'neural_network_model.h5'))
            probabilities = model.predict(np.array([image]), verbose=0)
            prediction = np.argmax(probabilities)
            confidence = np.max(probabilities)

        # Display the image with prediction
        self._display_prediction(image_path, prediction, confidence, model_type)

        return prediction, confidence

    def _preprocess_image(self, image_path):
        """Preprocess custom image for prediction"""
        # Load image
        img = Image.open(image_path).convert('L')  # Convert to grayscale

        # Resize to 28x28
        img = img.resize((28, 28))

        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0

        # Invert colors if background is dark (MNIST has white digits on black background)
        if np.mean(img_array) > 0.5:  # If background is light
            img_array = 1 - img_array

        # Flatten to 1D array (784 pixels)
        img_flat = img_array.flatten()

        return img_flat

    def _display_prediction(self, image_path, prediction, confidence, model_type):
        """Display the image with prediction results"""
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))

        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        plt.title(f'Prediction: {prediction}\nConfidence: {confidence:.2%}\nModel: {model_type.upper()}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def predict_batch(self, model_type, images_folder):
        """Predict digits for multiple images"""
        predictions = []

        for filename in os.listdir(images_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(images_folder, filename)
                prediction, confidence = self.predict_custom_image(model_type, image_path)

                predictions.append({
                    'filename': filename,
                    'prediction': int(prediction),
                    'confidence': float(confidence)
                })

        return predictions

    def predict_digit_from_array(self, model_type, image_array):
        """Predict digit from numpy array"""
        # Preprocess the array
        if image_array.shape != (784,):
            # Reshape if necessary
            if image_array.shape == (28, 28):
                image_flat = image_array.flatten()
            else:
                raise ValueError("Image array must be shape (784,) or (28, 28)")
        else:
            image_flat = image_array

        # Normalize
        image_flat = image_flat / 255.0

        # Load model and predict
        if model_type == 'rf':
            model = joblib.load(os.path.join(self.model_path, 'random_forest_model.pkl'))
            prediction = model.predict([image_flat])[0]
            confidence = np.max(model.predict_proba([image_flat]))
        else:
            model = tf.keras.models.load_model(os.path.join(self.model_path, 'neural_network_model.h5'))
            probabilities = model.predict(np.array([image_flat]), verbose=0)
            prediction = np.argmax(probabilities)
            confidence = np.max(probabilities)

        return prediction, confidence