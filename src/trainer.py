from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
import json
import numpy as np


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model_path = config['model']['save_path']
        os.makedirs(self.model_path, exist_ok=True)

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        print("Training Random Forest...")

        rf_config = self.config['model']['random_forest']
        model = RandomForestClassifier(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            random_state=rf_config['random_state'],
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Calculate training accuracy
        train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"Random Forest Training Accuracy: {train_accuracy:.4f}")

        # Save model
        model_path = os.path.join(self.model_path, 'random_forest_model.pkl')
        joblib.dump(model, model_path)
        print(f"Random Forest model saved to {model_path}")

        return model

    def train_neural_network(self, X_train, y_train):
        """Train Neural Network classifier"""
        print("Training Neural Network...")

        # Build model
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(784,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Neural Network Architecture:")
        model.summary()

        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=self.config['model']['neural_network']['epochs'],
            batch_size=self.config['model']['neural_network']['batch_size'],
            validation_split=self.config['training']['validation_split'],
            verbose=1
        )

        # Calculate final training accuracy
        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        print(f"Neural Network Training Accuracy: {train_accuracy:.4f}")

        # Save model
        model_path = os.path.join(self.model_path, 'neural_network_model.h5')
        model.save(model_path)
        print(f"Neural Network model saved to {model_path}")

        # Save training history
        history_path = os.path.join(self.model_path, 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_dict = {key: [float(x) for x in values] for key, values in history.history.items()}
            json.dump(history_dict, f)

        return model

    def load_model(self, model_type):
        """Load a trained model"""
        if model_type == 'rf':
            model_path = os.path.join(self.model_path, 'random_forest_model.pkl')
            return joblib.load(model_path)
        elif model_type == 'nn':
            model_path = os.path.join(self.model_path, 'neural_network_model.h5')
            return keras.models.load_model(model_path)
        else:
            raise ValueError("Model type must be 'rf' or 'nn'")