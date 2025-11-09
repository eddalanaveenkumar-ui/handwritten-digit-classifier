import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import tensorflow as tf
import os
import json


class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.model_path = config['model']['save_path']

    def evaluate_model(self, model, X_test, y_test, model_type):
        """Evaluate model performance"""
        print(f"Evaluating {model_type} model...")

        if model_type == 'rf':
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        else:  # neural network
            y_proba = model.predict(X_test)
            y_pred = np.argmax(y_proba, axis=1)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_pred, model_type)

        # Save performance metrics
        self._save_performance_metrics(accuracy, model_type)

        return accuracy

    def _plot_confusion_matrix(self, y_true, y_pred, model_type):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(10), yticklabels=range(10))
        plt.title(f'Confusion Matrix - {model_type.upper()}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Save plot
        plot_path = os.path.join(self.model_path, f'confusion_matrix_{model_type}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to {plot_path}")

    def _save_performance_metrics(self, accuracy, model_type):
        """Save performance metrics to JSON file"""
        metrics_file = os.path.join(self.model_path, 'model_performance.json')

        # Initialize empty metrics dictionary
        metrics = {}

        # Check if file exists and has content
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    file_content = f.read().strip()
                    if file_content:  # Only load if file is not empty
                        metrics = json.loads(file_content)
                    else:
                        print("Metrics file is empty, creating new metrics dictionary.")
                        metrics = {}
            except (json.JSONDecodeError, Exception) as e:
                print(f"Warning: Could not load existing metrics file: {e}")
                print("Creating new metrics dictionary.")
                metrics = {}

        # Update metrics with current model
        metrics[model_type] = {
            'accuracy': float(accuracy),
            'model_type': model_type
        }

        # Save updated metrics
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Performance metrics saved to {metrics_file}")
        except Exception as e:
            print(f"Error saving metrics: {e}")

    def load_and_evaluate(self, model_type):
        """Load and evaluate saved model"""
        # Load test data
        X_test = np.load(os.path.join(self.config['data']['processed_path'], 'X_test.npy'))
        y_test = np.load(os.path.join(self.config['data']['processed_path'], 'y_test.npy'))

        # Load model
        if model_type == 'rf':
            model_path = os.path.join(self.model_path, 'random_forest_model.pkl')
            model = joblib.load(model_path)
        else:
            model_path = os.path.join(self.model_path, 'neural_network_model.h5')
            model = tf.keras.models.load_model(model_path)

        print(f"Loaded {model_type} model from {model_path}")
        return self.evaluate_model(model, X_test, y_test, model_type)

    def compare_models(self):
        """Compare performance of all trained models"""
        metrics_file = os.path.join(self.model_path, 'model_performance.json')

        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    file_content = f.read().strip()
                    if file_content:
                        metrics = json.loads(file_content)
                    else:
                        print("Metrics file is empty.")
                        return None
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error loading metrics file: {e}")
                return None

            print("\n=== Model Comparison ===")
            for model_type, model_metrics in metrics.items():
                print(f"{model_type.upper()}: Accuracy = {model_metrics['accuracy']:.4f}")

            return metrics
        else:
            print("No performance metrics found. Please train models first.")
            return None