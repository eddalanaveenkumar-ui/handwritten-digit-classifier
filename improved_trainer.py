#!/usr/bin/env python3
"""
Improved Model Training with Better Architecture and Data Handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV


class ImprovedDigitTrainer:
    def __init__(self):
        self.config = {
            'data_path': 'data/raw/train.csv',
            'model_save_path': 'models/improved_models/',
            'test_size': 0.2,
            'random_state': 42
        }
        os.makedirs(self.config['model_save_path'], exist_ok=True)

    def load_and_analyze_data(self):
        """Load data and perform thorough analysis"""
        print("📊 Loading and analyzing dataset...")

        # Load data
        data = pd.read_csv(self.config['data_path'])

        # Check basic info
        print(f"Dataset shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")

        # Separate features and labels
        y = data['label']
        X = data.drop('label', axis=1)

        # Data quality checks
        print(f"\n🔍 Data Quality Analysis:")
        print(f"Number of samples: {len(X):,}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Label distribution:\n{y.value_counts().sort_index()}")
        print(f"Missing values: {X.isnull().sum().sum()}")
        print(f"Pixel value range: {X.values.min()} to {X.values.max()}")

        # Check if data needs normalization
        if X.values.max() > 1:
            print("✅ Data needs normalization (scaling to 0-1)")
            X = X / 255.0

        return X, y

    def enhanced_preprocessing(self, X, y):
        """Enhanced preprocessing with data augmentation"""
        from sklearn.model_selection import train_test_split

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )

        print(f"\n📈 Data Split:")
        print(f"Training set: {X_train.shape}")
        print(f"Testing set: {X_test.shape}")

        # Simple data augmentation - add slight variations
        X_train_augmented, y_train_augmented = self.simple_augmentation(X_train, y_train)

        return X_train_augmented, X_test, y_train_augmented, y_test

    def simple_augmentation(self, X, y):
        """Add simple data augmentation"""
        print("🔄 Applying data augmentation...")

        # Original data
        X_augmented = [X]
        y_augmented = [y]

        # Add slight shifts (1 pixel in each direction)
        for shift_x, shift_y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            X_shifted = self.shift_images(X, shift_x, shift_y)
            X_augmented.append(X_shifted)
            y_augmented.append(y)

        # Combine all
        X_combined = np.vstack(X_augmented)
        y_combined = np.hstack(y_augmented)

        print(f"Augmented dataset: {X_combined.shape}")
        return X_combined, y_combined

    def shift_images(self, X, shift_x, shift_y):
        """Shift images by given pixels"""
        X_shifted = np.zeros_like(X)

        for i in range(len(X)):
            img = X[i].reshape(28, 28)
            img_shifted = np.roll(img, shift_x, axis=1)
            img_shifted = np.roll(img_shifted, shift_y, axis=0)
            X_shifted[i] = img_shifted.flatten()

        return X_shifted

    def train_improved_random_forest(self, X_train, y_train):
        """Train an improved Random Forest with better parameters"""
        print("\n🌲 Training Improved Random Forest...")

        # Use better parameters
        model = RandomForestClassifier(
            n_estimators=200,  # More trees
            max_depth=30,  # Deeper trees
            min_samples_split=5,  # Prevent overfitting
            min_samples_leaf=2,  # Better generalization
            max_features='sqrt',  # Better feature selection
            bootstrap=True,
            random_state=self.config['random_state'],
            n_jobs=-1,  # Use all cores
            verbose=1
        )

        print("Fitting model... (this may take a few minutes)")
        model.fit(X_train, y_train)

        # Save model
        model_path = os.path.join(self.config['model_save_path'], 'improved_rf_model.pkl')
        joblib.dump(model, model_path)
        print(f"✅ Improved Random Forest saved to: {model_path}")

        return model

    def train_neural_network(self, X_train, y_train):
        """Train a better neural network"""
        print("\n🧠 Training Improved Neural Network...")

        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),  # Deeper architecture
            activation='relu',
            solver='adam',
            alpha=0.001,  # Regularization
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=100,  # More iterations
            random_state=self.config['random_state'],
            verbose=True,
            early_stopping=True,
            validation_fraction=0.1
        )

        print("Fitting neural network...")
        model.fit(X_train, y_train)

        # Save model
        model_path = os.path.join(self.config['model_save_path'], 'improved_nn_model.pkl')
        joblib.dump(model, model_path)
        print(f"✅ Improved Neural Network saved to: {model_path}")

        return model

    def evaluate_models(self, models, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n📊 Model Evaluation Results:")
        print("=" * 50)

        results = {}

        for name, model in models.items():
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            print(f"\n{name.upper()} Results:")
            print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

            # Per-class accuracy
            class_accuracy = cm.diagonal() / cm.sum(axis=1)
            print("Per-class accuracy:")
            for digit, acc in enumerate(class_accuracy):
                print(f"  Digit {digit}: {acc:.3f}")

            # Save results
            results[name] = {
                'accuracy': accuracy,
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            # Plot confusion matrix
            self.plot_confusion_matrix(cm, name)

        return results

    def plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(10), yticklabels=range(10))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Save plot
        plot_path = os.path.join(self.config['model_save_path'], f'confusion_matrix_{model_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved: {plot_path}")

    def run_complete_training(self):
        """Run complete training pipeline"""
        print("🚀 Starting Improved Model Training Pipeline")
        print("=" * 60)

        # 1. Load and analyze data
        X, y = self.load_and_analyze_data()

        # 2. Preprocessing
        X_train, X_test, y_train, y_test = self.enhanced_preprocessing(X, y)

        # 3. Train models
        rf_model = self.train_improved_random_forest(X_train, y_train)
        nn_model = self.train_neural_network(X_train, y_train)

        # 4. Evaluate
        models = {
            'improved_random_forest': rf_model,
            'improved_neural_network': nn_model
        }

        results = self.evaluate_models(models, X_test, y_test)

        # 5. Save performance metrics
        self.save_performance_metrics(results)

        print("\n🎉 Training Completed Successfully!")
        print("Improved models saved in: models/improved_models/")

        return results

    def save_performance_metrics(self, results):
        """Save performance metrics"""
        metrics = {}
        for name, result in results.items():
            metrics[name] = {
                'accuracy': float(result['accuracy']),
                'model_type': name
            }

        metrics_path = os.path.join(self.config['model_save_path'], 'performance_metrics.json')
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"📈 Performance metrics saved: {metrics_path}")


if __name__ == "__main__":
    trainer = ImprovedDigitTrainer()
    trainer.run_complete_training()