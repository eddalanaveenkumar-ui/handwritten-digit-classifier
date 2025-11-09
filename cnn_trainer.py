#!/usr/bin/env python3
"""
FIXED CNN Model for Handwritten Digit Recognition (Proper Architecture for 28x28)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class FixedCNNTrainer:
    def __init__(self):
        self.config = {
            'data_path': 'data/raw/train.csv',
            'model_save_path': 'models/cnn_model/',
            'img_size': 28,
            'batch_size': 128,  # Increased for better performance
            'epochs': 50,  # More epochs for better convergence
            'random_state': 42
        }
        os.makedirs(self.config['model_save_path'], exist_ok=True)

        # Set memory growth to avoid GPU issues
        self.setup_gpu()

    def setup_gpu(self):
        """Configure GPU settings"""
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print("✅ GPU configured with memory growth")
            except:
                print("⚠️ Could not configure GPU, using CPU")
        else:
            print("🔶 Using CPU for training")

    def load_and_prepare_data(self):
        """Load and prepare data for CNN"""
        print("📊 Loading and preparing data for CNN...")

        # Load data
        data = pd.read_csv(self.config['data_path'])
        y = data['label'].values
        X = data.drop('label', axis=1).values

        print(f"Original data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")

        # Normalize and reshape for CNN
        X = X.astype('float32') / 255.0
        X = X.reshape(-1, 28, 28, 1)  # Reshape for CNN (samples, height, width, channels)

        print(f"Reshaped data: {X.shape}")

        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("Class distribution:")
        for digit, count in zip(unique, counts):
            print(f"  Digit {digit}: {count} samples ({count / len(y) * 100:.1f}%)")

        return X, y

    def create_proper_cnn_model(self):
        """Create a PROPER CNN model architecture for 28x28 images"""
        model = keras.Sequential([
            # First Conv Block - Input: 28x28x1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),  # Output: 14x14x32
            layers.Dropout(0.25),

            # Second Conv Block - Input: 14x14x32
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),  # Output: 7x7x64
            layers.Dropout(0.25),

            # Third Conv Block - Input: 7x7x64
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), padding='same'),  # Output: 4x4x128 (with padding)
            layers.Dropout(0.25),

            # Classifier
            layers.Flatten(),  # Output: 4*4*128 = 2048
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])

        # Compile model with better optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def create_simple_cnn_model(self):
        """Create a simpler but effective CNN model"""
        model = keras.Sequential([
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Classifier
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def create_data_augmentation(self):
        """Create data augmentation pipeline"""
        return keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,  # Random rotation between -10 and +10 degrees
            width_shift_range=0.1,  # Random horizontal shift
            height_shift_range=0.1,  # Random vertical shift
            zoom_range=0.1,  # Random zoom
            shear_range=0.1,  # Random shear
            fill_mode='nearest'  # Fill points outside boundaries
        )

    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the CNN model with data augmentation"""
        # Create model - using simple model for stability
        model = self.create_simple_cnn_model()

        print("🧠 CNN Model Architecture:")
        model.summary()

        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,  # More patience
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,  # More patience for LR reduction
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.config['model_save_path'], 'best_cnn_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                os.path.join(self.config['model_save_path'], 'training_log.csv')
            )
        ]

        # Data augmentation
        datagen = self.create_data_augmentation()

        print("🚀 Starting CNN Training with Data Augmentation...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")

        # Calculate steps per epoch
        steps_per_epoch = len(X_train) // self.config['batch_size']

        # Train model
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=self.config['batch_size']),
            steps_per_epoch=steps_per_epoch,
            epochs=self.config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        return model, history

    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n📊 CNN Model Evaluation:")

        # Predictions
        y_pred_proba = model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate accuracy
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)

        print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # Detailed classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)

        # Per-class accuracy
        cm = confusion_matrix(y_test, y_pred)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        print("\n🎯 Per-class Accuracy:")
        for digit, acc in enumerate(class_accuracy):
            print(f"  Digit {digit}: {acc:.3f}")

        return accuracy, y_pred

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(10), yticklabels=range(10),
                    cbar_kws={'shrink': 0.8})
        plt.title('CNN Model - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        # Save plot
        plot_path = os.path.join(self.config['model_save_path'], 'confusion_matrix_cnn.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Confusion matrix saved: {plot_path}")

    def plot_training_history(self, history):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot learning rate if available
        if 'lr' in history.history:
            ax3.plot(history.history['lr'], label='Learning Rate', linewidth=2, color='red')
            ax3.set_title('Learning Rate', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.axis('off')

        # Training summary
        ax4.axis('off')
        best_val_acc = max(history.history['val_accuracy'])
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]

        summary_text = f"""Training Summary:
Final Training Accuracy: {final_train_acc:.3f}
Final Validation Accuracy: {final_val_acc:.3f}
Best Validation Accuracy: {best_val_acc:.3f}
Epochs Trained: {len(history.history['accuracy'])}
Expected Test Accuracy: {best_val_acc * 100:.1f}%"""

        ax4.text(0.1, 0.5, summary_text, fontsize=12, va='center', ha='left',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['model_save_path'], 'training_history.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print("📈 Training history plots saved")

    def save_model_info(self, accuracy, history):
        """Save model information and performance"""
        model_info = {
            'accuracy': float(accuracy),
            'training_accuracy': float(history.history['accuracy'][-1]),
            'validation_accuracy': float(history.history['val_accuracy'][-1]),
            'best_validation_accuracy': float(max(history.history['val_accuracy'])),
            'epochs_trained': len(history.history['accuracy']),
            'model_architecture': 'Simple CNN (Fixed)',
            'parameters': {
                'batch_size': self.config['batch_size'],
                'epochs': self.config['epochs'],
                'img_size': self.config['img_size']
            },
            'expected_accuracy_range': '98.5% - 99.5%'
        }

        info_path = os.path.join(self.config['model_save_path'], 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        print(f"📄 Model info saved: {info_path}")
        return model_info

    def run_training(self):
        """Run complete CNN training pipeline"""
        print("🚀 Starting FIXED CNN Training Pipeline")
        print("=" * 60)

        try:
            # 1. Load data
            X, y = self.load_and_prepare_data()

            # 2. Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.15,  # 15% for final test
                random_state=self.config['random_state'],
                stratify=y
            )

            # Further split for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=0.15,  # 15% of training for validation
                random_state=self.config['random_state'],
                stratify=y_train
            )

            print(f"\n📊 Data Split Summary:")
            print(f"Training set: {X_train.shape}")
            print(f"Validation set: {X_val.shape}")
            print(f"Test set: {X_test.shape}")

            # 3. Train model
            model, history = self.train_model(X_train, y_train, X_val, y_val)

            # 4. Evaluate on test set
            accuracy, y_pred = self.evaluate_model(model, X_test, y_test)

            # 5. Plot training history
            self.plot_training_history(history)

            # 6. Save final model
            model_path = os.path.join(self.config['model_save_path'], 'final_cnn_model.h5')
            model.save(model_path)
            print(f"✅ Final CNN model saved: {model_path}")

            # 7. Save model info
            model_info = self.save_model_info(accuracy, history)

            print(f"\n🎉 CNN Training Completed Successfully!")
            print(f"🏆 Final Test Accuracy: {accuracy * 100:.2f}%")
            print(f"📈 Best Validation Accuracy: {model_info['best_validation_accuracy'] * 100:.2f}%")
            print(f"🎯 Expected Real-world Accuracy: {model_info['expected_accuracy_range']}")

            return model, accuracy, model_info

        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise


if __name__ == "__main__":
    trainer = FixedCNNTrainer()
    model, accuracy, info = trainer.run_training()