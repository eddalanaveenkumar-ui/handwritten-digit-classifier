import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor


def plot_sample_images(X, y, num_samples=16, title="Sample Images"):
    """Plot sample images from the dataset"""
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    indices = np.random.choice(len(X), num_samples, replace=False)

    for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
        if hasattr(X, 'iloc'):
            image = X.iloc[idx].values.reshape(28, 28)
            label = y.iloc[idx]
        else:
            image = X[idx].reshape(28, 28)
            label = y[idx]

        ax.imshow(image, cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_training_history():
    """Plot training history for neural network"""
    import json
    import os

    history_path = os.path.join('models', 'training_history.json')

    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Plot loss
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
    else:
        print("Training history not found. Please train the neural network first.")


def explore_dataset_statistics():
    """Explore and display dataset statistics"""
    config = {'data': {'raw_path': 'data/raw/'}}
    loader = DataLoader(config)
    X, y = loader.load_data()

    print("=== Dataset Statistics ===")
    print(f"Total samples: {len(X):,}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Image dimensions: 28x28 = 784 pixels")
    print(f"Number of classes: {len(y.unique())}")

    # Class distribution
    class_dist = y.value_counts().sort_index()
    print("\nClass Distribution:")
    for digit, count in class_dist.items():
        print(f"  Digit {digit}: {count:>5} samples ({count / len(y) * 100:.1f}%)")

    # Pixel value statistics
    print(f"\nPixel Value Statistics:")
    print(f"  Min value: {X.values.min()}")
    print(f"  Max value: {X.values.max()}")
    print(f"  Mean value: {X.values.mean():.2f}")
    print(f"  Std value: {X.values.std():.2f}")

    # Plot class distribution
    plt.figure(figsize=(10, 6))
    class_dist.plot(kind='bar')
    plt.title('Distribution of Handwritten Digits')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return X, y


def create_sample_image(digit=None):
    """Create a sample digit image for testing"""
    if digit is None:
        digit = np.random.randint(0, 10)

    # Create a simple representation of the digit
    img = np.zeros((28, 28))

    # Simple patterns for each digit
    patterns = {
        0: [(10, 10), (10, 17), (17, 10), (17, 17)],
        1: [(13, 10), (13, 17)],
        2: [(10, 10), (17, 10), (13, 13), (10, 17), (17, 17)],
        # Add more patterns as needed
    }

    if digit in patterns:
        for point in patterns[digit]:
            img[point] = 255

    return img, digit