#!/usr/bin/env python3
"""
Main script for Handwritten Digit Classifier
"""

import argparse
import yaml
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.predictor import Predictor


def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(description='Handwritten Digit Classifier')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate'],
                        required=True, help='Operation mode')
    parser.add_argument('--model', choices=['rf', 'nn'], default='rf',
                        help='Model type: rf (Random Forest) or nn (Neural Network)')
    parser.add_argument('--image_path', help='Path to image for prediction')

    args = parser.parse_args()
    config = load_config()

    if args.mode == 'train':
        train_model(config, args.model)
    elif args.mode == 'predict':
        predict_digit(config, args.model, args.image_path)
    elif args.mode == 'evaluate':
        evaluate_model(config, args.model)


def train_model(config, model_type):
    """Train the selected model"""
    print(f"Training {model_type} model...")

    # Load data
    loader = DataLoader(config)
    X, y = loader.load_data()

    # Preprocess data
    preprocessor = Preprocessor(config)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

    # Train model
    trainer = ModelTrainer(config)
    if model_type == 'rf':
        model = trainer.train_random_forest(X_train, y_train)
    else:
        model = trainer.train_neural_network(X_train, y_train)

    # Evaluate model
    evaluator = ModelEvaluator(config)
    evaluator.evaluate_model(model, X_test, y_test, model_type)

    print("Training completed!")


def predict_digit(config, model_type, image_path):
    """Predict digit from custom image"""
    print(f"Predicting digit using {model_type} model...")

    predictor = Predictor(config)
    prediction, confidence = predictor.predict_custom_image(model_type, image_path)

    print(f"Predicted digit: {prediction} (Confidence: {confidence:.2f})")


def evaluate_model(config, model_type):
    """Evaluate existing model"""
    print(f"Evaluating {model_type} model...")

    evaluator = ModelEvaluator(config)
    evaluator.load_and_evaluate(model_type)


if __name__ == "__main__":
    main()