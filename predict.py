#!/usr/bin/env python3
"""
Prediction script for Handwritten Digit Classifier
"""

import argparse
import yaml
from src.predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description='Predict handwritten digit')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--model', choices=['rf', 'nn'], default='rf',
                        help='Model to use for prediction')

    args = parser.parse_args()

    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Predict digit
    predictor = Predictor(config)
    prediction, confidence = predictor.predict_custom_image(args.model, args.image_path)

    print(f"\n🎯 Prediction Result:")
    print(f"📊 Model: {args.model.upper()}")
    print(f"🔢 Predicted Digit: {prediction}")
    print(f"💯 Confidence: {confidence:.2%}")
    print(f"🖼️  Image: {args.image_path}")


if __name__ == "__main__":
    main()