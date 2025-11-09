#!/usr/bin/env python3
"""
Training script.js for Handwritten Digit Classifier
"""

import yaml
import os
import traceback
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.submission import SubmissionGenerator


def main():
    try:
        # Load configuration
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        print("=== Handwritten Digit Classifier Training ===")

        # 1. Load data
        print("Step 1: Loading data...")
        loader = DataLoader(config)
        X, y = loader.load_training_data()

        # Check if test data exists
        test_data = loader.load_test_data()
        if test_data is not None:
            print(f"Test data available: {test_data.shape[0]} samples")

        # 2. Preprocess data
        print("Step 2: Preprocessing data...")
        preprocessor = Preprocessor(config)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

        # 3. Train Random Forest
        print("Step 3: Training Random Forest model...")
        trainer = ModelTrainer(config)
        rf_model = trainer.train_random_forest(X_train, y_train)

        # 4. Train Neural Network
        print("Step 4: Training Neural Network model...")
        nn_model = trainer.train_neural_network(X_train, y_train)

        # 5. Evaluate models
        print("Step 5: Evaluating models...")
        evaluator = ModelEvaluator(config)

        print("\n--- Random Forest Evaluation ---")
        rf_accuracy = evaluator.evaluate_model(rf_model, X_test, y_test, 'rf')

        print("\n--- Neural Network Evaluation ---")
        nn_accuracy = evaluator.evaluate_model(nn_model, X_test, y_test, 'nn')

        # 6. Compare models
        print("\n--- Model Comparison ---")
        evaluator.compare_models()

        # 7. Generate submissions if test data exists
        if test_data is not None:
            print("\nStep 6: Generating submission files...")
            submission_gen = SubmissionGenerator(config)

            # Generate individual model submissions
            rf_submission = submission_gen.generate_submission('rf', 'rf_submission.csv')
            nn_submission = submission_gen.generate_submission('nn', 'nn_submission.csv')

            # Generate ensemble submission
            ensemble_submission = submission_gen.generate_ensemble_submission('ensemble_submission.csv')

            print("\n=== Submission Files Generated ===")
            print("1. rf_submission.csv - Random Forest predictions")
            print("2. nn_submission.csv - Neural Network predictions")
            print("3. ensemble_submission.csv - Ensemble predictions")

        print("\n🎉 Training Completed Successfully! 🎉")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()