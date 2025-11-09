import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os


class SubmissionGenerator:
    def __init__(self, config):
        self.config = config
        self.model_path = config['model']['save_path']

    def generate_submission(self, model_type='rf', submission_file='submission.csv'):
        """Generate submission file for test dataset"""
        from src.data_loader import DataLoader
        from src.predictor import Predictor

        # Load test data
        loader = DataLoader(self.config)
        test_data = loader.load_test_data()

        if test_data is None:
            print("No test data found. Cannot generate submission.")
            return

        # Load sample submission for format
        sample_submission = loader.load_sample_submission()

        # Load predictor
        predictor = Predictor(self.config)

        print(f"Generating predictions for {len(test_data)} test samples...")

        predictions = []

        # Convert test data to numpy array and normalize
        test_data_normalized = test_data / 255.0

        # Load model based on type
        if model_type == 'rf':
            model_path = os.path.join(self.model_path, 'random_forest_model.pkl')
            model = joblib.load(model_path)
            predictions = model.predict(test_data_normalized)
        else:
            model_path = os.path.join(self.model_path, 'neural_network_model.h5')
            model = tf.keras.models.load_model(model_path)
            prob_predictions = model.predict(test_data_normalized, verbose=1)
            predictions = np.argmax(prob_predictions, axis=1)

        # Create submission dataframe
        if sample_submission is not None:
            submission = sample_submission.copy()
            submission['Label'] = predictions
        else:
            # Create submission from scratch
            submission = pd.DataFrame({
                'ImageId': range(1, len(predictions) + 1),
                'Label': predictions
            })

        # Save submission file
        submission_path = os.path.join(self.config['data']['raw_path'], submission_file)
        submission.to_csv(submission_path, index=False)

        print(f"Submission file saved to: {submission_path}")
        print(f"First 10 predictions: {predictions[:10]}")

        return submission

    def generate_ensemble_submission(self, submission_file='ensemble_submission.csv'):
        """Generate ensemble prediction using both models"""
        from src.data_loader import DataLoader

        # Load test data
        loader = DataLoader(self.config)
        test_data = loader.load_test_data()

        if test_data is None:
            print("No test data found. Cannot generate submission.")
            return

        test_data_normalized = test_data / 255.0

        # Load both models
        rf_model_path = os.path.join(self.model_path, 'random_forest_model.pkl')
        nn_model_path = os.path.join(self.model_path, 'neural_network_model.h5')

        rf_model = joblib.load(rf_model_path)
        nn_model = tf.keras.models.load_model(nn_model_path)

        # Get predictions from both models
        rf_predictions = rf_model.predict(test_data_normalized)
        nn_prob_predictions = nn_model.predict(test_data_normalized, verbose=1)
        nn_predictions = np.argmax(nn_prob_predictions, axis=1)

        # Ensemble: use NN predictions, fall back to RF when confidence is low
        ensemble_predictions = []
        nn_confidence = np.max(nn_prob_predictions, axis=1)

        for i in range(len(test_data)):
            if nn_confidence[i] > 0.8:  # High confidence threshold
                ensemble_predictions.append(nn_predictions[i])
            else:
                ensemble_predictions.append(rf_predictions[i])

        # Create submission
        submission = pd.DataFrame({
            'ImageId': range(1, len(ensemble_predictions) + 1),
            'Label': ensemble_predictions
        })

        # Save submission file
        submission_path = os.path.join(self.config['data']['raw_path'], submission_file)
        submission.to_csv(submission_path, index=False)

        print(f"Ensemble submission saved to: {submission_path}")
        print(f"Used NN predictions: {(nn_confidence > 0.8).sum()}/{len(nn_confidence)}")

        return submission