import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.processed_path = config['data']['processed_path']

    def split_data(self, X, y):
        """Split data into training and testing sets"""
        test_size = self.config['training']['test_size']
        random_state = self.config['training']['random_state']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # Normalize pixel values
        if self.config['preprocessing']['normalize']:
            X_train = X_train / 255.0
            X_test = X_test / 255.0

        print(f"Training set: {X_train.shape}")
        print(f"Testing set: {X_test.shape}")

        # Save processed data
        self._save_processed_data(X_train, X_test, y_train, y_test)

        return X_train, X_test, y_train, y_test

    def _save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data to files"""
        os.makedirs(self.processed_path, exist_ok=True)

        np.save(os.path.join(self.processed_path, 'X_train.npy'), X_train)
        np.save(os.path.join(self.processed_path, 'X_test.npy'), X_test)
        np.save(os.path.join(self.processed_path, 'y_train.npy'), y_train)
        np.save(os.path.join(self.processed_path, 'y_test.npy'), y_test)

        print("Processed data saved successfully")

    def load_processed_data(self):
        """Load previously processed data"""
        try:
            X_train = np.load(os.path.join(self.processed_path, 'X_train.npy'))
            X_test = np.load(os.path.join(self.processed_path, 'X_test.npy'))
            y_train = np.load(os.path.join(self.processed_path, 'y_train.npy'))
            y_test = np.load(os.path.join(self.processed_path, 'y_test.npy'))

            print("Processed data loaded successfully")
            return X_train, X_test, y_train, y_test

        except FileNotFoundError:
            print("Processed data not found. Please run preprocessing first.")
            return None, None, None, None