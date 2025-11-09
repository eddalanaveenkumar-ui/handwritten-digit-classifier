import pandas as pd
import numpy as np
import os


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.raw_path = config['data']['raw_path']

    def load_training_data(self):
        """Load training dataset from CSV file"""
        try:
            train_path = os.path.join(self.raw_path, 'train.csv')

            if os.path.exists(train_path):
                print(f"Loading training data from {train_path}")
                data = pd.read_csv(train_path)

                # Separate features and labels
                y = data['label']
                X = data.drop('label', axis=1)

                print(f"Training dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
                return X, y
            else:
                raise FileNotFoundError(f"Training data not found at {train_path}")

        except Exception as e:
            print(f"Error loading training data: {e}")
            raise

    def load_test_data(self):
        """Load test dataset from CSV file"""
        try:
            test_path = os.path.join(self.raw_path, 'test.csv')

            if os.path.exists(test_path):
                print(f"Loading test data from {test_path}")
                test_data = pd.read_csv(test_path)

                print(f"Test dataset loaded: {test_data.shape[0]} samples, {test_data.shape[1]} features")
                return test_data
            else:
                print(f"Test data not found at {test_path}")
                return None

        except Exception as e:
            print(f"Error loading test data: {e}")
            return None

    def load_sample_submission(self):
        """Load sample submission file"""
        try:
            submission_path = os.path.join(self.raw_path, 'sample_submission.csv')

            if os.path.exists(submission_path):
                print(f"Loading sample submission from {submission_path}")
                submission = pd.read_csv(submission_path)
                return submission
            else:
                print("Sample submission file not found")
                return None

        except Exception as e:
            print(f"Error loading sample submission: {e}")
            return None

    def load_data(self):
        """Main method to load data (backward compatibility)"""
        return self.load_training_data()

    def load_custom_image(self, image_path):
        """Load custom image for prediction"""
        from PIL import Image
        import numpy as np

        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img)

        return img_array

    def explore_dataset(self, X, y):
        """Explore and display dataset information"""
        print("\n=== Dataset Exploration ===")
        print(f"Total samples: {len(X)}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Image dimensions: 28x28 pixels")
        print(f"Number of classes: {len(y.unique())}")
        print(f"Class distribution:")
        print(y.value_counts().sort_index())