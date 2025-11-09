import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append('../')

from src.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.config = {'data': {'raw_path': '../data/raw/'}}
        self.loader = DataLoader(self.config)

        # Create a sample CSV file for testing
        os.makedirs('../data/raw/', exist_ok=True)
        sample_data = pd.DataFrame({
            'label': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'pixel_0': [0] * 10,
            'pixel_1': [255] * 10,
            # ... more pixels
        })
        # For simplicity, just create a small test file
        sample_data.to_csv('../data/raw/test_sample.csv', index=False)

    def test_data_loader_initialization(self):
        self.assertIsNotNone(self.loader)
        self.assertEqual(self.loader.raw_path, '../data/raw/')

    def test_load_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.loader.load_data()

    def tearDown(self):
        # Clean up test file
        if os.path.exists('../data/raw/test_sample.csv'):
            os.remove('../data/raw/test_sample.csv')


if __name__ == '__main__':
    unittest.main()