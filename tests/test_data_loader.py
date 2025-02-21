import unittest
import numpy as np
import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import load_data

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Load MNIST data
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = load_data()

    def test_loading(self):
        # Check that the data has been loaded
        self.assertIsNotNone(self.train_images)
        self.assertIsNotNone(self.train_labels)
        self.assertEqual(len(self.train_images), len(self.train_labels))
        self.assertEqual(self.train_images.ndim, 2)  # Expecting (num_samples, height x width)

    def test_preprocessing(self):
        # Check that the data has been preprocessed
        self.assertTrue(np.all(self.train_images >= 0))
        self.assertTrue(np.all(self.train_images <= 1))

    def test_one_hot_encoding(self):
        # Check that the labels have been one-hot encoded
        self.assertEqual(self.train_labels.shape[1], 10)
        self.assertTrue(np.all(self.train_labels.sum(axis=1) == 1))

    def test_transform_data(self):
        # Check that the data has been transformed
        self.assertEqual(self.train_images.shape[1], 784)
        self.assertEqual(self.test_images.shape[1], 784)
    
if __name__ == '__main__':
    unittest.main()