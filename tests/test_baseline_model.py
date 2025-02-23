import unittest
import numpy as np
import os
import sys

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_baseline import NeuralNetwork  # Adjust the import according to your project structure

class TestNeuralNetwork(unittest.TestCase):
    
    def setUp(self):
        # Initialize a small NN for testing purposes.
        self.input_size = 4
        self.hidden_size = 5
        self.output_size = 3
        self.nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        
        # Create synthetic input data: 10 samples, 4 features each.
        self.X = np.random.randn(10, self.input_size)
        
        # Create one-hot encoded labels for 10 samples (with labels in 0, 1, or 2).
        labels = np.random.randint(0, self.output_size, size=10)
        self.y = np.zeros((10, self.output_size))
        self.y[np.arange(10), labels] = 1

    def test_relu(self):
        # Verify that relu returns 0 for negatives and identity for positives.
        x = np.array([[-1, 0, 1]])
        expected = np.array([[0, 0, 1]])
        np.testing.assert_array_equal(self.nn.relu(x), expected)

    def test_softmax(self):
        # Check that softmax outputs sum to 1 along axis 1.
        x = np.array([[1, 2, 3]])
        sm = self.nn.softmax(x)
        row_sum = np.sum(sm, axis=1)
        np.testing.assert_allclose(row_sum, np.ones_like(row_sum), rtol=1e-5)
        # Optionally, check a known softmax value.
        expected = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        np.testing.assert_allclose(sm, expected, rtol=1e-5)

    def test_cross_entropy_loss(self):
        # For a single sample with known softmax output.
        y_hat = np.array([[0.1, 0.7, 0.2]])
        y_true = np.array([[0, 1, 0]])
        loss = self.nn.cross_entropy_loss(y_true, y_hat)
        expected_loss = -np.log(0.7)  # since loss = -log(probability of the true class)
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_forward_propagation_output_shape(self):
        # Check that forward propagation produces output with shape (num_samples, output_size).
        output = self.nn.forward_propagation(self.X)
        self.assertEqual(output.shape, (self.X.shape[0], self.output_size))

    def test_compute_gradients_shapes(self):
        # Run forward propagation to set intermediate values.
        self.nn.forward_propagation(self.X)
        grads = self.nn.compute_gradients(self.X, self.y)
        # There should be 6 gradients: [dW1, db1, dW2, db2, dW3, db3].
        self.assertEqual(len(grads), 6)
        self.assertEqual(grads[0].shape, (self.input_size, self.hidden_size))
        self.assertEqual(grads[1].shape, (1, self.hidden_size))
        self.assertEqual(grads[2].shape, (self.hidden_size, self.hidden_size))
        self.assertEqual(grads[3].shape, (1, self.hidden_size))
        self.assertEqual(grads[4].shape, (self.hidden_size, self.output_size))
        self.assertEqual(grads[5].shape, (1, self.output_size))

    def test_predict_output(self):
        # Ensure that predict returns a 1-D array of labels.
        predictions = self.nn.predict(self.X)
        self.assertEqual(predictions.shape, (self.X.shape[0],))
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions < self.output_size))

    def test_evaluate(self):
        # Train the network briefly and then evaluate on the training set.
        self.nn.train(self.X, self.y, learning_rate=0.01, epochs=5)
        accuracy = self.nn.evaluate(self.X, self.y)
        # Accuracy should be a percentage between 0 and 100.
        self.assertTrue(0 <= accuracy <= 100)

    def test_train_updates_parameters(self):
        # Check that training updates the network's parameters.
        params_before = [self.nn.W1.copy(), self.nn.b1.copy(), 
                         self.nn.W2.copy(), self.nn.b2.copy(), 
                         self.nn.W3.copy(), self.nn.b3.copy()]
        
        self.nn.train(self.X, self.y, learning_rate=0.01, epochs=1)
        params_after = [self.nn.W1, self.nn.b1, self.nn.W2, self.nn.b2, self.nn.W3, self.nn.b3]
        
        for before, after in zip(params_before, params_after):
            # The parameters should have changed after one epoch.
            self.assertFalse(np.allclose(before, after), "Parameters should be updated after training")

    def test_save_and_load_model(self):
        # Train the network briefly, then save the model.
        self.nn.train(self.X, self.y, learning_rate=0.01, epochs=1)
        temp_file = "temp_model.npy"
        self.nn.save_model(temp_file)
        
        # Modify one parameter so we can detect the load.
        self.nn.W1 += 1.0
        modified_param = self.nn.W1.copy()
        
        # Load the model from file.
        self.nn.load_model(temp_file)
        
        # Clean up the temporary file.
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # After loading, the parameter should not match the modified version.
        self.assertFalse(np.allclose(self.nn.W1, modified_param), "Model load should restore original parameters")
        # Check that the shape remains correct.
        self.assertEqual(self.nn.W1.shape, (self.input_size, self.hidden_size))

if __name__ == '__main__':
    unittest.main()
