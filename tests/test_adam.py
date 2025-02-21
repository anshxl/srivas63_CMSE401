import unittest
import numpy as np
import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adam import AdamOptimizer

class TestAdamOptimizer(unittest.TestCase):

    def test_initialization(self):
        # Create dummy parameters
        param1 = np.array([1.0, 2.0])
        param2 = np.array([[1.0, 2.0], [3.0, 4.0]])
        parameters = [param1, param2]
        optimizer = AdamOptimizer(parameters)
        
        # Check that moment arrays (m and v) are correctly initialized as zeros
        self.assertTrue(np.array_equal(optimizer.m[0], np.zeros_like(param1)))
        self.assertTrue(np.array_equal(optimizer.m[1], np.zeros_like(param2)))
        self.assertTrue(np.array_equal(optimizer.v[0], np.zeros_like(param1)))
        self.assertTrue(np.array_equal(optimizer.v[1], np.zeros_like(param2)))
        
        # Check that time step starts at 0.
        self.assertEqual(optimizer.t, 0)

    def test_single_update(self):
        # Single parameter test: parameter = [1.0], gradient = [0.1]
        # Using learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8.
        param = np.array([1.0])
        grad = np.array([0.1])
        optimizer = AdamOptimizer([param], learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        
        # Perform one update.
        updated_params = optimizer.update([grad])
        
        # Manual calculation for the first update:
        # m = 0.9*0 + 0.1*0.1 = 0.01, then m_hat = 0.01/(1 - 0.9) = 0.1
        # v = 0.999*0 + 0.001*(0.1**2) = 0.00001, then v_hat = 0.00001/(1 - 0.999) = 0.01
        # Update: new_param = 1.0 - 0.001 * (0.1 / (sqrt(0.01) + 1e-8))
        #         sqrt(0.01) = 0.1, so new_param = 1.0 - 0.001 * (0.1/0.1) = 1.0 - 0.001 = 0.999
        expected_param = np.array([0.999])
        np.testing.assert_allclose(updated_params[0], expected_param, rtol=1e-5)

    def test_zero_gradient_update(self):
        # When the gradient is zero, the parameter should remain unchanged.
        param = np.array([5.0, -3.0])
        grad = np.array([0.0, 0.0])
        optimizer = AdamOptimizer([param], learning_rate=0.001)
        updated_params = optimizer.update([grad])
        np.testing.assert_array_equal(updated_params[0], param)

    def test_vector_update(self):
        # Test the update for vector parameters.
        param = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.2, 0.3])
        optimizer = AdamOptimizer([param], learning_rate=0.001)
        updated_params = optimizer.update([grad])
        
        # Ensure the shape remains the same and that an update has occurred.
        self.assertEqual(updated_params[0].shape, param.shape)
        self.assertFalse(np.allclose(updated_params[0], param))
    
    def test_multiple_update_steps(self):
        # Test that multiple updates accumulate correctly.
        param = np.array([1.0])
        grad1 = np.array([0.1])
        grad2 = np.array([0.2])
        optimizer = AdamOptimizer([param], learning_rate=0.001, beta1=0.9, beta2=0.999)
        
        # First update
        updated_params1 = optimizer.update([grad1])
        param_after_first = updated_params1[0].copy()
        
        # Second update with a different gradient
        updated_params2 = optimizer.update([grad2])
        param_after_second = updated_params2[0].copy()
        
        # Check that the timestep is now 2 and parameters have changed
        self.assertEqual(optimizer.t, 2)
        self.assertNotEqual(param_after_first, param_after_second)

if __name__ == '__main__':
    unittest.main()
