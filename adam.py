# Implement ADAM optimizer

import numpy as np

class AdamOptimizer:
    """
    Adam optimizer for updating parameters using the Adam optimization algorithm.

    Args:
        parameters (list): List of parameters to be optimized.
        learning_rate (float, optional): The learning rate. Default is 0.001.
        beta1 (float, optional): Exponential decay rate for the first moment estimates. Default is 0.9.
        beta2 (float, optional): Exponential decay rate for the second raw moment estimates. Default is 0.999.
        epsilon (float, optional): Small value added to the denominator for numerical stability. Default is 1e-8.
    """

    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # timestep
        # Initialize moment estimates for each parameter
        self.m = [np.zeros_like(param) for param in parameters]
        self.v = [np.zeros_like(param) for param in parameters]
        self.parameters = parameters

    def update(self, grads):
        """
        Update the parameters using the Adam optimization algorithm.

        Args:
            grads (list): List of gradients for each parameter.

        Returns:
            list: Updated parameters.
        """
        self.t += 1
        updated_params = []
        for i, (param, grad) in enumerate(zip(self.parameters, grads)):
            # Update biased first moment estimate.
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate.
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            # Compute bias-corrected first moment.
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second moment.
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # Update parameter.
            param_updated = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_params.append(param_updated)
        # Update the parameters stored internally.
        self.parameters = updated_params
        return self.parameters