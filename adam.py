# Implement ADAM optimizer

import numpy as np

class AdamOptimizer:
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