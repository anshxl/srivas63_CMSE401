# GPU Optimizer
# Adam optimizer using CuPy arrays
import cupy as cp #type: ignore[import]

class GPUAdamOptimizer:
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # timestep
        # Initialize moment estimates for each parameter with CuPy
        self.m = [cp.zeros_like(param) for param in parameters]
        self.v = [cp.zeros_like(param) for param in parameters]
        self.parameters = parameters

    def update(self, grads):
        self.t += 1
        updated_params = []
        for i, (param, grad) in enumerate(zip(self.parameters, grads)):
            # Update biased first moment estimate.
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate.
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            # Compute bias-corrected estimates.
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # Parameter update.
            param_updated = param - self.learning_rate * m_hat / (cp.sqrt(v_hat) + self.epsilon)
            updated_params.append(param_updated)
        # Update stored parameters.
        self.parameters = updated_params
        return self.parameters
