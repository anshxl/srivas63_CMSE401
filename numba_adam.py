import math
from LinearAlgebra import LinAlg as la

la = la()

class AdamOptimizer:
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        la: An instance of your LinearAlgebra class, for helper methods like
            scalar_multiply, subtract_matrices, etc.
        parameters: A list of model parameters [W1, b1, W2, b2, W3, b3], each of which
                    is a Python list (or list of lists).
        learning_rate, beta1, beta2, epsilon: Adam hyperparameters.
        """
        self.parameters = parameters  # [W1, b1, W2, b2, W3, b3], etc.
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Timestep

        # Initialize first and second moment vectors for each parameter
        self.m = [self.zeros_like(p) for p in self.parameters]
        self.v = [self.zeros_like(p) for p in self.parameters]

    def zeros_like(self, x):
        """Recursively create a zero structure matching the input."""
        if isinstance(x, list):
            return [self.zeros_like(elem) for elem in x]
        else:
            return 0.0

    def update(self, grads):
        """
        grads: A list of gradients [dW1, db1, dW2, db2, dW3, db3],
               each with the same structure as the corresponding parameter.
        """
        self.t += 1
        updated_params = []

        for i, (param, grad) in enumerate(zip(self.parameters, grads)):
            # m = beta1 * m + (1 - beta1) * grad
            self.m[i] = la.add_matrices(
                la.scalar_multiply(self.beta1, self.m[i]),
                la.scalar_multiply(1 - self.beta1, grad)
            )

            # v = beta2 * v + (1 - beta2) * (grad^2)
            # We'll define a helper to square 'grad' recursively.
            grad_sq = self.square(grad)
            self.v[i] = la.add_matrices(
                la.scalar_multiply(self.beta2, self.v[i]),
                la.scalar_multiply(1 - self.beta2, grad_sq)
            )

            # Compute bias-corrected first moment: m_hat = m / (1 - beta1^t)
            m_hat = la.scalar_multiply(
                1.0 / (1.0 - self.beta1 ** self.t), self.m[i]
            )
            # Compute bias-corrected second moment: v_hat = v / (1 - beta2^t)
            v_hat = la.scalar_multiply(
                1.0 / (1.0 - self.beta2 ** self.t), self.v[i]
            )

            # update = lr * m_hat / (sqrt(v_hat) + epsilon)
            # We'll define a helper to take elementwise sqrt, then add epsilon
            # and do elementwise division.
            sqrt_v_hat = self.sqrt_list(v_hat)
            denom = self.add_epsilon(sqrt_v_hat, self.epsilon)
            ratio = self.elementwise_divide(m_hat, denom)
            update_val = la.scalar_multiply(self.lr, ratio)

            # param = param - update_val
            param_updated = la.subtract_matrices(param, update_val)
            updated_params.append(param_updated)

        # Update the internal parameters
        self.parameters[:] = updated_params
        return self.parameters

    # --------------------------
    # Helper Methods for Adam
    # --------------------------

    def square(self, x):
        """Recursively square elements of x."""
        if isinstance(x, list):
            return [self.square(elem) for elem in x]
        else:
            return x * x

    def sqrt_list(self, x):
        """Recursively apply sqrt to elements of x."""
        if isinstance(x, list):
            return [self.sqrt_list(elem) for elem in x]
        else:
            return math.sqrt(x)

    def add_epsilon(self, x, eps):
        """Add epsilon to each element of x."""
        if isinstance(x, list):
            return [self.add_epsilon(elem, eps) for elem in x]
        else:
            return x + eps

    def elementwise_divide(self, a, b):
        """Recursively do elementwise division a / b."""
        if isinstance(a, list) and isinstance(b, list):
            return [self.elementwise_divide(x, y) for x, y in zip(a, b)]
        else:
            return a / b
