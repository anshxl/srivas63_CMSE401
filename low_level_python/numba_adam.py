import math
from numba.typed import List # type: ignore
from LinearAlgebra import scalar_multiply, subtract_matrices, add_matrices

class AdamOptimizer:
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        la: An instance of your LinearAlgebra class, for helper methods like
            scalar_multiply, subtract_matrices, etc.
        parameters: A list of model parameters [W1, b1, W2, b2, W3, b3], each of which
                    is a typed list (or a typed list of typed lists).
        learning_rate, beta1, beta2, epsilon: Adam hyperparameters.
        """
        self.parameters = parameters  # [W1, b1, W2, b2, W3, b3], etc.
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Timestep

        # Initialize first and second moment vectors for each parameter as typed lists.
        self.m = List()
        self.v = List()
        for p in self.parameters:
            self.m.append(self.zeros_like(p))
            self.v.append(self.zeros_like(p))

    def zeros_like(self, x):
        """Recursively create a zero structure matching the input using typed lists."""
        if isinstance(x, list):
            new_list = List()
            for elem in x:
                new_list.append(self.zeros_like(elem))
            return new_list
        else:
            return 0.0

    def update(self, grads):
        """
        grads: A list of gradients [dW1, db1, dW2, db2, dW3, db3],
               each with the same structure as the corresponding parameter.
        """
        self.t += 1
        updated_params = List()

        for i, (param, grad) in enumerate(zip(self.parameters, grads)):
            # m = beta1 * m + (1 - beta1) * grad
            self.m[i] = add_matrices(
                scalar_multiply(self.beta1, self.m[i]),
                scalar_multiply(1 - self.beta1, grad)
            )

            # v = beta2 * v + (1 - beta2) * (grad^2)
            grad_sq = self.square(grad)
            self.v[i] = add_matrices(
                scalar_multiply(self.beta2, self.v[i]),
                scalar_multiply(1 - self.beta2, grad_sq)
            )

            # Compute bias-corrected first moment: m_hat = m / (1 - beta1^t)
            m_hat = scalar_multiply(
                1.0 / (1.0 - self.beta1 ** self.t), self.m[i]
            )
            # Compute bias-corrected second moment: v_hat = v / (1 - beta2^t)
            v_hat = scalar_multiply(
                1.0 / (1.0 - self.beta2 ** self.t), self.v[i]
            )

            # update = lr * m_hat / (sqrt(v_hat) + epsilon)
            sqrt_v_hat = self.sqrt_list(v_hat)
            denom = self.add_epsilon(sqrt_v_hat, self.epsilon)
            ratio = self.elementwise_divide(m_hat, denom)
            update_val = scalar_multiply(self.lr, ratio)

            # param = param - update_val
            param_updated = subtract_matrices(param, update_val)
            updated_params.append(param_updated)

        # Update the internal parameters using slice assignment on the typed list.
        self.parameters[:] = updated_params
        return self.parameters

    # --------------------------
    # Helper Methods for Adam
    # --------------------------

    def square(self, x):
        """Recursively square elements of x using typed lists."""
        if isinstance(x, list):
            new_list = List()
            for elem in x:
                new_list.append(self.square(elem))
            return new_list
        else:
            return x * x

    def sqrt_list(self, x):
        """Recursively apply sqrt to elements of x using typed lists."""
        if isinstance(x, list):
            new_list = List()
            for elem in x:
                new_list.append(self.sqrt_list(elem))
            return new_list
        else:
            return math.sqrt(x)

    def add_epsilon(self, x, eps):
        """Add epsilon to each element of x using typed lists."""
        if isinstance(x, list):
            new_list = List()
            for elem in x:
                new_list.append(self.add_epsilon(elem, eps))
            return new_list
        else:
            return x + eps

    def elementwise_divide(self, a, b):
        """Recursively do elementwise division a / b using typed lists."""
        if isinstance(a, list) and isinstance(b, list):
            new_list = List()
            for x, y in zip(a, b):
                new_list.append(self.elementwise_divide(x, y))
            return new_list
        else:
            return a / b
