import numpy as np

class Sgd:

    def __init__(self, learning_rate):
        if not isinstance(learning_rate, float):
            learning_rate = float(learning_rate)
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum:

    def __init__(self, learning_rate, momentum_rate):
        self.v_k = 0
        self.momentum_rate = momentum_rate
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v_k = self.momentum_rate * self.v_k - (self.learning_rate * gradient_tensor)
        return weight_tensor + self.v_k


class Adam:

    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.g_k = 0    # k refers to the kth step/iteration of the gradient update
        self.v_k = 0
        self.r_k = 0
        self.counter_k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.counter_k += 1
        self.g_k = gradient_tensor
        self.v_k = self.mu * self.v_k + (1 - self.mu) * self.g_k
        self.r_k = self.rho * self.r_k + np.multiply((1 - self.rho) * self.g_k, self.g_k)

        # bias correction
        v_hat = self.v_k / (1 - self.mu**self.counter_k)
        r_hat = self.r_k / (1 - self.rho**self.counter_k)

        return weight_tensor - self.learning_rate * (v_hat / (np.sqrt(r_hat) + np.finfo(np.float64).eps))
    
