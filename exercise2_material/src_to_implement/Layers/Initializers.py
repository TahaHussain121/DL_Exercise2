import numpy as np


class Constant:

    def __init__(self, value=0.1):
        self.constant = value

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.ones(weights_shape) * self.constant


class UniformRandom:

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(size=weights_shape)


class Xavier:

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_out + fan_in))
        return np.random.normal(scale=sigma, size=weights_shape)


class He:

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(scale=sigma, size=weights_shape)


Constant(0.5)