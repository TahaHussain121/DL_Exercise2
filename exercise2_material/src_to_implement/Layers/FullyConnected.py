#if __name__ == '__main__':
from Layers import Base  # the dot means the current folder / path

import numpy as np


class FullyConnected(Base.BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.rand(input_size+1, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = None
        self.current_input = None
        self.current_error = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt_function):
        self._optimizer = opt_function

    def forward(self, input_tensor):
        input_tensor = np.concatenate((input_tensor.T, np.ones((1, input_tensor.shape[0])))).T
        self.current_input = input_tensor
        output = input_tensor @ self.weights
        return output

    def backward(self, error_tensor):
        #print('shape error_tensor', error_tensor.shape)
        self.current_error = error_tensor
        weights_without_error = np.delete(self.weights, -1, axis=0)
        error_tensor_new = error_tensor @ weights_without_error.T

        #if isinstance(self._optimizer, (Sgd)):
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)

        return error_tensor_new

    @property
    def gradient_weights(self):
        #print('error', self.current_error)
        return (self.current_error.T @ self.current_input).T

    def initialize(self, weights_initializer, bias_initializer):
        # assumes the initializers are the initialize method of the classes in the Initializers.py file
        just_weight = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        just_bias = bias_initializer.initialize((1, self.output_size), self.input_size, self.output_size)
        self.weights = np.concatenate((just_weight, just_bias))



# B = Base.BaseLayer()
# print(B.trainable)
# F = FullyConnected(2,2)
#
# print('F.trainable', F.trainable)
# print('B.trainable', B.trainable)
#print(help(F))

#print(Base.BaseLayer().trainable)