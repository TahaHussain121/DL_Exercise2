import numpy as np
import scipy
from scipy.signal import correlate, correlate2d

from Layers.Base import BaseLayer


class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):

        self.output_shape = None
        self.input_tensor = None
        self.padded_input = None
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.stride_row = self.stride_shape[0]
        self.stride_col = self.stride_shape[1]
        self.convolution_row_shape = convolution_shape[1]
        self.convolution_col_shape = convolution_shape[2]
        self.dim1 = False
        self.trainable = True

        # Initialize weights and biases
        if len(convolution_shape) == 2:  # 1D convolution
            c, m = convolution_shape
            self.weights = np.random.uniform(size=(num_kernels, c, m))
            self.dim1 = True
        elif len(convolution_shape) == 3:  # 2D convolution
            c, m, n = convolution_shape
            self.dim1 = False
            self.stride_col = 1
            self.convolution_col_shape = 1
            self.weights = np.random.uniform(size=(num_kernels, c, m, n))

        self.bias = np.random.uniform(size=(num_kernels,))
        self._optimizer = None  # weight optimizer
        self._bias_optimizer = None
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = None


    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        print(self.input_tensor.shape)
        if self.dim1:
            # Get the batch size and number of channels from the input tensor shape
            batch_size, num_channels, input_length = input_tensor.shape

            # Reshape the input tensor to have shape (batch_size, num_channels * input_length)
            self.input_tensor = np.reshape(input_tensor, (batch_size, num_channels * input_length))


        else:
            batch_size, num_channels, input_height, input_width = input_tensor.shape
            self.input_tensor = np.reshape(input_tensor, (batch_size, num_channels * input_height * input_width))

        # Calculate output shape with same padding
        output_height = (input_tensor.shape[2] + 2 * (self.convolution_shape[0] // 2) - self.convolution_shape[0]) // self.stride_shape[0] + 1
        output_width = (input_tensor.shape[3] + 2 * (self.convolution_shape[1] // 2) - self.convolution_shape[1]) // self.stride_shape[1] + 1

        # Initialize output tensor
        output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, output_height, output_width))

        # Perform correlation operation with same padding
        for batch in range(input_tensor.shape[0]):
            for kernel_idx in range(self.num_kernels):
                for input_channel in range(input_tensor.shape[1]):
                    output_tensor[batch, kernel_idx, :, :] += scipy.signal.correlate2d(
                        input_tensor[batch, input_channel, :, :],
                        self.weights[kernel_idx, input_channel, :, :],
                        mode='same')

                # Add bias term
                output_tensor[batch, kernel_idx, :, :] += self.bias[kernel_idx]

        # Apply stride and reshape
        s_Row = int(np.ceil(output_tensor.shape[2] / self.stride_shape[0]))
        s_Col = int(np.ceil(output_tensor.shape[3] / self.stride_shape[1]))
        output_tensor_stride = np.zeros((input_tensor.shape[0], self.num_kernels, s_Row, s_Col))

        for batch in range(input_tensor.shape[0]):
            for kernel_idx in range(self.num_kernels):
                for j in range(s_Row):
                    for k in range(s_Col):
                        j_in_output_tensor = j * self.stride_shape[0]
                        k_in_output_tensor = k * self.stride_shape[1]
                        output_tensor_stride[batch, kernel_idx, j, k] = output_tensor[
                            batch, kernel_idx, j_in_output_tensor, k_in_output_tensor]

        # Reshape output tensor based on dimensionality
        if self.dim1:
            output_tensor_with_stride = output_tensor_stride.reshape(
                output_tensor_stride.shape[0],
                output_tensor_stride.shape[1],
                output_tensor_stride.shape[2])
        else:
            output_tensor_with_stride = output_tensor_stride.reshape(
                output_tensor_stride.shape[0],
                output_tensor_stride.shape[1],
                output_tensor_stride.shape[2],
                output_tensor_stride.shape[3])

        self.output_shape = output_tensor_with_stride.shape  # store the output shape

        return output_tensor_stride

    def backward(self, error_tensor):
        pass

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.weights.shape[1] * self.weights.shape[2] * self.weights.shape[3]
        fan_out = self.num_kernels * self.weights.shape[2] * self.weights.shape[3]
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.weights.shape[0])
        return self.weights, self.bias

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def gradient_bias(self, value):
        self._gradient_bias = value
    def optimizer(self):
        return self._optimizer

    def optimizer(self, value):
        self._optimizer = value

    def bias_optimizer(self):
        return self._bias_optimizer

    def bias_optimizer(self, value):
        self._bias_optimizer = value

