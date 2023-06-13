import numpy as np

from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        self.previousShape = None
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.previousShape = input_tensor.shape
        num_horizontal_pools = np.ceil((input_tensor.shape[2] - self.pooling_shape[0] + 1) / self.stride_shape[0])
        num_vertical_pools = np.ceil((input_tensor.shape[3] - self.pooling_shape[1] + 1) / self.stride_shape[1])
        output_tensor = np.zeros((*input_tensor.shape[0:2], int(num_horizontal_pools), int(num_vertical_pools)))
        x_indices = np.zeros((*input_tensor.shape[0:2], int(num_horizontal_pools), int(num_vertical_pools)), dtype=int)
        y_indices = np.zeros((*input_tensor.shape[0:2], int(num_horizontal_pools), int(num_vertical_pools)), dtype=int)

        # Iterate over the horizontal and vertical pooling positions
        for pool_row in range(0, input_tensor.shape[2] - self.pooling_shape[0] + 1, self.stride_shape[0]):
            pool_row_index = (pool_row + self.pooling_shape[0]) // self.stride_shape[0] - 1

            for pool_col in range(0, input_tensor.shape[3] - self.pooling_shape[1] + 1, self.stride_shape[1]):
                pool_col_index = (pool_col + self.pooling_shape[1]) // self.stride_shape[1] - 1

                # Extract the pooling region from the input tensor
                pooling_region = input_tensor[:, :, pool_row:pool_row + self.pooling_shape[0],
                                 pool_col:pool_col + self.pooling_shape[1]].reshape(*input_tensor.shape[0:2], -1)

                # Find the indices of maximum values in the pooling region
                max_values_indices = np.argmax(pooling_region, axis=2)
                x_indices_values = max_values_indices // self.pooling_shape[1]
                y_indices_values = max_values_indices % self.pooling_shape[1]

                # Store the indices and the maximum values in the output tensor
                x_indices[:, :, pool_row_index, pool_col_index] = x_indices_values
                y_indices[:, :, pool_row_index, pool_col_index] = y_indices_values
                output_tensor[:, :, pool_row_index, pool_col_index] = np.choose(max_values_indices,
                                                                                np.moveaxis(pooling_region, 2, 0))

        return output_tensor


def backward(self, error_tensor):
    # Initialize the return tensor with zeros
    return_tensor = np.zeros(self.lastShape)

    # Iterate over the dimensions of the index tensors
    for batch_idx in range(self.x_indices.shape[0]):
        for channel_idx in range(self.x_indices.shape[1]):
            for pool_row_idx in range(self.x_indices.shape[2]):
                for pool_col_idx in range(self.y_indices.shape[3]):
                    # Calculate the original indices within the input tensor
                    input_row_idx = pool_row_idx * self.stride_shape[0] + self.x_indices[
                        batch_idx, channel_idx, pool_row_idx, pool_col_idx]
                    input_col_idx = pool_col_idx * self.stride_shape[1] + self.y_indices[
                        batch_idx, channel_idx, pool_row_idx, pool_col_idx]

                    # Accumulate the error in the return tensor
                    return_tensor[batch_idx, channel_idx, input_row_idx, input_col_idx] += error_tensor[
                        batch_idx, channel_idx, pool_row_idx, pool_col_idx]

    return return_tensor
