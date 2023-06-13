from Layers import Base
import numpy as np
from scipy.signal import fftconvolve, correlate, correlate2d

class Conv(Base.BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        print('stride_shape = ', self.stride_shape)
        self._optimizer = None
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.array([list(np.random.random(self.convolution_shape)) for k in range(self.num_kernels)])
        self.bias = np.random.random(self.num_kernels)
        self.input_tensor_shape = None
        self.input_tensor = None
        self.gradient_kernels = None
        self.output_of_forward = None
        self.gradient_b = None

    def forward(self, input_tensor):
        #print('input_tensor.shape = ', input_tensor.shape, '\ninput_tensor\n', input_tensor)
        self.input_tensor = input_tensor
        self.input_tensor_shape = input_tensor.shape
        # output convolution array: 1dim -> contains the batch elements of inpout_tensor,
        #                           2dim -> contains the batch elements of the self.weights,
        #                           3dim -> contains the y axis elements of the input_tensor,
        #                           4dim -> contains the x axis elements of the input_tensor, this dim must not always exist -> if clause needed

        # list for appending all the batch elements of the input tensor
        # conv_array includes the strides
        conv_array = []
        # output_without_stride is the output of forward wihtout strides
        #output_without_stride = []
        # iterate over the input_tensor batch
        for b in range(0, input_tensor.shape[0]):
            # list for appending all the batch elements of weight array
            conv_list = []
            #conv_list_without_stride = []
            # iterate over the weight batch
            for batch_weights_index in range(0, self.num_kernels):
                #print('\ninput_tensor[b]\n', input_tensor[b], '\n\nself.weights.shape[0] aka batch shape\n', self.weights.shape[0])
                temp = 0
                # convolute with mode same per depth, then add the outputs together
                for depth_index in range(0, input_tensor.shape[1]):
                    #why = np.flip(self.weights[batch_weights_index][depth_index])
                    temp += correlate(input_tensor[b][depth_index], self.weights[batch_weights_index][depth_index], mode='same')

                # adding bias
                temp += self.bias[batch_weights_index]
                #conv_list_without_stride.append(temp)
                if self.stride_shape[0] > 1 or self.stride_shape[1] > 1:
                    # we have to take the stripes depending on the shape of input_tensor being (depth,y) or (depth,y,x)
                    if len(input_tensor.shape) == 3:
                        # for some reason stride shape give can be either [x] or (x,y), no clue why the datatype is different
                        # -> we need another if clause
                        if len(self.stride_shape) == 1:
                            #print('\nconvolution output after stride = \n', temp[::self.stride_shape[0]])
                            conv_list.append(list(temp[::self.stride_shape[0]]))
                            #print('\ncase1\n')
                            continue
                        else:
                            print('ERROR: strid_shape is tuple of length 2 but input tensor has no width')
                            continue
                    else:
                        if len(self.stride_shape) == 1:
                            conv_list.append(list(temp[::self.stride_shape[0], ::self.stride_shape[0]]))
                            #print('case2')
                            continue
                        else:
                            conv_list.append(list(temp[::self.stride_shape[0], ::self.stride_shape[1]]))
                            #print('case3')
                            continue

                conv_list.append(temp)
            conv_array.append(conv_list)
            #output_without_stride.append(conv_list_without_stride)
        #self.output_of_forward = np.array(output_without_stride)
        conv_array = np.array(conv_array)
        #print('\nconvolution output array, should have a shape of (tensor batch, weight batch, convolution ouput length\n', conv_array)
        return conv_array

    def backward(self, error_tensor):
        self.output_of_forward = error_tensor
        #print('\nerror_tensor for backpropagation\n', error_tensor)
        print('\nerror_tensor shape = ', error_tensor.shape, '\nkernel shape = ', self.weights.shape)
        stride_bool = False

        # list for appending all the batch elements of the input tensor
        conv_array = []
        # outmost list (batch axis) for grabbing just the padded error tensor
        padded_error_list_alls_channels_batch = []
        # iterate over the input_tensor batch
        for b in range(0, error_tensor.shape[0]):
            # list for appending all the batch elements of weight array
            conv_list = []

            # iterate over the weight batch
            for depth_index in range(0, self.weights.shape[1]):
                temp = 0
                # list for grabbing just the padded error tensor
                padded_error_list_all_channels = []
                for batch_weights_index in range(0, self.num_kernels):
                    # cross correlate with mode same per depth, then add the outputs together
                    # but before decide if error tensor has x,y spatial dim or only x (we do this by checking the length of the error_tensor shape)
                    padded_error_list = []
                    if len(error_tensor.shape) == 3:
                        # if self.stride > 1 we need to pad zeros back in to get the correct size so correlation can lead to the input.shape
                        if self.stride_shape[0] > 1:
                            stride_bool = True
                            index_location = np.tile(np.arange(1, error_tensor[b][batch_weights_index].shape[0]), self.stride_shape[0]-1)
                            padded_error_tensor = np.insert(error_tensor[b][batch_weights_index], index_location, np.zeros(error_tensor[b][batch_weights_index][index_location].shape))
                            # the following line appears in every if clause, when stride is >1. It is saving the padded error tensor which is used in backward pass
                            padded_error_list = padded_error_tensor.tolist()
                        else:
                            padded_error_tensor = error_tensor[b][batch_weights_index]

                        temp += fftconvolve(padded_error_tensor, self.weights[self.weights.shape[0] - 1 - batch_weights_index][depth_index], mode='same')
                    else:
                        if self.stride_shape[0] > 1 or self.stride_shape[1] > 1:
                            stride_bool = True
                            # big problem in this whole ifclause: so far I only pad once on the outside to fit the input_tensor.shape,
                            # it is highly likely that you need to fit more (e.g. [1,0,2,0,3,0,0,0]

                            # padding on axis 0
                            index_location = np.tile(np.arange(1, error_tensor[b][batch_weights_index].shape[0]), self.stride_shape[0] - 1)
                            padded_error_tensor_x = np.insert(error_tensor[b][batch_weights_index], index_location, np.zeros(error_tensor[b][batch_weights_index][index_location].shape), axis=0)
                            # special case that padding needs to be expanded on the outside of the array (e.g [1,0,2,0,3,0] <- here the 0 is padded outside)
                            if padded_error_tensor_x.shape[0] != self.input_tensor_shape[2]:
                                index_location = np.tile(np.arange(0, error_tensor[b][batch_weights_index].shape[0]), self.stride_shape[0] - 1)
                                padded_error_tensor_x = np.insert(error_tensor[b][batch_weights_index], index_location, np.zeros(error_tensor[b][batch_weights_index][index_location].shape), axis=0)

                            # padding on axis 1
                            index_location = np.tile(np.arange(1, padded_error_tensor_x.shape[1]), self.stride_shape[1] - 1)
                            padded_error_tensor_y = np.insert(padded_error_tensor_x, index_location, np.zeros(padded_error_tensor_x[1, index_location].shape), axis=1)
                            padded_error_tensor = padded_error_tensor_y
                            padded_error_list = padded_error_tensor.tolist()    # again this overrides the full error_tensor to the padded error_tensor because strides took place
                            if padded_error_tensor_y.shape[1] != self.input_tensor_shape[3]:
                                index_location = np.tile(np.arange(0, padded_error_tensor_x.shape[1]), self.stride_shape[1] - 1)
                                padded_error_tensor = np.insert(padded_error_tensor_x, index_location, np.zeros(padded_error_tensor_x[1, index_location].shape), axis=1)
                                padded_error_list = padded_error_tensor.tolist()    # again this overrides the full error_tensor to the padded error_tensor because strides took place
                                #print('\npadded error tensor\n', padded_error_tensor)

                        else:
                            padded_error_tensor = error_tensor[b][batch_weights_index]
                        #print('padded_error_tensor\n', padded_error_tensor)
                        temp += fftconvolve(padded_error_tensor, self.weights[batch_weights_index][depth_index], mode='same')
                    padded_error_list_all_channels.append(padded_error_list)
                #print('\npadded_error_list_all_channels\n', padded_error_list_all_channels)
                conv_list.append(list(temp))
            padded_error_list_alls_channels_batch.append(padded_error_list_all_channels)
            #print('\npadded_error_list_all_channels\n', padded_error_list_all_channels)
            conv_array.append(conv_list)
        conv_array = np.array(conv_array)

        # if clause for when stride > 1, because then we need a padded error tensor
        if stride_bool:
            self.output_of_forward = np.array(padded_error_list_alls_channels_batch)
            print('\npadded_error_list_all_channels_batch\n', self.output_of_forward.shape)

        # -- Now compute the gradient kernels to update the existing kernels (see slide 22)

        # check whether our kernel has 2 dim or 3 dim
        flag_kernel_dim = True
        if len(self.convolution_shape) == 2:
            flag_kernel_dim = False
            half_kernel_length = self.convolution_shape[1] // 2
            # If the spatial dim is an even number you only pad on one side
            if self.convolution_shape[1] % 2 == 0:
                padded_input_tensor = np.pad(self.input_tensor, ((0, 0), (0, 0), (half_kernel_length - 1, half_kernel_length)))
            else:
                padded_input_tensor = np.pad(self.input_tensor, ((0, 0), (0, 0), (half_kernel_length, half_kernel_length)))
        else:
            # If one of the spatial dimensions of the kernel is an even number you have to pad unevenly in this corresponding side
            if self.convolution_shape[1] % 2 == 0:
                half_kernel_length = self.convolution_shape[1] // 2
                half_kernel_width = self.convolution_shape[2] // 2
                if flag_kernel_dim and self.convolution_shape[2] % 2 == 0:
                    padded_input_tensor = np.pad(self.input_tensor, ((0, 0), (0, 0), (half_kernel_length-1, half_kernel_length), (half_kernel_width-1, half_kernel_width)))
                else:
                    padded_input_tensor = np.pad(self.input_tensor, ((0, 0), (0, 0), (half_kernel_length-1, half_kernel_length), (half_kernel_width, half_kernel_width)))
            else:
                half_kernel_length = self.convolution_shape[1] // 2
                half_kernel_width = self.convolution_shape[2] // 2
                padded_input_tensor = np.pad(self.input_tensor, ((0, 0), (0, 0), (half_kernel_length, half_kernel_length), (half_kernel_width, half_kernel_width)))

            if flag_kernel_dim and self.convolution_shape[2] % 2 == 0:
                half_kernel_length = self.convolution_shape[1] // 2
                half_kernel_width = self.convolution_shape[2] // 2
                if self.convolution_shape[1] % 2 == 0:
                    padded_input_tensor = np.pad(self.input_tensor, ((0, 0), (0, 0), (half_kernel_length-1, half_kernel_length), (half_kernel_width-1, half_kernel_width)))
                else:
                    padded_input_tensor = np.pad(self.input_tensor, ((0, 0), (0, 0), (half_kernel_length, half_kernel_length), (half_kernel_width-1, half_kernel_width)))

        print('input tensor shape = ', self.input_tensor.shape, 'padded input_tensor shape = ',
              padded_input_tensor.shape, 'error tensor shape', self.output_of_forward.shape)

        # here the actual gradient computation takes place (see slide 22)
        gradient_kernels_batch = []
        for b in range(0, self.input_tensor_shape[0]):
            gradient_kernels = []
            for index_kernels in range(0, self.num_kernels):
                temp = []
                for depth_index in range(0, self.input_tensor_shape[1]):
                    why = correlate(padded_input_tensor[b][depth_index], self.output_of_forward[b][index_kernels], mode='valid')
                    temp.append(why.tolist())

                gradient_kernels.append(temp)
            gradient_kernels_batch.append(gradient_kernels)
        # all the gradients of all the kernels for each batch (as a numpy array)
        gradient_kernels_batch = np.array(gradient_kernels_batch)

        print('\ngradient_kernels_batch shape =', gradient_kernels_batch.shape)
        self.gradient_kernels = np.sum(gradient_kernels_batch, axis=0)

        # gradient bias (with if clause if input spatial dim is only 1)
        if len(error_tensor.shape) == 3:
            self.gradient_b = np.sum(self.output_of_forward, axis=(0, 2))
        else:
            self.gradient_b = np.sum(self.output_of_forward, axis=(0, 2, 3))
        print('\ngradient kernels shape = ', self.gradient_kernels.shape, '\ngradient kernels\n', self.gradient_kernels)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_kernels)
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_b)
        return conv_array

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, self.convolution_shape[0] * self.convolution_shape[1] * self.convolution_shape[2],
                                                      self.num_kernels * self.convolution_shape[1] * self.convolution_shape[2])
        # the bias initialization is probably false
        self.bias = bias_initializer.initialize(self.num_kernels, 0, 0)

    @property
    def gradient_weights(self):
        return self.gradient_kernels

    @property
    def gradient_bias(self):
        return self.gradient_b

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt_function):
        self._optimizer = opt_function
