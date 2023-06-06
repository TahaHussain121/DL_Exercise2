


class Flatten:

    def __init__(self):
        self.trainable = False  # maybe Flaaten class should inherit from the base layer and then change it like in FullycConnected.py for example
        self.shape = None

    def forward(self, input_tensor):
        # store shape of input so you can access it in the backward method
        self.shape = input_tensor.shape

        number_of_weights = 1
        # only flatten from dim 2 onwards, because first dim seems to contain the different weight "blocks"
        for i in list(input_tensor.shape)[1:]:
            number_of_weights *= i

        return np.reshape(input_tensor, (input_tensor.shape[0], number_of_weights))

    def backward(self, error_tensor):

        return np.reshape(error_tensor, self.shape)
