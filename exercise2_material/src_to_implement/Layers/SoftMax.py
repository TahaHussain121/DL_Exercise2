import numpy as np
from Layers import Base


class SoftMax(Base.BaseLayer):

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_tensor = None

    def forward(self, input_tensor):    # softmax applied "row-wise" as described in pdf
        input_tensor = input_tensor - np.max(input_tensor)
        input_tensor = np.exp(input_tensor)
        sum = np.sum(input_tensor, axis=1)
        sum = sum[..., np.newaxis]
        self.input_tensor = input_tensor / sum
        #print('prediction tensor', self.input_tensor)
        return self.input_tensor

    def backward(self, error_tensor):
        #print('error_tensor', error_tensor)
        soft = self.input_tensor
        #temp = error_tensor - np.sum(np.diag(error_tensor.T @ soft), axis=0)

        temp = 0
        for j in range(0, soft.shape[1]):
            temp += np.multiply(error_tensor[:, j], soft[:, j])
        temp = (error_tensor.T - temp).T

        return np.multiply(soft, temp)      # element wise multiplication
