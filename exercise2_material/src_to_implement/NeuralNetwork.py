from copy import deepcopy


class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.w_ini = weights_initializer
        self.b_ini = bias_initializer

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next() #tensoris just fancy word for array?
        self.label_tensor = label_tensor
        output = input_tensor
        for layer in self.layers:
            #print('self.layers\n', self.layers)
            output = layer.forward(output)
        #print('label tensor\n', label_tensor)
        return self.loss_layer.forward(output, label_tensor)

    def backward(self, label_tensor):
        error_tensor = self.loss_layer.backward(label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.initialize(self.w_ini, self.b_ini)    # extension from ex2 1.
            layer.optimizer = deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            loss = self.forward()
            #loss = self.loss_layer.forward(prediction, self.data_layer.label_tensor)
            self.loss.append(loss)
            self.backward(self.label_tensor)
            # for layer in self.layers:
            #     if layer.trainable:
            #         layer.weights = layer.optimizer.calculate_update(layer.weights, layer.gradient_weights)

    def test(self, input_tensor):
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)
        return output
