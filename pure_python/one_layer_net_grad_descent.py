from math import pow, exp
from one_layer_net_base import OneLayerNetBase


def get_derivative(net, act_func):
    return act_func(net) * (1.0 - act_func(net))


class OneLayerNet(OneLayerNetBase):
    def calc_corrections(self, vector, learning_rate):
        for j in range(len(self.neurons)):
            sigma = (vector.get_desired_outputs()[j] - self.neurons[j].get_out()) * \
                    get_derivative(self.neurons[j].get_out(), self.activation_func)
            weights_deltas = [0] * len(self.neurons[j].get_weights())
            weights_deltas[0] = learning_rate * sigma
            for i in range(len(self.neurons[j].get_weights()) - 1):
                weights_deltas[i + 1] = learning_rate * sigma * vector.get_x()[i]
            self.neurons[j].correct_weights(weights_deltas)

    def activation_func(self, net):
        return 1.0 / (1.0 + exp(-net))

    def calc_loss(self, vector):
        loss = 0
        for j in range(len(self.neurons)):
            loss += pow(vector.get_desired_outputs()[j] - self.neurons[j].get_out(), 2)
        return 0.5 * loss
