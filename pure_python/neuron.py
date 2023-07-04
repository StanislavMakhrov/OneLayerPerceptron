from math import exp
from typing import List


def activation_func(net):
    return 1.0 / (1.0 + exp(-net))


class Neuron:

    def __init__(self, bias: float, weights: List[float]):
        self.__weights = [bias] + weights
        self.__out = 0.0
        self.__net = 0.0

    def get_derivative(self):
        return activation_func(self.__net) * (1.0 - activation_func(self.__net))

    def calc_out(self, x):
        self.__net = self.__weights[0]
        for i in range(len(x)):
            self.__net += x[i] * self.__weights[i + 1]
        self.__out = activation_func(self.__net)

    def get_out(self):
        return self.__out

    def correct_weights(self, weights_deltas):
        for i in range(len(self.__weights)):
            self.__weights[i] += weights_deltas[i]

    def get_weights(self):
        return self.__weights

