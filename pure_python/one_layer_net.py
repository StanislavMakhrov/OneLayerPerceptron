from pure_python.neuron import Neuron
from math import pow


class OneLayerNet:

    def __init__(self, inputs_count, output_neurons_count):
        self.__inputs_count = inputs_count
        self.__neurons = []
        for j in range(output_neurons_count):
            self.__neurons.append(Neuron(inputs_count))

    def train(self, vector, learning_rate):

        for j in range(len(self.__neurons)):
            self.__neurons[j].calc_y(vector.get_x())

        weights_deltas = [[0] * (len(vector.get_x()) + 1)] * len(self.__neurons)

        for j in range(len(self.__neurons)):
            sigma = (vector.get_d()[j] - self.__neurons[j].get_y()) * self.__neurons[j].get_derivative()
            weights_deltas[j][0] = learning_rate * sigma
            for i in range(len(self.__neurons[j].get_weights())):
                weights_deltas[j][i] = learning_rate * sigma * vector.get_x()[i]
            self.__neurons[j].correct_weights(weights_deltas[j])

        loss = 0
        for j in range(len(self.__neurons)):
            loss += pow(vector.get_d()[j] - self.__neurons[j].get_y(), 2)

        return 0.5 * loss

    def test(self, vector):
        y = [0] * len(self.__neurons)
        for j in range(len(self.__neurons)):
            self.__neurons[j].calc_y(vector.get_x())
            y[j] = self.__neurons[j].get_y()
        return y
