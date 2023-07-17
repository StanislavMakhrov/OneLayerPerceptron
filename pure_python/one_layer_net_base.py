from pure_python.neuron import Neuron
from abc import ABC, abstractmethod
import random


def generate_bias(range_min=-0.0003, range_max=0.0003):
    return random.SystemRandom().uniform(range_min, range_max)


def generate_weights(weights_count, range_min=-0.0003, range_max=0.0003):
    return [random.SystemRandom().uniform(range_min, range_max) for _ in range(weights_count)]


class OneLayerNetBase(ABC):
    def __init__(self, input_vector_size, output_neurons_count):
        self.input_vector_size = input_vector_size
        self.neurons = []
        for j in range(output_neurons_count):
            self.neurons.append(Neuron(generate_bias(),
                                       generate_weights(input_vector_size),
                                       self.activation_func))

    def train(self, vector, learning_rate):
        self.calc_outs(vector)
        self.calc_corrections(vector, learning_rate)
        return self.calc_loss(vector)

    @abstractmethod
    def activation_func(self, net):
        pass

    def calc_outs(self, vector):
        for j in range(len(self.neurons)):
            self.neurons[j].calc_out(vector.get_x())

    @abstractmethod
    def calc_corrections(self, vector, learning_rate):
        pass

    @abstractmethod
    def calc_loss(self, vector):
        pass

    def test(self, vector):
        y = [0] * len(self.neurons)
        for j in range(len(self.neurons)):
            self.neurons[j].calc_out(vector.get_x())
            y[j] = self.neurons[j].get_out()
        return y
