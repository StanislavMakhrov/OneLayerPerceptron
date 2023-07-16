from pure_python.delta_rule.neuron import Neuron
import random


def generate_weights(weights_count, range_min=-0.0003, range_max=0.0003):
    return [random.SystemRandom().uniform(range_min, range_max) for _ in range(weights_count)]


def generate_bias(range_min=-0.0003, range_max=0.0003):
    return random.SystemRandom().uniform(range_min, range_max)


class OneLayerNet:
    def __init__(self, input_vector_size, output_neurons_count):
        self.input_vector_size = input_vector_size
        self.__neurons = []
        for j in range(output_neurons_count):
            self.__neurons.append(Neuron(generate_bias(), generate_weights(input_vector_size)))

    def train(self, vector, learning_rate):

        for j in range(len(self.__neurons)):
            self.__neurons[j].calc_out(vector.get_x())

        for j in range(len(self.__neurons)):
            error = vector.get_desired_outputs()[j] - self.__neurons[j].get_out()
            weights_deltas = [0] * len(self.__neurons[j].get_weights())
            weights_deltas[0] = learning_rate * error
            for i in range(len(self.__neurons[j].get_weights()) - 1):
                weights_deltas[i + 1] = learning_rate * error * vector.get_x()[i]
            self.__neurons[j].correct_weights(weights_deltas)

        loss = 0
        for j in range(len(self.__neurons)):
            loss += abs(vector.get_desired_outputs()[j] - self.__neurons[j].get_out())

        return loss

    def test(self, vector):
        y = [0] * len(self.__neurons)
        for j in range(len(self.__neurons)):
            self.__neurons[j].calc_out(vector.get_x())
            y[j] = self.__neurons[j].get_out()
        return y
