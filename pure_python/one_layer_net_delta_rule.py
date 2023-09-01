from one_layer_net_base import OneLayerNetBase


class OneLayerNet(OneLayerNetBase):
    def calc_corrections(self, vector, learning_rate):
        for j in range(len(self.neurons)):
            error = vector.get_desired_outputs()[j] - self.neurons[j].get_out()
            weights_deltas = [0] * len(self.neurons[j].get_weights())
            weights_deltas[0] = learning_rate * error
            for i in range(len(self.neurons[j].get_weights()) - 1):
                weights_deltas[i + 1] = learning_rate * error * vector.get_x()[i]
            self.neurons[j].correct_weights(weights_deltas)

    def activation_func(self, net):
        return 1.0 if net >= 0 else 0.0

    def calc_loss(self, vector):
        loss = 0
        for j in range(len(self.neurons)):
            loss += abs(vector.get_desired_outputs()[j] - self.neurons[j].get_out())
        return loss
