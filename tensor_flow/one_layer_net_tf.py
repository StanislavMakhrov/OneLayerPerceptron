from tensor_flow import neural_util as nu


class OneLayerNet(object):

    def __init__(self, x, num_classes):

        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes

        # Call the create function to build the computational graph
        self.output = self.create()

    def create(self):
        return nu.fc(self.X, self.X.get_shape()[1], self.NUM_CLASSES, name='one_layer_perceptron')
