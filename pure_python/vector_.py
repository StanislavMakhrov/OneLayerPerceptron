import numpy as np


class Vector:

    def __init__(self, x, desired_outputs):
        if len(x.shape) > 2:
            self.__x = list(np.asarray(x).reshape(-2))
        else:
            self.__x = list(x)
        self.__desired_outputs = desired_outputs

    def get_x(self):
        return self.__x

    def get_desired_outputs(self):
        return self.__desired_outputs
