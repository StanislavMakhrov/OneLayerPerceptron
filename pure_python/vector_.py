import numpy as np


class Vector:

    def __init__(self, x, desireOutputs):
        if len(x.shape) > 2:
            self.__x = list(np.asarray(x).reshape(-2))
        else:
            self.__x = list(x)
        self.__desireOutputs = desireOutputs

    def get_x(self):
        return self.__x

    def get_desire_outputs(self):
        return self.__desireOutputs
