import numpy as np


class Vector:

    def __init__(self, x, d):
        if len(x.shape) > 2:
            self.__x = list(np.asarray(x).reshape(-2))
        else:
            self.__x = list(x)
        self.__d = d

    def get_x(self):
        return self.__x

    def get_d(self):
        return self.__d
