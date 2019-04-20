import numpy as np


class Vector:

    def __init__(self, x, d):
        if len(x.shape) > 2:
            self.__x = np.asarray(x).reshape(-1)
            self.__x = self.__x.reshape(1, self.__x.shape[0]).astype(np.float32)
        self.__d = d.reshape(1, d.shape[0])

    def get_x(self):
        return self.__x

    def get_d(self):
        return self.__d
