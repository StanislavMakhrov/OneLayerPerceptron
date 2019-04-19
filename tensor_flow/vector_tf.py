import numpy as np


class Vector:

    def __init__(self, x, desireOutputs):
        if len(x.shape) > 2:
            self.__x = np.asarray(x).reshape(-1)
            self.__x = self.__x.reshape(1, self.__x.shape[0]).astype(np.float32)
        self.__desireOutputs = desireOutputs.reshape(1, desireOutputs.shape[0])

    def get_x(self):
        return self.__x

    def get_desire_outputs(self):
        return self.__desireOutputs
