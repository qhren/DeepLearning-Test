import numpy as np


def soft_max(a):
    return np.exp(a)/np.sum(np.exp(a))

