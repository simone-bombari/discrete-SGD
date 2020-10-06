import numpy as np


def single_well_1d(x, a):
    return a * x**2 / 2


def single_well_2d(x, y, a, b):
    return a * x**2 + b * y**2


def grad_single_well_1d(x, a):
    return a * x


def grad_single_well_2d(x, y, a, b):
    return 2 * a * x + 2 * b * y


def double_well_1d(x, h1, a1, h2, a2):
    if x >= 0:
        return x * (a2 / 2 * x - np.sqrt(2 * a2 * h2))
    else:
        return x * (a1 / 2 * x + np.sqrt(2 * a1 * h1))


def double_well_2d(x, y, h1, a1, h2, a2, b1, b2):
    if x >= 0:
        return double_well_1d(x, h1, a1, h2, a2) + b2/2 * y**2
    else:
        return double_well_1d(x, h1, a1, h2, a2) + b1/2 * y**2


def grad_double_well_1d(x, h1, a1, h2, a2):
    if x >= 0:
        return x * a2 - np.sqrt(2 * a2 * h2)
    else:
        return x * a1 + np.sqrt(2 * a1 * h1)


def grad_double_well_2d(x, y, h1, a1, h2, a2, b1, b2):
    if x >= 0:
        return [x * a2 - np.sqrt(2 * a2 * h2), b2 * y]
    else:
        return [x * a1 + np.sqrt(2 * a1 * h1), b1 * y]
