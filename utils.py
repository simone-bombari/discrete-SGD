import numpy as np
from scipy import special


def comul_norm(x):
    return np.sqrt(np.pi) / 2 * (1 + special.erf(x))


def c1_c2(sigma, eta, h1, a1, h2, a2):
    A1 = np.exp(-2 * h1 / (eta * sigma**2))
    A2 = np.exp(-2 * h2 / (eta * sigma**2))
    B1 = np.sqrt(eta * sigma**2 / a1) * comul_norm(np.sqrt(2 * h1 / (eta * sigma**2)))
    B2 = np.sqrt(eta * sigma**2 / a2) * comul_norm(np.sqrt(2 * h2 / (eta * sigma**2)))
    c1 = A2 / (B1*A2 + A1*B2)
    c2 = A1 / (B1*A2 + A1*B2)
    return c1, c2


def pdf_double_well_1d(x, sigma, eta, h1, a1, h2, a2):
    c1, c2 = c1_c2(sigma**2, sigma**2, eta, h1, a1, h2, a2)
    if x >= 0:
        return c2 * np.exp(-a2 * (x - np.sqrt(2 * h2 / a2))**2 / (eta * sigma**2))
    if x < 0:
        return c1 * np.exp(-a1 * (x + np.sqrt(2 * h1 / a1))**2 / (eta * sigma**2))


def pdf_single_well_1d(x, sigma, eta, a):
    c = np.sqrt(a / (np.pi * eta * sigma**2))
    return c * np.exp(-a * x**2 / (eta * sigma**2))
