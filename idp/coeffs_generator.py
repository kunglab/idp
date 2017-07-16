import numpy as np


def uniform(n):
    return [1 for i in range(n)]


def harmonic(n, k=1):
    return [1. * k / i for i in range(1, n + 1)]


def linear(n, k=1):
    return [1. - (i * 1. * k / n) for i in range(n)]


def exp(n, k=1):
    return [np.exp(-k * i) for i in range(n)]


def uniform_exp(n, k=1):
    n_half = n // 2
    n_rest = n - n // 2
    return uniform(n_half) + exp(n_rest, k)


def step(n, steps=[1]):
    num = len(steps)
    i = 0
    coeffs = []
    for step in steps:
        coeffs.extend([step for j in range(int(np.ceil(1. * n / num)))])
        i = i + 1
    return coeffs[:n]

def three_steps(n):
    return step(n, steps=[1, 0.5, 0.25])

def four_steps(n):
    return step(n, steps=[1, 0.5, 0.25, 0.0])

def mag_steps(n):
    return step(n, steps=[1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])