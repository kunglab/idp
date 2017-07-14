import numpy


def uniform(n):
    return [1 for i in range(n)]


def harmonic(n, k=1):
    return [1. * k / i for i in range(1, n + 1)]


def linear(n, k=1):
    return [1. - (i * 1. * k / n) for i in range(n)]


def exp(n, k=1):
    return [numpy.exp(-k * i) for i in range(n)]


def uniform_exp(n, k=1):
    n_half = n // 2
    n_rest = n - n // 2
    return uniform(n_half) + exp(n_rest, k)


def step(n, steps=[1]):
    num = len(steps)
    i = 0
    coeffs = []
    for step in steps:
        coeffs.extend([step for j in range(int(round(1. * n / num)))])
        i = i + 1
    return coeffs[:n]
