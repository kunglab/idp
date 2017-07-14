from functools import partial

import numpy as np


def gen_blackout(dist):
    if dist == 'exp':
        return exp(w=0.25)
    if dist == 'slow_exp':
        return exp(w=0.1)
    if dist == 'linear':
        return linear()
    if dist == 'all':
        return 0
    else:
        raise NameError('Blackout function: {}'.format(dist))


def exp(w=0.16666):
    while True:
        do = np.random.exponential(w)
        if do <= 1.0:
            break
    return do


def linear(w=10):
    w += 1
    weights = np.linspace(0, 1, w) / np.sum(np.linspace(0, 1, w))
    return 1 - np.random.choice(range(w), p=weights) / float(w)
