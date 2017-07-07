import numpy

def uniform_seq(n):
    return [1 for i in range(n)]
def harmonic_seq(n):
    return [1./i for i in range(1,n+1)]
def linear_seq(n):
    return [1.-(i*1./n) for i in range(n)]
def exp_seq(n):
    return [numpy.exp(-i) for i in range(n)]
def uniform_exp_seq(n):
    n_half = n//2
    n_rest = n - n//2
    return uniform_seq(n_half) + exp_seq(n_rest)
