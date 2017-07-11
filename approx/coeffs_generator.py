import numpy

def uniform_seq(n):
    return [1 for i in range(n)]

def harmonic_seq(n,k=1):
    return [1.*k/i for i in range(1,n+1)]

def linear_seq(n,k=1):
    return [1.-(i*1.*k/n) for i in range(n)]

def exp_seq(n,k=1):
    return [numpy.exp(-k*i) for i in range(n)]

def uniform_exp_seq(n,k=1):
    n_half = n//2
    n_rest = n - n//2
    return uniform_seq(n_half) + exp_seq(n_rest,k)

def step(n,steps):
    num = len(steps)
    i = 0
    coeffs = []
    for step in steps:
        coeffs.extend([step for j in range(n//num+1)])
        i = i + 1
    return coeffs[:n]