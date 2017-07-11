from approx.coeffs_generator import *
from functools import partial
uniform_seq(10)
partial(harmonic_seq,k=2)(10)
partial(linear_seq,k=2)(10)
partial(exp_seq,k=2)(10)
partial(uniform_exp_seq,k=2)(10)
partial(step,steps=[0.75,0.25])(10)