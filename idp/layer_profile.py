import numpy as np
from .coeffs_generator import step


def zero_end(coefs, coef_ratio):
    if coef_ratio is None:
        return np.array(coefs)
    coefs = np.array(coefs)
    coefs[int(coef_ratio * len(coefs)):] = 0
    return coefs


def layer_profile(coeff_generator, start_profile,
                  end_profile, in_channels, out_channels,
                  comp_ratio):
    if type(start_profile) != int or start_profile < 0 or start_profile > 10:
        raise ValueError('start_profile must be between 1 and 10')
    if type(end_profile) != int or end_profile < 0 or end_profile > 10:
        raise ValueError('end_profile must be between 1 and 10')

    bp_steps = np.zeros(10)
    bp_steps[start_profile:end_profile] = 1
    in_steps = step(in_channels, steps=bp_steps)
    out_steps = step(out_channels, steps=bp_steps)
    chan_weights = np.array(coeff_generator(in_channels))
    chan_weights = zero_end(chan_weights, comp_ratio)
    chan_weights = zero_end(chan_weights, end_profile / 10.)
    return chan_weights, in_steps, out_steps
