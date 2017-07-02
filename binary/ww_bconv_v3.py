import math

import numpy

from chainer.functions.connection import convolution_2d
from chainer import link
from chainer import initializers
from chainer import cuda
from chainer.utils import argument
from chainer import variable
from chainer import initializer

from binary.function_ww_binary_convolution_2d_v3 import ww_binary_convolution_2d_v3

from scipy.misc import factorial

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

class WWBinaryConvolution2DV3(link.Link):
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 coeffs_generator=harmonic_seq, nobias=False, initialW=None, initial_bias=None, **kwargs):
        super(WWBinaryConvolution2DV3, self).__init__()

        argument.check_unexpected_kwargs(
            kwargs, deterministic="deterministic argument is not "
            "supported anymore. "
            "Use chainer.using_config('cudnn_deterministic', value) "
            "context where value is either `True` or `False`.")
        argument.assert_kwargs_empty(kwargs)
        
        self.coeffs_generator = coeffs_generator

        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.out_channels = out_channels
        
        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            if in_channels is not None:
                self._initialize_params(in_channels)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_channels)

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        W_shape = (self.out_channels, in_channels, kh, kw)
        self.W.initialize(W_shape)
        
    def __call__(self, x, ratio=1, coeffs_generator=None):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        _,in_channels,_,_ = self.W.shape
        if coeffs_generator is None:
            weight = self.coeffs_generator(in_channels)
        else:
            weight = coeffs_generator(in_channels)
                
        return ww_binary_convolution_2d_v3(weight, 
            x, self.W, self.b, self.stride, self.pad, ratio=ratio)

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
