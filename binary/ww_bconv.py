import math

import numpy

from chainer.functions.connection import convolution_2d
from chainer import link
from chainer import initializers
from chainer import cuda
from chainer.utils import argument
from chainer import variable
from chainer import initializer

from binary.function_ww_binary_convolution_2d import ww_binary_convolution_2d

from scipy.misc import factorial

def harmonic_seq(W_shape):
    n = numpy.prod(W_shape)
    return [2./i for i in range(1,n+1)]

def harmonic_seq_group(W_shape):
    out_channels, in_channels, kh, kw = W_shape
    n = out_channels
    return numpy.asarray([[2./i]*in_channels*kh*kw for i in range(1,n+1)]).flatten()

def harmonic_seq_group_inout(W_shape):
    out_channels, in_channels, kh, kw = W_shape
    n = out_channels*in_channels
    return numpy.asarray([[2./i]*kh*kw for i in range(1,n+1)]).flatten()

def harmonic_seq_group_half_one(W_shape):
    n = numpy.prod(W_shape)
    out_channels, in_channels, kh, kw = W_shape
    half = out_channels//2
    rest = out_channels - half
    first = list(numpy.asarray([[1]*in_channels*kh*kw for i in range(1,half+1)]).flatten())
    second = list(numpy.asarray([[2./i]*in_channels*kh*kw for i in range(1,rest+1)]).flatten())
    return first + second


class WWBinaryConvolution2D(link.Link):
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 mode='harmonic_seq', nobias=False, initialW=None, initial_bias=None, **kwargs):
        super(WWBinaryConvolution2D, self).__init__()

        argument.check_unexpected_kwargs(
            kwargs, deterministic="deterministic argument is not "
            "supported anymore. "
            "Use chainer.using_config('cudnn_deterministic', value) "
            "context where value is either `True` or `False`.")
        argument.assert_kwargs_empty(kwargs)

        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.out_channels = out_channels
        self.mode = mode

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
        if self.mode == 'harmonic_seq':
            self.weight = harmonic_seq(W_shape)
        elif self.mode == 'harmonic_seq_group':
            self.weight = harmonic_seq_group(W_shape)
        elif self.mode == 'harmonic_seq_group_inout':
            self.weight = harmonic_seq_group_inout(W_shape)
        elif self.mode == 'harmonic_seq_group_half_one':
            self.weight = harmonic_seq_group_half_one(W_shape)
        
    def __call__(self, x, ratio=1):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return ww_binary_convolution_2d(self.weight, 
            x, self.W, self.b, self.stride, self.pad, ratio=ratio)

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
