#!/usr/bin/env python

import argparse

import chainer
from chainer import reporter
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.functions import softmax_cross_entropy as softmax
import cupy
import numpy as np

from binary.blinear import BinaryLinear
from binary.bconv import BinaryConvolution2D
from binary.function_binary_convolution_2d import binary_convolution_2d
from binary.bst import bst, mbst, mbst_bp

def ordered_do(x, ratio):
    shape = x.shape
    l = int(np.prod(shape))
    n_zeros = int(l * ratio)
    n_ones = l - n_zeros
    mask = cupy.concatenate((cupy.zeros(n_zeros), cupy.ones(n_ones)))
    mask = mask.reshape(shape)
    return mask

def random_do(x, ratio):
    mask = cupy.random.choice([0, 1],
                              size=(np.prod(x.shape),),
                              p=[ratio, 1-ratio])
    mask = mask.reshape(x.shape)
    return mask

def sample_filter(f, ratio=0.5, do_type='ordered'):
    if do_type == 'ordered':
        mask = ordered_do(f, ratio)
    elif do_type == 'random':
        mask = random_do(f, ratio)
    else:
        raise NotImplementedError()

    mask = mask.reshape(f.shape)
    return mask * f

def conv_do(layer, x, ratio=0.5, do_type='random'):
    mask = cupy.zeros(layer.W.shape)
    for wi in range(len(layer.W)):
        if do_type == 'ordered':
            mask[wi] = ordered_do(layer.W[0], ratio)
        elif do_type == 'random':
            mask[wi] = random_do(layer.W[0], ratio)
        else:
            raise NotImplementedError()

    mask = mask.reshape(layer.W.shape)
    h = binary_convolution_2d(x, mask*layer.W, layer.b, layer.stride, layer.pad)
    return h

class BinConvNet(chainer.Chain):
    def __init__(self, n_out):
        super(BinConvNet, self).__init__()
        self.n_out = n_out
        with self.init_scope():
            self.l1 = BinaryConvolution2D(32, 3, pad=1)
            self.bn1 = L.BatchNormalization(32)
            self.l2 = BinaryConvolution2D(64, 3, pad=1)
            self.bn2 = L.BatchNormalization(64)
            self.l3 = BinaryLinear(n_out)

    def __call__(self, x, t, ret_param='loss'):
        h = mbst_bp(self.bn1(self.l1(x)), 2)
        h = mbst_bp(self.bn2(self.l2(h)), 2)
        h = self.l3(h)

        report = {
            'loss': softmax(h, t),
            'acc': F.accuracy(h, t)
        }

        reporter.report(report, self)
        return report[ret_param]

    def approx(self, x, t, ratio, do_type):
        h = mbst_bp(self.bn1(self.l1(x)), 2)
        h = conv_do(self.l2, h, ratio=ratio, do_type=do_type)
        h = mbst_bp(self.bn2(h), 2)
        h = self.l3(h)
        return F.accuracy(h, t)

    def layer(self, x, layer=1):
        h = bst(self.bn1(self.l1(x)))
        if layer == 1:
            return h
        else:
            return bst(self.bn2(self.l2(h)))

    def report_params(self):
        return ['validation/main/acc']

    def param_names(self):
        return  'bin'