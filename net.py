#!/usr/bin/env python

import argparse
from functools import partial

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
from binary.m_bconv import MBinaryConvolution2D
from binary.ss_bconv import SSBinaryConvolution2D
from binary.ww_bconv import WWBinaryConvolution2D
from binary.ww_bconv_v3 import WWBinaryConvolution2DV3, uniform_seq, harmonic_seq, linear_seq, exp_seq, uniform_exp_seq
from binary.function_binary_convolution_2d import binary_convolution_2d
from binary.bst import bst, mbst, mbst_bp
from approx.links_depthwise_convolution_2d import IncompleteDepthwiseConvolution2D
from approx.links_convolution_2d import IncompleteConvolution2D
from approx.links_linear import IncompleteLinear
import util


class BinaryBlock(chainer.Chain):
    def __init__(self, num_fs, ksize=3, pksize=2):
        super(BinaryBlock, self).__init__()
        self.pksize = pksize

        if isinstance(num_fs, (int),):
            l1_f = l2_f = num_fs
        else:
            l1_f, l2_f = num_fs[0], num_fs[1]

        with self.init_scope():
            self.l1 = BinaryConvolution2D(l1_f, ksize, pad=1)
            self.bn1 = L.BatchNormalization(l1_f)
            self.l2 = BinaryConvolution2D(l2_f, ksize, pad=1)
            self.bn2 = L.BatchNormalization(l2_f)

    def __call__(self, x):
        h = F.relu(self.bn1(self.l1(x)))
        h = F.max_pooling_2d(h, self.pksize, stride=1)
        h = F.relu(self.bn2(self.l2(h)))
        return h


class ApproxBlock(chainer.Chain):
    def __init__(self, num_fs, ksize=3, pksize=2, m=1, comp_f='exp',
                 act='ternary', coeffs_generator=uniform_seq):
        super(ApproxBlock, self).__init__()
        self.comp_f = comp_f
        self.m = m
        self.pksize = pksize
        self.coeffs_generator = coeffs_generator

        if isinstance(num_fs, (int),):
            l1_f = l2_f = num_fs
        else:
            l1_f, l2_f = num_fs[0], num_fs[1]

        self.l1_act = partial(mbst_bp, m=self.m)
        if act == 'ternary':
            self.act = partial(mbst_bp, m=self.m)
        elif act == 'binary':
            self.act = bst
        elif act == 'relu':
            self.act = F.relu
        else:
            raise NameError("act={}".format(act))

        with self.init_scope():
            self.l1 = BinaryConvolution2D(
                l1_f, 1, pad=0)
            self.bn1 = L.BatchNormalization(l1_f)
            self.l2 = WWBinaryConvolution2DV3(
                l2_f, ksize, pad=1, coeffs_generator=self.coeffs_generator)
            self.bn2 = L.BatchNormalization(l2_f)

    def __call__(self, x, comp_ratio=None, ret_param='loss', coeffs_generator=None):
        if not comp_ratio:
            comp_ratio = 1 - util.gen_prob(self.comp_f)

        h = self.l1(x)
        h = self.l1_act(self.bn1(h))
        h = self.l2(h, ratio=comp_ratio,
                    coeffs_generator=coeffs_generator or self.coeffs_generator)
        h = F.max_pooling_2d(h, self.pksize, stride=1)
        h = self.act(self.bn2(h))
        return h


class IncompleteDepthwiseBlock(chainer.Chain):
    def __init__(self, in_chan, num_f, coeffs_generator, act, stride=1):
        super(IncompleteDepthwiseBlock, self).__init__()
        self.in_chan = in_chan
        self.num_f = num_f
        self.coeffs_generator = coeffs_generator
        self.act = act
        with self.init_scope():
            self.dc = IncompleteDepthwiseConvolution2D(
                self.in_chan, 1, 3, pad=1, stride=stride)
            self.bn1 = L.BatchNormalization(self.in_chan)
            self.pc = IncompleteConvolution2D(self.num_f, 1)
            self.bn2 = L.BatchNormalization(self.num_f)

    def __call__(self, x, comp_ratio=None):
        def coeff_f(n):
            return util.zero_end(self.coeffs_generator(n), comp_ratio)

        h = self.dc(x, coeff_f(self.in_chan))
        h = self.bn1(h)
        h = self.act(h)
        h = self.pc(h, coeff_f(self.in_chan))
        h = self.bn2(h)
        h = self.act(h)
        return h


class ApproxNet(chainer.Chain):
    def __init__(self, n_out, l1_f, l2_f=None, l3_f=None, m=0, comp_f='exp',
                 act='ternary', coeffs_generator=uniform_seq):
        super(ApproxNet, self).__init__()
        self.n_out = n_out
        self.comp_f = comp_f
        self.l1_f = l1_f
        self.l2_f = l2_f
        self.l3_f = l3_f
        self.coeffs_generator = coeffs_generator
        self.act = act
        self.m = m

        if not l2_f and l3_f:
            raise ValueError("l2_f must be set if l3_f is set.")

        with self.init_scope():
            self.l1 = ApproxBlock(l1_f, m=m, comp_f=comp_f,
                                  act=act, coeffs_generator=coeffs_generator)
            if l2_f:
                self.l2 = BinaryBlock(l2_f)
            if l3_f:
                self.l3 = BinaryBlock(l3_f)
            self.l4 = BinaryLinear(n_out)

    def __call__(self, x, t, comp_ratio=None, ret_param='loss', coeffs_generator=None):
        h = self.l1(x, comp_ratio,
                    coeffs_generator=coeffs_generator or self.coeffs_generator)
        if self.l2_f:
            h = self.l2(h)
        if self.l3_f:
            h = self.l3(h)
        h = self.l4(h)

        report = {
            'loss': softmax(h, t),
            'acc': F.accuracy(h, t)
        }

        reporter.report(report, self)
        return report[ret_param]

    def report_params(self):
        return ['validation/main/acc']

    def param_names(self):
        l1_f = util.layers_str(self.l1_f)
        l2_f = util.layers_str(self.l2_f)
        l3_f = util.layers_str(self.l3_f)
        return 'approx_[{},{},{}]_{}_[{}:{}]'.format(l1_f, l2_f, l3_f,  self.coeffs_generator.__name__,
                                                     self.act, self.m)


class BinaryNet(chainer.Chain):
    def __init__(self, n_out, l1_f, l2_f=None, l3_f=None, coeffs_generator=uniform_seq):
        super(BinaryNet, self).__init__()
        self.l1_f = l1_f
        self.l2_f = l2_f
        self.l3_f = l3_f
        self.coeffs_generator = coeffs_generator

        if not l2_f and l3_f:
            raise ValueError("l2_f must be set if l3_f is set.")

        with self.init_scope():
            self.l1 = ApproxBlock(
                l1_f, m=1, comp_f='id', act='ternary', coeffs_generator=coeffs_generator)
            if l2_f:
                self.l2 = BinaryBlock(l2_f)
            if l3_f:
                self.l3 = BinaryBlock(l3_f)
            self.l4 = BinaryLinear(n_out)

    def __call__(self, x, t, comp_ratio=None, filter_ratio=None, ret_param='loss', coeffs_generator=None):
        h = self.l1(x, comp_ratio, filter_ratio,
                    coeffs_generator=coeffs_generator or self.coeffs_generator)
        if self.l2_f:
            h = self.l2(h)
        if self.l3_f:
            h = self.l3(h)
        h = self.l4(h)

        report = {
            'loss': softmax(h, t),
            'acc': F.accuracy(h, t)
        }

        reporter.report(report, self)
        return report[ret_param]

    def report_params(self):
        return ['validation/main/acc']

    def param_names(self):
        l1_f = util.layers_str(self.l1_f)
        l2_f = util.layers_str(self.l2_f)
        l3_f = util.layers_str(self.l3_f)
        return 'standard_[{},{},{}]'.format(l1_f, l2_f, l3_f)


class IncompleteDepthwiseNet(chainer.Chain):
    def __init__(self, coeffs_generator, act):
        super(IncompleteDepthwiseNet, self).__init__()
        self.coeffs_generator = coeffs_generator
        self.act = act
        self.l1_f = 32
        self.l2_f = 64
        self.l3_f = 128

        with self.init_scope():
            self.c0 = IncompleteConvolution2D(self.l1_f, 3, pad=1, stride=2)
            self.bn0 = L.BatchNormalization(self.l1_f)
            self.d1 = IncompleteDepthwiseBlock(
                self.l1_f, self.l2_f, coeffs_generator, self.act, stride=1)
            self.d2 = IncompleteDepthwiseBlock(
                self.l2_f, self.l3_f, coeffs_generator, self.act, stride=2)
            self.d3 = IncompleteDepthwiseBlock(
                self.l3_f, self.l3_f, coeffs_generator, self.act, stride=1)
            self.d4 = IncompleteDepthwiseBlock(
                self.l3_f, self.l3_f, coeffs_generator, self.act, stride=1)
            self.d5 = IncompleteDepthwiseBlock(
                self.l3_f, self.l3_f, coeffs_generator, self.act, stride=1)
            self.d6 = IncompleteDepthwiseBlock(
                self.l3_f, self.l3_f, coeffs_generator, self.act, stride=1)
            self.l1 = IncompleteLinear(10)

    def __call__(self, x, t, comp_ratio=None, ret_param='loss'):
        def coeff_f(n):
            return util.zero_end(self.coeffs_generator(n), comp_ratio)

        h = self.act(self.bn0(self.c0(x)))
        h = self.d1(h, comp_ratio)
        h = self.d2(h, comp_ratio)
        h = self.d3(h, comp_ratio)
        h = self.d4(h, comp_ratio)
        h = self.d5(h, comp_ratio)
        h = self.d6(h, comp_ratio)
        # h = self.l1(h, coeff_f(np.prod(h.shape[1:])))
        h = self.l1(h)

        report = {
            'loss': softmax(h, t),
            'acc': F.accuracy(h, t)
        }

        reporter.report(report, self)
        return report[ret_param]

    def report_params(self):
        return ['validation/main/acc']

    def param_names(self):
        return 'approxdnet_{}_{}'.format(self.coeffs_generator.__name__, self.act.__name__)
