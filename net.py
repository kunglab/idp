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
from binary.ww_bconv_v2 import WWBinaryConvolution2DV2
from binary.function_binary_convolution_2d import binary_convolution_2d
from binary.bst import bst, mbst, mbst_bp
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
                 act='ternary', comp_mode='harmonic_seq_group'):
        super(ApproxBlock, self).__init__()
        self.comp_f = comp_f
        self.comp_mode = comp_mode
        self.m = m
        self.pksize = pksize

        if isinstance(num_fs, (int),):
            l1_f = l2_f = num_fs
        else:
            l1_f, l2_f = num_fs[0], num_fs[1]

        if act == 'ternary':
            self.act = partial(mbst_bp, m=self.m)
        elif act == 'binary':
            self.act = bst
        elif act == 'relu':
            self.act = F.relu
        else:
            raise NameError("act={}".format(act))

        with self.init_scope():
            self.l1 = WWBinaryConvolution2DV2(
                l1_f, ksize, pad=1, mode=self.comp_mode)
            self.bn1 = L.BatchNormalization(l1_f)
            self.l2 = WWBinaryConvolution2DV2(
                l2_f, ksize, pad=1, mode=self.comp_mode)
            self.bn2 = L.BatchNormalization(l2_f)

    def __call__(self, x, comp_ratio=None, ret_param='loss'):
        if not comp_ratio:
            comp_ratio = 1 - util.gen_prob(self.comp_f)

        h = self.l1(x, ratio=comp_ratio)
        h = self.act(self.bn1(h))
        h = self.l2(h, ratio=comp_ratio)
        h = F.max_pooling_2d(h, self.pksize, stride=1)
        h = self.act(self.bn2(h))
        return h


class ApproxNet(chainer.Chain):
    def __init__(self, n_out, l1_f, l2_f=None, l3_f=None, m=0, comp_f='exp',
                 act='ternary', comp_mode='harmonic_seq_group'):
        super(ApproxNet, self).__init__()
        self.n_out = n_out
        self.comp_f = comp_f
        self.comp_mode = comp_mode
        self.l1_f = l1_f
        self.l2_f = l2_f
        self.l3_f = l3_f

        if not l2_f and l3_f:
            raise ValueError("l2_f must be set if l3_f is set.")

        with self.init_scope():
            self.l1 = ApproxBlock(l1_f, m=m, comp_f=comp_f,
                                  act=act, comp_mode=comp_mode)
            if l2_f:
                self.l2 = BinaryBlock(l2_f)
            if l3_f:
                self.l3 = BinaryBlock(l3_f)
            self.l4 = BinaryLinear(n_out)

    def __call__(self, x, t, comp_ratio=None, filter_ratio=None, ret_param='loss'):
        h = self.l1(x, comp_ratio, filter_ratio)
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
        return 'approx'


class BinaryNet(chainer.Chain):
    def __init__(self, n_out, l1_f, l2_f=None, l3_f=None):
        super(BinaryNet, self).__init__()
        self.l1_f = l1_f
        self.l2_f = l2_f
        self.l3_f = l3_f

        if not l2_f and l3_f:
            raise ValueError("l2_f must be set if l3_f is set.")

        with self.init_scope():
            self.l1 = ApproxBlock(l1_f, m=1, comp_f='id', filter_f='id',
                                  act='ternary', comp_mode='harmonic_seq')
            if l2_f:
                self.l2 = BinaryBlock(l2_f)
            if l3_f:
                self.l3 = BinaryBlock(l3_f)
            self.l4 = BinaryLinear(n_out)

    def __call__(self, x, t, comp_ratio=None, filter_ratio=None, ret_param='loss'):
        h = self.l1(x, comp_ratio, filter_ratio)
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
        return 'standard'
