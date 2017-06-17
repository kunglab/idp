#!/usr/bin/env python

import argparse

import chainer
from chainer import reporter
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.functions import softmax_cross_entropy as softmax

from binary.blinear import BinaryLinear
from binary.bconv import BinaryConvolution2D
from binary.bst import bst

class BinConvNet(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(BinConvNet, self).__init__()
        self.n_out = n_out
        with self.init_scope():
            self.l1 = BinaryConvolution2D(n_units, 3, pad=1)
            self.bn1 = L.BatchNormalization(n_units)
            self.l2 = BinaryLinear(n_out)

    def __call__(self, x, t, ret_param='loss'):
        h = bst(self.bn1(self.l1(x)))
        h = self.l2(h)

        report = {
            'loss': softmax(h, t),
            'acc': F.accuracy(h, t)
        }

        reporter.report(report, self)
        return report[ret_param]

    def report_params(self):
        return ['validation/main/acc']