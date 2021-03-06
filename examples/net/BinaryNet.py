import os
import sys
from functools import partial
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter

from idp.binary.links_convolution_2d import IncompleteBinaryConvolution2D
from idp.binary.links_linear import IncompleteBinaryLinear
import idp.coeffs_generator as cg
from idp.blackout_generator import gen_blackout
from idp.binary.bst import mbst_bp, bst, sbst
import util


class Block(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, coeff_generator,
                 bo_generator, act, input_layer=False, pksize=2):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.coeff_generator = coeff_generator
        self.bo_generator = bo_generator
        self.act = act
        self.input_layer = input_layer
        self.pksize = pksize

        if act == 'ternary':
            self.act = partial(mbst_bp, m=1)
        elif act == 'binary':
            self.act = bst
        elif act == 'relu':
            self.act = F.relu
        else:
            raise NameError("act={}".format(act))

        with self.init_scope():
            self.l1 = IncompleteBinaryConvolution2D(out_channels, ksize, pad=1)
            self.bn1 = L.BatchNormalization(out_channels)
            self.l2 = IncompleteBinaryConvolution2D(out_channels, ksize, pad=1)
            self.bn2 = L.BatchNormalization(out_channels)

    def __call__(self, x, comp_ratio=None):
        if comp_ratio == None:
            comp_ratio = 1 - gen_blackout(self.bo_generator)

        def coeff_f(n):
            return util.zero_end(self.coeff_generator(n), comp_ratio)

        if self.input_layer:
            h = self.l1(x)
        else:
            h = self.l1(x, coeff_f(self.in_channels))
        h = self.act(self.bn1(h))
        h = self.l2(h, coeff_f(self.out_channels))
        h = F.max_pooling_2d(h, self.pksize, stride=1)
        h = self.act(self.bn2(h))
        return h


class BinaryConvNet(chainer.Chain):
    def __init__(self, class_labels, coeff_generator, bo_generator, act='binary'):
        super(BinaryConvNet, self).__init__()
        self.coeff_generator = coeff_generator
        self.bo_generator = bo_generator
        self.act = act
        with self.init_scope():
            self.l1 = Block(1, 8, 3, coeff_generator, bo_generator,
                            act, input_layer=True)
            self.l2 = Block(8, 32, 3, cg.uniform, bo_generator, act)
            self.l3 = Block(32, 64, 3, cg.uniform, bo_generator, act)
            self.l4 = IncompleteBinaryLinear(None, class_labels)

    def __call__(self, x, t, ret_param='loss', comp_ratio=None):
        # device
        h = self.l1(x, comp_ratio)

        # cloud
        h = self.l2(h, 1.0)
        h = self.l3(h, 1.0)
        h = self.l4(h, 1.0)

        report = {
            'loss': F.softmax_cross_entropy(h, t),
            'acc': F.accuracy(h, t)
        }

        reporter.report(report, self)
        return report[ret_param]

    def report_params(self):
        return ['validation/main/acc']

    def param_names(self):
        return 'BinaryNet_{}_{}'.format(self.coeff_generator.__name__, self.bo_generator)
