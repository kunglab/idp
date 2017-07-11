import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter

from idp.links_convolution_2d import IncompleteConvolution2D
from idp.links_depthwise_convolution_2d import IncompleteDepthwiseConvolution2D
from idp.links_linear import IncompleteLinear
import util


class Block(chainer.Chain):
    def __init__(self, in_chan, num_f, coeffs_generator, act, stride=1):
        super(Block, self).__init__()
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


class MobileNet(chainer.Chain):
    def __init__(self, coeffs_generator):
        super(MobileNet, self).__init__()
        self.coeffs_generator = coeffs_generator
        self.l1_f = 32
        self.l2_f = 64
        self.l3_f = 128
        self.l4_f = 128

        with self.init_scope():
            self.c0 = IncompleteConvolution2D(self.l1_f, 3, pad=1, stride=1)
            self.bn0 = L.BatchNormalization(self.l1_f)
            self.d1 = Block(self.l1_f, self.l2_f,
                            coeffs_generator, F.relu, stride=1)
            self.d2 = Block(
                self.l2_f, self.l3_f, coeffs_generator, F.relu, stride=2)
            self.d3 = Block(
                self.l3_f, self.l3_f, coeffs_generator, F.relu, stride=1)
            self.d4 = Block(
                self.l3_f, self.l3_f, coeffs_generator, F.relu, stride=2)
            self.d5 = Block(
                self.l3_f, self.l4_f, coeffs_generator, F.relu, stride=1)
            self.d6 = Block(
                self.l4_f, self.l4_f, coeffs_generator, F.relu, stride=1)
            self.l1 = IncompleteLinear(10)

    def __call__(self, x, t, ret_param='loss', comp_ratio=None):
        def coeff_f(n):
            return util.zero_end(self.coeffs_generator(n), comp_ratio)

        h = F.relu(self.bn0(self.c0(x)))
        h = self.d1(h, comp_ratio)
        h = self.d2(h, comp_ratio)
        # h = self.d3(h, comp_ratio)
        # h = self.d4(h, comp_ratio)
        # h = self.d5(h, comp_ratio)
        # h = self.d6(h, comp_ratio)
        # h = self.l1(h, coeff_f(np.prod(h.shape[1:])))
        h = self.l1(h)

        report = {
            'loss': F.softmax_cross_entropy(h, t),
            'acc': F.accuracy(h, t)
        }

        reporter.report(report, self)
        return report[ret_param]

    def report_params(self):
        return ['validation/main/acc']

    def param_names(self):
        return 'MobileNet_{}'.format(self.coeffs_generator.__name__)
