from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from approx.links_convolution_2d import IncompleteConvolution2D
import util

conv = IncompleteConvolution2D


class Block(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, coeff_generator, pad=1):
        super(Block, self).__init__()
        self.coeff_generator = coeff_generator
        self.in_channels = in_channels
        self.out_channels = out_channels
        with self.init_scope():
            self.conv = conv(out_channels, ksize, pad=pad, nobias=True)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x, comp_ratio=None):
        def coeff_f(n):
            return util.zero_end(self.coeff_generator(n), comp_ratio)

        h = self.conv(x, coeff_f(self.in_channels))
        h = self.bn(h)
        return F.relu(h)


class VGG(chainer.Chain):
    def __init__(self, class_labels, coeff_generator):
        super(VGG, self).__init__()
        self.coeff_generator = coeff_generator
        g = coeff_generator
        with self.init_scope():
            self.block1_1 = Block(3, 64, 3, g)
            self.block1_2 = Block(64, 64, 3, g)
            self.block2_1 = Block(64, 128, 3, g)
            self.block2_2 = Block(128, 128, 3, g)
            self.block3_1 = Block(128, 256, 3, g)
            self.block3_2 = Block(256, 256, 3, g)
            self.block3_3 = Block(256, 256, 3, g)
            self.block4_1 = Block(256, 512, 3, g)
            self.block4_2 = Block(512, 512, 3, g)
            self.block4_3 = Block(512, 512, 3, g)
            self.block5_1 = Block(512, 512, 3, g)
            self.block5_2 = Block(512, 512, 3, g)
            self.block5_3 = Block(512, 512, 3, g)
            self.fc1 = L.Linear(None, 512, nobias=True)
            self.bn_fc1 = L.BatchNormalization(512)
            self.fc2 = L.Linear(None, class_labels, nobias=True)

    def __call__(self, x, t, ret_param='loss', comp_ratio=None):
        # 64 channel blocks:
        h = self.block1_1(x, comp_ratio)
        h = F.dropout(h, ratio=0.3)
        h = self.block1_2(h, comp_ratio)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 128 channel blocks:
        h = self.block2_1(h, comp_ratio)
        h = F.dropout(h, ratio=0.4)
        h = self.block2_2(h, comp_ratio)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 256 channel blocks:
        h = self.block3_1(h, comp_ratio)
        h = F.dropout(h, ratio=0.4)
        h = self.block3_2(h, comp_ratio)
        h = F.dropout(h, ratio=0.4)
        h = self.block3_3(h, comp_ratio)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block4_1(h, comp_ratio)
        h = F.dropout(h, ratio=0.4)
        h = self.block4_2(h, comp_ratio)
        h = F.dropout(h, ratio=0.4)
        h = self.block4_3(h, comp_ratio)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block5_1(h, comp_ratio)
        h = F.dropout(h, ratio=0.4)
        h = self.block5_2(h, comp_ratio)
        h = F.dropout(h, ratio=0.4)
        h = self.block5_3(h, comp_ratio)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(h, ratio=0.5)
        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        h = self.fc2(h)

        report = {
            'loss': F.softmax_cross_entropy(h, t),
            'acc': F.accuracy(h, t)
        }

        reporter.report(report, self)
        return report[ret_param]

    def report_params(self):
        return ['validation/main/acc']

    def param_names(self):
        return 'VGG_{}'.format(self.coeff_generator.__name__)
