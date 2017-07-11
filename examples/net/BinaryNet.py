import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter

from approx.binary.link
from binary.ww_bconv_v3 import uniform_seq
import util

class Block(chainer.Chain):
    def __init__(self, num_fs, ksize=3, pksize=2):
        super(Block, self).__init__()
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