import argparse

import visualize # matplotlib is being imported somewhere else..

import cupy
from chainer.datasets import get_mnist, get_cifar10
import chainer.functions as F
import numpy as np

from binary.bst import bst
import net
import util

def compute_approx(model, dataset, do_type):
    h_full = util.get_approx(model, dataset, 0, do_type)
    h_full = bst(h_full).data
    hs, ratios = [], []
    for ratio in np.linspace(0, 0.95, 50):
        h = util.get_approx(model, test, ratio, do_type)
        h = bst(h).data
        alike = util.pct_alike(h, h_full)
        hs.append(alike*100.)
        ratios.append((1 - ratio)*100.)
    return hs, ratios

def compute_layer_01(model, dataset, layer_idx):
    layer = util.get_layer(model, dataset, layer=layer_idx)
    layer.to_cpu()
    layer_sum = np.sum(layer.data > 0, axis=(1,2,3))
    layer_sum = layer_sum / np.prod(layer.shape[1:])
    layer_sum.sort()
    return layer_sum

def compute_filter_match(model, dataset, filter, do_type, ratios):
    l1 = util.get_layer(model, dataset, layer=1)
    hs = []
    for ratio in ratios:
        subsamp_filter = net.sample_filter(filter, ratio=ratio, do_type=do_type)
        h = F.convolution_2d(l1, subsamp_filter, pad=1)
        h.to_cpu()
        hs.append(h[0].data)
    return np.array(hs)

def compute_approx(model, dataset):
    do_type = 'random'
    accs = []
    ratios = np.linspace(0, 0.25, 5)
    for ratio in ratios:
        acc = util.get_approx_acc(model, test, do_type=do_type, ratio=ratio)
        accs.append(acc)
    return accs, (1-ratios)*100.

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=128,
                    help='learning minibatch size')
parser.add_argument('--out', '-o', default='result',
                    help='Directory to output the result')
args = parser.parse_args()
train, test = get_mnist(ndim=3)
# ms = [0.5, 1, 2, 3, 4]
ms = [1]
acc_dict = {}
for m in ms:
    key = str(m)
    model = net.ApproxNet(10, m=m)
    util.train_model(model, train, test, args)
    accs, ratios = compute_approx(model, test)
    acc_dict[key] = accs
visualize.approx_acc(acc_dict, ratios)
assert False

ones_filter = cupy.ones((1, 16, 3, 3))

hs = compute_filter_match(model, test, ones_filter, 'random', ratios)
visualize.conv_approx(hs, ratios, 'ones')
fs = bst(model.l2.W).data

hs = compute_filter_match(model, test, cupy.expand_dims(fs[0], 0), 'random', ratios)
visualize.conv_approx(hs, ratios, 'learned')
hs = bst(hs).data
visualize.conv_approx(hs, ratios, 'learned_binary')


# l1 = compute_layer_01(model, test, layer=1)
# l2 = compute_layer_01(model, test, layer=2)
# hs, ratios = compute_approx(model, test, do_type)
# visualize.layer_01s(l1, l2)
# visualize.approx_match(hs, ratios, do_type)