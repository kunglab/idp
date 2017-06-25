import argparse
from itertools import product

import visualize # matplotlib is being imported somewhere else..

import cupy
import chainer
from chainer.datasets import get_mnist, get_cifar10
import chainer.functions as F
import numpy as np

from binary.bst import bst
import net
import util

def compute_alike(model, dataset, do_type):
    h_full = util.get_approx_features(model, dataset, 0, do_type)
    h_full = h_full.astype(int)
    hs, ratios = [], []
    for ratio in np.linspace(0, 0.95, 20):
        h = util.get_approx_features(model, test, ratio, do_type)
        h = h.astype(int)
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

ms = [1.0]
acc_dict, alike_dict = {}, {}
do_types = ['random', 'equal']
tr_ratios = [0.5]
for tr_ratio, do_type, m in product(tr_ratios, do_types, ms):
    key = "{}_{}_{}".format(tr_ratio, do_type, m)
    model = net.ApproxNetSS(10, m, ratio=tr_ratio)
    chainer.config.train = True
    util.train_model(model, train, test, args)
    chainer.config.train = False
    alike_pcts, alike_ratios = compute_alike(model, test, do_type=do_type)
    alike_dict[key] = alike_pcts
    hs = [util.get_approx_features(model, test, 1-h, do_type=do_type) for h in [0, 0.125, 0.25, 0.5]]
    visualize.conv_approx(hs, [0, 0.125, 0.25, 0.5], key)
    ratios = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    accs = [util.get_approx_acc(model, test, do_type=do_type, ratio=r) for r in ratios]
    acc_dict[key] = accs
        
visualize.approx_acc(acc_dict, ratios, prefix="SS_")
visualize.approx_match(alike_dict, alike_ratios, prefix="SS_")