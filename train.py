import argparse

import visualize # matplotlib is being imported somewhere else..

import cupy
import chainer
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

comp_ratios = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
acc_dict = {}

acts = ['ternary', 'binary', 'relu']
m = 1
f_ratio = 0.5
c_ratio = 0.5
for act in acts:
    model = net.ApproxNetWW(10, m=m, comp_ratio=c_ratio, filter_ratio=f_ratio, act=act)
    util.train_model(model, train, test, args)
    key = "{}".format(act)
    accs = []
    for cr in comp_ratios:
        acc = util.get_approx_acc(model, test, comp_ratio=cr, filter_ratio=f_ratio)
        accs.append(acc)
    acc_dict[key] = accs
    
visualize.approx_acc(acc_dict, comp_ratios*100., prefix="act")
