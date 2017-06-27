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

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=128,
                    help='learning minibatch size')
parser.add_argument('--out', '-o', default='result',
                    help='Directory to output the result')
parser.add_argument('--mode', '-m', default='harmonic_seq_group')
parser.add_argument('--comp_f', default='exp')
parser.add_argument('--filter_f', default='exp')

args = parser.parse_args()
train, test = get_mnist(ndim=3)

comp_ratios = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
acc_dict = {}
ratios_dict = {}

models = [
  net.BinaryNet(10),
  net.ApproxNetWW(10, m=1, comp_f='exp', filter_f='exp', act='ternary', comp_mode='harmonic_seq'),
]

for model in models:
    util.train_model(model, train, test, args)
    key = model.param_names()
    accs = []
    if key == 'approx':
        for cr in comp_ratios:
            acc = util.get_approx_acc(model, test, comp_ratio=cr, filter_ratio=0.5)
            accs.append(acc)
        ratios_dict[key] = comp_ratios*100.
    else:
        acc = util.get_approx_acc(model, test, comp_ratio=1.0, filter_ratio=0.0)
        accs.append(acc)
        ratios_dict[key] = [50]
    acc_dict[key] = accs
    
visualize.approx_acc(acc_dict, ratios_dict, prefix="act")