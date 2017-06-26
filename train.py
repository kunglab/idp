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
args = parser.parse_args()
train, test = get_mnist(ndim=3)

comp_ratios = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
acc_dict = {}

acts = ['ternary', 'binary', 'relu']
m = 1
filter_f = 'exp'
comp_f = 'exp'
for act in acts:
    model = net.ApproxNetWW(10, m=m, comp_f=comp_f, filter_f=filter_f, act=act)
    util.train_model(model, train, test, args)
    key = "{}".format(act)
    accs = []
    for cr in comp_ratios:
        acc = util.get_approx_acc(model, test, comp_ratio=cr, filter_ratio=0.5)
        accs.append(acc)
    acc_dict[key] = accs
    
visualize.approx_acc(acc_dict, comp_ratios*100., prefix="act")
