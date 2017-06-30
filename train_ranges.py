import argparse

import visualize  # matplotlib is being imported somewhere else..

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
parser.add_argument('--dataset', '-d', default='mnist')

args = parser.parse_args()
train, test = util.get_dataset(args.dataset)
l1_f, l2_f, l3_f = util.get_net_settings(args.dataset)

comp_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
acc_dict = {}
ratios_dict = {}

names = ['approx_range', 'approx_point']
models = [
    net.ApproxNetWWV2(10, l1_f, l2_f, l3_f, m=1, comp_f='mid_exp', filter_f='id',
                      act='ternary', comp_mode='harmonic_seq_group'),
    net.ApproxNetWWV2(10, l1_f, l2_f, l3_f, m=1, comp_f='50', filter_f='id',
                      act='ternary', comp_mode='harmonic_seq_group')
]

for name, model in zip(names, models):
    fr = 0.0
    acc_dict[name] = []
    ratios_dict[name] = []
    util.train_model(model, train, test, args)
    for cr in comp_ratios:
        acc = util.get_approx_acc(model, test, comp_ratio=cr, filter_ratio=fr)
        acc_dict[name].append(acc)
        ratios_dict[name].append(100. * (1 - fr) * cr)

visualize.approx_acc(acc_dict, ratios_dict, names,
                     prefix="approx_point_{}".format(args.dataset))
