import argparse

import visualize  # matplotlib is being imported somewhere else..

import cupy
import chainer
from chainer.datasets import get_mnist, get_cifar10
import chainer.functions as F
import numpy as np

from binary.bst import bst
import net_v3 as net
import util
from binary.ww_bconv_v3 import uniform_seq, harmonic_seq, linear_seq, exp_seq, uniform_exp_seq

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
#l1_f, l2_f, l3_f = 16,16,16

comp_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
acc_dict = {}
ratios_dict = {}

names = ['100', '50', '10']
models = [
    net.ApproxNet(10, l1_f, l2_f, l3_f, m=1, comp_f='0',
                  act='ternary', coeffs_generator=uniform_exp_seq),
    net.ApproxNet(10, l1_f, l2_f, l3_f, m=1, comp_f='50',
                  act='ternary', coeffs_generator=uniform_exp_seq),
    net.ApproxNet(10, l1_f, l2_f, l3_f, m=1, comp_f='90',
                  act='ternary', coeffs_generator=uniform_exp_seq),
]

for name, model in zip(names, models):
    acc_dict[name] = []
    ratios_dict[name] = []
    util.train_model(model, train, test, args)
    for cr in comp_ratios:
        acc = util.get_approx_acc(model, test, comp_ratio=cr)
        acc_dict[name].append(acc)
        ratios_dict[name].append(100. * cr)

visualize.approx_acc(acc_dict, ratios_dict, names,
                     prefix="approx_point_{}".format(args.dataset))
