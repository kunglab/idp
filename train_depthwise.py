import argparse

import visualize  # matplotlib is being imported somewhere else..

import cupy
import chainer
from chainer.datasets import get_mnist, get_cifar10
import chainer.functions as F
import numpy as np

from binary.bst import bst
from binary.ww_bconv_v3 import uniform_seq, harmonic_seq, linear_seq, exp_seq, uniform_exp_seq
import net
import util

parser = util.default_parser('')
args = parser.parse_args()

train, test = util.get_dataset(args.dataset)
l1_f, l2_f, l3_f = util.get_net_settings(args.dataset)

colors = ['#377eb8', '#d73027']
comp_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
acc_dict = {}
ratios_dict = {}
names = ['all-one', 'harmonic']
models = [
    net.ApproxDNet(uniform_seq),
    net.ApproxDNet(harmonic_seq),
]
for name, model in zip(names, models):
    util.train_model(model, train, test, args)
    acc_dict[name] = []
    ratios_dict[name] = []
    for cr in comp_ratios:
        acc = util.get_approx_acc(model, test, comp_ratio=cr)
        print(cr, acc)
        acc_dict[name].append(acc)
        ratios_dict[name].append(100. * cr)

filename = "versus_depthwise_{}".format(args.dataset)
visualize.plot(ratios_dict, acc_dict, names, filename, colors=colors, folder=args.figure_path, ext=args.ext,
               xlabel='Dot Product Component (%)', ylabel='Classification Accuracy (%)',
               ylim=(90, 100))