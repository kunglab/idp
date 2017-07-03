import visualize

import cupy
import chainer
from chainer.datasets import get_mnist, get_cifar10
import chainer.functions as F
import numpy as np

from binary.bst import bst
import net
import util
from binary.ww_bconv_v3 import uniform_seq, harmonic_seq, linear_seq, exp_seq, uniform_exp_seq

parser = util.default_parser('Weight Coefficents Experiment')
args = parser.parse_args()

train, test = util.get_dataset(args.dataset)
nclass = np.bincount(test._datasets[1]).shape[0]
large_settings = util.get_net_settings(args.dataset, size='large')
comp_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
acc_dict = {}
ratios_dict = {}
names = ['uniform', 'harmonic', 'linear', 'exp', 'half_one']
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
models = [
    net.ApproxNet(nclass, *large_settings, m=1, comp_f='id',
                  act='ternary', coeffs_generator=uniform_seq),
    net.ApproxNet(nclass, *large_settings, m=1, comp_f='id',
                  act='ternary', coeffs_generator=harmonic_seq),
    net.ApproxNet(nclass, *large_settings, m=1, comp_f='id',
                  act='ternary', coeffs_generator=linear_seq),
    net.ApproxNet(nclass, *large_settings, m=1, comp_f='id',
                  act='ternary', coeffs_generator=exp_seq),
    net.ApproxNet(nclass, *large_settings, m=1, comp_f='id',
                  act='ternary', coeffs_generator=uniform_exp_seq),
]
for name, model in zip(names, models):
    acc_dict[name] = []
    ratios_dict[name] = []
    util.load_or_train_model(model, train, test, args)
    for cr in comp_ratios:
        acc = util.get_approx_acc(model, test, comp_ratio=cr)
        acc_dict[name].append(acc)
        ratios_dict[name].append(100. * cr)

filename = "coef_comparison_{}".format(args.dataset)
visualize.plot(ratios_dict, acc_dict, names, filename, colors=colors,
               xlabel='Dot Product Component (%)', ylabel='Classification Accuracy (%)')

filename = "coef_comparison_{}_zoom".format(args.dataset)
visualize.plot(ratios_dict, acc_dict, names, filename, colors=colors,
               xlabel='Dot Product Component (%)', ylabel='Classification Accuracy (%)',
               ylim=(90,100))