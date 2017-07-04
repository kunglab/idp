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

parser = util.default_parser('Standard versus Incomplete Experiment')
args = parser.parse_args()

train, test = util.get_dataset(args.dataset)
nclass = np.bincount(test._datasets[1]).shape[0]
small_settings = util.get_net_settings(args.dataset, size='small')
large_settings = util.get_net_settings(args.dataset, size='large')
large_crs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
small_crs = [0.6, 0.7, 0.8, 0.9, 1.0]
names = ['incomplete-large', 'incomplete-small',
         'standard-large', 'standard-small']
colors = ['#d73027', '#f46d43', '#1a9850', '#66bd63']
models = [
    net.ApproxNet(nclass, *large_settings, m=1, comp_f='id',
                  act='ternary', coeffs_generator=harmonic_seq),
    net.ApproxNet(nclass, *small_settings, m=1, comp_f='id',
                  act='ternary', coeffs_generator=harmonic_seq),
    net.ApproxNet(nclass, *large_settings, m=1, comp_f='id',
                  act='ternary', coeffs_generator=uniform_seq),
    net.ApproxNet(nclass, *small_settings, m=1, comp_f='id',
                  act='ternary', coeffs_generator=uniform_seq),
]
acc_dict = {}
ratios_dict = {}
for name, model in zip(names, models):
    acc_dict[name] = []
    ratios_dict[name] = []
    util.load_or_train_model(model, train, test, args)
    if 'small' in name:
        comp_ratios = small_crs
    else:
        comp_ratios = large_crs
    for cr in comp_ratios:
        acc = util.get_approx_acc(model, test, comp_ratio=cr)
        acc_dict[name].append(acc)
        ratios_dict[name].append(100. * cr)

    if 'small' in name:
        ratios_dict[name] = [0.5 * cr for cr in ratios_dict[name]]

filename = "versus_standard_{}".format(args.dataset)
visualize.plot(ratios_dict, acc_dict, names, filename, colors=colors, folder=args.figure_path, ext=args.ext,
               xlabel='Dot Product Component (%)', ylabel='Classification Accuracy (%)')

filename = "versus_standard_{}_zoom".format(args.dataset)
visualize.plot(ratios_dict, acc_dict, names, filename, colors=colors, folder=args.figure_path, ext=args.ext,
               xlabel='Dot Product Component (%)', ylabel='Classification Accuracy (%)',
               ylim=(90, 100))
