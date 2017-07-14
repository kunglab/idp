from __future__ import absolute_import

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import visualize as vz
import idp.coeffs_generator as cg
from net import BinaryNet
import util


def run(args):
    train, test = util.get_dataset(args.dataset)
    names = ['all-ones,exp', 'all-ones,all', 'linear,exp', 'linear,all']
    colors = [vz.colors.all_one_lg, vz.colors.all_one_sm,
              vz.colors.linear_lg, vz.colors.linear_sm]
    models = [
        BinaryNet.BinaryConvNet(10, cg.uniform, 'slow_exp'),
        BinaryNet.BinaryConvNet(10, cg.uniform, 'all'),
        BinaryNet.BinaryConvNet(10, cg.linear, 'slow_exp'),
        BinaryNet.BinaryConvNet(10, cg.linear, 'all'),
    ]
    comp_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    acc_dict = {}
    ratios_dict = {}
    for name, model in zip(names, models):
        util.load_or_train_model(model, train, test, args)
        acc_dict[name] = util.sweep_idp(model, test, comp_ratios, args)
        ratios_dict[name] = [100. * cr for cr in comp_ratios]

    filename = "BinaryNet_{}".format(args.dataset)
    vz.plot(ratios_dict, acc_dict, names, filename, colors=colors,
            folder=args.figure_path, ext=args.ext,
            xlabel='Dot Product Component (%)',
            ylabel='Classification Accuracy (%)', ylim=(85, 100))


if __name__ == '__main__':
    args = util.default_parser('BinaryNet Example').parse_args()
    run(args)
