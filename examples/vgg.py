import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import visualize as vz
from net import VGG
import util
import idp.coeffs_generator as cg


def run(args):
    train, test = util.get_dataset(args.dataset)
    names = ['all-one (standard)', 'linear']
    colors = [vz.colors.all_one_lg, vz.colors.linear_lg]
    models = [
        VGG.VGG(10, cg.uniform, 'all'),
        VGG.VGG(10, cg.linear, 'slow_exp')
    ]
    comp_ratios = np.linspace(0.1, 1.0, 20)
    acc_dict = {}
    ratios_dict = {}
    for name, model in zip(names, models):
        util.load_or_train_model(model, train, test, args)
        acc_dict[name] = util.sweep_idp(model, test, comp_ratios, args)
        ratios_dict[name] = [100. * cr for cr in comp_ratios]

    filename = "VGG_{}".format(args.dataset)
    vz.plot(ratios_dict, acc_dict, names, filename, colors=colors,
            folder=args.figure_path, ext=args.ext,
            title='VGG (CIFAR-10)',
            xlabel='IDP (%)',
            ylabel='Classification Accuracy (%)', ylim=(90, 100))

if __name__ == '__main__':
    args = util.default_parser('VGG Example').parse_args()
    run(args)
