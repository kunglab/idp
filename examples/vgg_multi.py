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
    names = ['linear']
    colors = [vz.colors.linear_sm, vz.colors.linear_lg]
    models = [
        VGG.VGG(10, cg.linear, profiles=[(0, 2), (2, 10)])
    ]
    comp_ratios = np.linspace(0.1, 1, 20)
    acc_dict = {}
    ratios_dict = {}
    key_names = []
    for name, model in zip(names, models):
        util.train_model_profiles(model, train, test, args)
        for profile in range(len(model.profiles)):
            key = name + '-' + str(profile + 1)
            key_names.append(key)
            acc_dict[key] = util.sweep_idp(
                model, test, comp_ratios, args, profile=profile)
            ratios_dict[key] = [100. * cr for cr in comp_ratios]

    filename = "VGG_{}_multi".format(args.dataset)
    vz.plot(ratios_dict, acc_dict, key_names, filename, colors=colors,
            folder=args.figure_path, ext=args.ext,
            title='VGG-16 (CIFAR-10)',
            xlabel='IDP (%)',
            ylabel='Classification Accuracy (%)')


if __name__ == '__main__':
    args = util.default_parser('VGG Example').parse_args()
    run(args)
