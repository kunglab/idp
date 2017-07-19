import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import visualize as vz
import idp.coeffs_generator as cg
from net import MLP
import util


def run(args):
    train, test = util.get_dataset(args.dataset)
    # names = ['all-ones,exp', 'all-ones,all', 'linear,exp', 'linear,all']
    names = ['linear']
    colors = [vz.colors.linear_sm, vz.colors.linear_lg]
    models = [
        MLP.MLP(10, cg.linear, [(0, 3), (3, 10)], n_units=100),
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

    filename = "MLP_{}_multi".format(args.dataset)
    vz.plot(ratios_dict, acc_dict, key_names, filename, colors=colors,
            folder=args.figure_path, ext=args.ext,
            xlabel='IDP (%)',
            ylabel='Classification Accuracy (%)',
            title='MLP (MNIST)', ylim=(85, 100))


if __name__ == '__main__':
    args = util.default_parser('MLP Example').parse_args()
    run(args)
