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
    names = ['uniform']
    colors = [vz.colors.all_one_lg]
    models = [
        # MLP.MLP(10, two_step, 'all'),
        MLP.MLP(10, cg.uniform),
        # MLP.MLP(10, cg.linear, 'all'),
        # MLP.MLP(10, two_steps, 'all'),
        # MLP.MLP(10, cg.three_steps),
        # MLP.MLP(10, cg.uniform, 'slow_exp'),
        # MLP.MLP(10, cg.linear, 'slow_exp')
    ]
    comp_ratios = np.linspace(0.1, 1, 20)
    acc_dict = {}
    ratios_dict = {}
    for name, model in zip(names, models):
        util.train_model_profiles(model, train, test, args)
        for profile in model.profiles():
            key = name + '_' + str(profile)
            acc_dict[key] = util.sweep_idp(
                model, test, comp_ratios, args, profile=profile)
        ratios_dict[key] = [100. * cr for cr in comp_ratios]

    filename = "MLP_{}".format(args.dataset)
    vz.plot(ratios_dict, acc_dict, names, filename, colors=colors,
            folder=args.figure_path, ext=args.ext,
            xlabel='Dot Product Component (%)',
            ylabel='Classification Accuracy (%)',
            title='MLP (MNIST)',
            ylim=(90, 100))


if __name__ == '__main__':
    args = util.default_parser('MLP Example').parse_args()
    run(args)
