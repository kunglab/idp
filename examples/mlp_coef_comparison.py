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
    colors = [vz.colors.all_one_lg, vz.colors.linear_lg]
    names = ['all-one', 'harmonic', 'linear', 'half_one']
    colors = [vz.colors.all_one_lg, vz.colors.harmonic_lg,
              vz.colors.linear_lg, vz.colors.half_one_lg]
    models = [
        MLP.MLP(10, cg.uniform),
        MLP.MLP(10, cg.harmonic),
        MLP.MLP(10, cg.linear),
        MLP.MLP(10, cg.uniform_exp),
    ]
    comp_ratios = np.linspace(0.1, 1.0, 20)
    acc_dict = {}
    ratios_dict = {}
    for name, model in zip(names, models):
        util.load_or_train_model(model, train, test, args)
        acc_dict[name] = util.sweep_idp(model, test, comp_ratios, args)
        ratios_dict[name] = [100. * cr for cr in comp_ratios]

    filename = "MLP_coef_comparison_{}".format(args.dataset)
    vz.plot(ratios_dict, acc_dict, names, filename, colors=colors,
            folder=args.figure_path, ext=args.ext,
            xlabel='IDP (%)',
            ylabel='Classification Accuracy (%)',
            title='MLP (MNIST)',
            legend_loc='lower right', ylim=(85, 100))


if __name__ == '__main__':
    args = util.default_parser('MLP Coef Functions Comparison').parse_args()
    run(args)
