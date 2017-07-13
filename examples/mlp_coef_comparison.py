import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import visualize as vz
from idp.coeffs_generator import uniform_seq, harmonic_seq, linear_seq, exp_seq, uniform_exp_seq
from net import MLP
import util


def run(args):
    train, test = util.get_dataset(args.dataset)
    colors = [vz.colors.all_one_lg, vz.colors.linear_lg]
    names = ['all-one', 'harmonic', 'linear', 'exp', 'half_one']
    colors = [vz.colors.all_one_lg, vz.colors.harmonic_lg,
              vz.colors.linear_lg, vz.colors.exp_lg, vz.colors.half_one_lg]
    models = [
        MLP.MLP(10, uniform_seq),
        MLP.MLP(10, harmonic_seq),
        MLP.MLP(10, linear_seq),
        MLP.MLP(10, exp_seq),
        MLP.MLP(10, uniform_exp_seq),
    ]
    comp_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    acc_dict = {}
    ratios_dict = {}
    for name, model in zip(names, models):
        util.load_or_train_model(model, train, test, args)
        acc_dict[name] = util.sweep_idp(model, test, comp_ratios, args)
        ratios_dict[name] = [100. * cr for cr in comp_ratios]

    filename = "MLP_coef_comparison_{}".format(args.dataset)
    vz.plot(ratios_dict, acc_dict, names, filename, colors=colors,
            folder=args.figure_path, ext=args.ext,
            xlabel='Dot Product Component (%)',
            ylabel='Classification Accuracy (%)',
            legend_loc='lower right', ylim=(85, 100))


if __name__ == '__main__':
    args = util.default_parser('MLP Coef Functions Comparison').parse_args()
    run(args)
