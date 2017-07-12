import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import visualize as vz
from idp.coeffs_generator import uniform_seq, linear_seq, harmonic_seq
from net import MLP
import util


def run(args):
    train, test = util.get_dataset(args.dataset)
    names = ['all-one (standard)', 'linear']
    colors = [vz.colors.all_one_lg, vz.colors.linear_lg]
    models = [
        MLP.MLP(10, uniform_seq),
        MLP.MLP(10, linear_seq)
    ]
    comp_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    acc_dict = {}
    ratios_dict = {}
    for name, model in zip(names, models):
        util.load_or_train_model(model, train, test, args)
        acc_dict[name] = util.sweep_idp(model, test, comp_ratios, args)
        ratios_dict[name] = [100. * cr for cr in comp_ratios]

    filename = "MLP_{}".format(args.dataset)
    vz.plot(ratios_dict, acc_dict, names, filename, colors=colors,
            folder=args.figure_path, ext=args.ext,
            xlabel='Dot Product Component (%)',
            ylabel='Classification Accuracy (%)', ylim=(90, 100))


if __name__ == '__main__':
    args = util.default_parser('MLP Example').parse_args()
    run(args)
