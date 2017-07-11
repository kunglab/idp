import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import visualize
from idp.coeffs_generator import uniform_seq, linear_seq, harmonic_seq
from net import MLP
import util


def run(args):
    train, test = util.get_dataset(args.dataset)

    colors = ['red', 'green', 'blue']
    comp_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    names = ['all-one', 'linear', 'harmonic']
    models = [
        MLP.MLP(100, 10, uniform_seq),
        MLP.MLP(100, 10, linear_seq),
        MLP.MLP(100, 10, harmonic_seq),
    ]
    acc_dict = {}
    ratios_dict = {}
    for name, model in zip(names, models):
        util.load_or_train_model(model, train, test, args)
        acc_dict[name] = []
        ratios_dict[name] = []
        for cr in comp_ratios:
            acc = util.get_approx_acc(model, test, comp_ratio=cr)
            print(cr, acc)
            acc_dict[name].append(acc)
            ratios_dict[name].append(100. * cr)

    filename = "MLP_{}".format(args.dataset)
    visualize.plot(ratios_dict, acc_dict, names, filename, colors=colors,
                   folder=args.figure_path, ext=args.ext,
                   xlabel='Dot Product Component (%)',
                   ylabel='Classification Accuracy (%)', ylim=(90, 100))


if __name__ == '__main__':
    args = util.default_parser('MLP Example').parse_args()
    run(args)
