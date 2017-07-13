import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import visualize as vz
import util
from idp.coeffs_generator import uniform_seq, harmonic_seq, linear_seq, exp_seq, uniform_exp_seq


args = util.default_parser('Generate Weight Functions').parse_args()
funcs = [uniform_seq, harmonic_seq, linear_seq, exp_seq, uniform_exp_seq]
names = ['all-one', 'harmonic', 'linear', 'exp', 'half_one']
colors = [vz.colors.all_one_lg, vz.colors.harmonic_lg,
          vz.colors.linear_lg, vz.colors.exp_lg, vz.colors.half_one_lg]
xs = {}
ys = {}
n = 16
for name, func in zip(names, funcs):
    xs[name] = np.arange(n) + 1
    ys[name] = func(n)

vz.plot(xs, ys, names, 'weight_functions', colors, marker=None,
        xlabel=r'Weight Coefficient Index ($c_{1}$ to $c_{16}$)',
        ylabel='Weight Coefficient', title='Weight Coefficient Functions',
        xticks=range(1,17,2), legend_loc='upper right',
        ext=args.ext, folder=args.figure_path)
