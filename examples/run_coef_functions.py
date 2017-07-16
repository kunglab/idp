import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import visualize as vz
import util
import idp.coeffs_generator as cg


args = util.default_parser('Generate Weight Functions').parse_args()
# funcs = [cg.uniform, cg.harmonic, cg.linear, cg.exp, cg.uniform_exp]
# names = ['all-one', 'harmonic', 'linear', 'exp', 'half_one']
# colors = [vz.colors.all_one_lg, vz.colors.harmonic_lg,
        #   vz.colors.linear_lg, vz.colors.exp_lg, vz.colors.half_one_lg]
funcs = [cg.uniform, cg.four_steps, cg.mag_steps, cg.exp, cg.uniform_exp]
names = ['all-one', 'four steps', 'magnitude steps']
colors = [vz.colors.all_one_lg, vz.colors.linear_sm, vz.colors.linear_lg]
xs = {}
ys = {}
n = 16
for name, func in zip(names, funcs):
    xs[name] = np.arange(n) + 1
    ys[name] = func(n)

vz.plot(xs, ys, names, 'weight_functions', colors, marker=None,
        xlabel=r'Weight Coefficient Index (for $c_{1}$ to $c_{16}$)',
        ylabel='Weight Coefficient', title='Weight Coefficient Functions',
        xticks=range(1,17,2), legend_loc='upper right',
        ext=args.ext, folder=args.figure_path)
