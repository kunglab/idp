import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import visualize as vz
import util
import idp.coeffs_generator as cg


args = util.default_parser('Generate Weight Functions').parse_args()
funcs = [cg.uniform, cg.harmonic, cg.linear, cg.uniform_exp]
names = ['all-one', 'harmonic', 'linear', 'half_one']
colors = [vz.colors.all_one_lg, vz.colors.harmonic_lg,
          vz.colors.linear_lg, vz.colors.half_one_lg]
xs = {}
ys = {}
n = 16
for name, func in zip(names, funcs):
    xs[name] = np.arange(n) + 1
    ys[name] = func(n)

vz.plot(xs, ys, names, 'coef_functions', colors, marker=None,
        xlabel=r'Channel Coefficient Index (for $\gamma_{1}$ to $\gamma_{16}$)',
        ylabel='Channel Coefficient', title='Channel Coefficient Functions',
        xticks=range(1, 17, 2), legend_loc='upper right',
        ext=args.ext, folder=args.figure_path)
