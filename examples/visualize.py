import os
from collections import namedtuple

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

dash_list = [(1e10, 1), (5, 2), (2, 5), (4, 10), (3, 3, 2, 2), (5, 2, 20, 2)]
ModelTypes = namedtuple('ModelTypes', ['all_one_lg', 'all_one_sm', 'linear_lg', 'linear_sm',
                                       'harmonic_lg', 'harmonic_sm', 'half_one_lg', 'half_one_sm',
                                       'exp_lg', 'exp_sm'])

colors = ModelTypes(
    '#a11214',
    '#e41a1c',
    '#377e35',
    '#4daf4a',
    '#235176',
    '#377eb8',
    '#b35900',
    '#ff7f00',
    '#5d3064',
    '#984ea3',
)


def plot(xd, yd, keys, filename, colors, folder='figures/', ext='png',
         marker='o', xlabel=None, ylabel=None, xlim=None, ylim=None, title=None, linewidth=2):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, key in enumerate(keys):
        if i == 0:
            plt.plot(xd[key], yd[key], label=key, ms=4,
                     marker=marker, color=colors[i], linewidth=linewidth)
        else:
            plt.plot(xd[key], yd[key], label=key, ms=4, marker=marker,
                     color=colors[i], linestyle='--', dashes=dash_list[i], linewidth=linewidth)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if title:
        plt.title(title)
    plt.legend(loc=0, handlelength=3)
    plt.tight_layout()
    plt.grid()
    full_file = '{}{}.{}'.format(folder, filename, ext)
    print(full_file)
    plt.savefig(full_file, dpi=300)
    plt.clf()
