import os
from collections import namedtuple

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
SMALL_SIZE = 14
MEDIUM_SIZE = 17
BIGGER_SIZE = 19

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
         marker='o', xlabel=None, ylabel=None, xlim=None, ylim=None, xticks=None,
         yticks=None, title=None, linewidth=3.5, ms=5, legend_loc=0):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, key in enumerate(keys):
        if i == 0:
            plt.plot(xd[key], yd[key], label=key, ms=ms,
                     marker=marker, color=colors[i], linewidth=linewidth)
        else:
            plt.plot(xd[key], yd[key], label=key, ms=ms, marker=marker,
                     color=colors[i], linestyle='--', dashes=dash_list[i], linewidth=linewidth)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if xticks:
        plt.xticks(xticks)
    if yticks:
        plt.yticks(yticks)
    if title:
        plt.title(title)
    plt.legend(loc=legend_loc, handlelength=3)
    plt.tight_layout()
    plt.grid()
    full_file = '{}{}.{}'.format(folder, filename, ext)
    plt.savefig(full_file, dpi=300)
    plt.clf()
