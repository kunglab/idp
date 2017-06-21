import pathlib

import cupy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import numpy as np

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def approx_acc(accs, ratios, folder='figures/', ext='.png'):
    #pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 
    try:
        pathlib.Path(folder).mkdir(parents=True) 
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise 
    plt.plot(ratios, accs, '-o')
    plt.ylabel("classification accuracy")
    plt.xlabel("% components used")
    plt.tight_layout()
    plt.grid()
    plt.savefig("{}acc{}".format(folder, ext), dpi=300)

def conv_approx(hs, ratios, save_name, folder='figures/slices/', ext='.png'):
    vmax = np.max([np.max(h) for h in hs])
    vmin = np.min([np.min(h) for h in hs])
    print(vmax)
    #pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 
    try:
        pathlib.Path(folder).mkdir(parents=True) 
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    fig, axarr = plt.subplots(1, len(hs))
    for i in range(len(hs)):
        im = axarr[i].imshow(hs[i][0].astype(int), vmin=vmin,
                             vmax=vmax, interpolation='nearest')
        axarr[i].set_title(str((1-ratios[i])*100.) + '%')
        axarr[i].axis('off')
    add_colorbar(im)
    #fig.colorbar(im, ax=axarr.ravel().tolist())
    plt.savefig("{}{}{}".format(folder, save_name, ext), dpi=300)
    plt.clf()

def layer_01s(l1, l2, folder='figures/', ext='.png'):
    #pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 
    try:
        pathlib.Path(folder).mkdir(parents=True) 
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.plot(l1, label='l1')
    plt.plot(l2, label='l2')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.grid()
    plt.savefig("{}pct_1{}".format(folder, ext), dpi=300)
    plt.clf()

def approx_match(hs, ratios, do_type, folder='figures/', ext='.png'):
    #pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 
    try:
        pathlib.Path(folder).mkdir(parents=True) 
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.plot(ratios, hs)
    plt.ylabel("percent matched output")
    plt.xlabel("percent elements used in dot-product")
    plt.tight_layout()
    plt.grid()
    plt.savefig("{}match_{}{}".format(folder, do_type, ext), dpi=300)
    plt.clf()