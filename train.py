from chainer.datasets import get_mnist, get_cifar10
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import net
import util
from binary.bst import bst

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=128,
                    help='learning minibatch size')
parser.add_argument('--out', '-o', default='result',
                    help='Directory to output the result')
args = parser.parse_args()
train, test = get_mnist(ndim=3)
model = net.BinConvNet(10)
do_type = 'random'
util.train_model(model, train, test, args)
xf = bst(util.get_approx(model, test, 0, do_type)).data
xs, ratios = [], []
for ratio in np.linspace(0, 1, 100):
    x = bst(util.get_approx(model, test, ratio, do_type)).data
    alike = util.pct_alike(x, xf)
    xs.append(alike*100.)
    ratios.append((1 - ratio)*100.)


plt.plot(ratios, xs)
plt.ylabel("percent matched output")
plt.xlabel("percent elements used in dot-product")
plt.tight_layout()
plt.grid()
plt.savefig("figures/out_" + do_type + ".png", dpi=300)