import os
import argparse
from random import shuffle
from functools import partial
from itertools import combinations

import chainer
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer.functions import dropout
from chainer import optimizers, optimizer
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.serializers import load_hdf5, save_hdf5
from chainer.datasets import get_mnist, get_cifar10
import numpy as np
from chainer import cuda
from chainer import function
from chainer.utils import type_check


def zero_end(coefs, coef_ratio):
    if coef_ratio is None:
        return np.array(coefs)
    coefs = np.array(coefs)
    coefs[int(coef_ratio * len(coefs)):] = 0
    return coefs


def gen_prob(dist):
    if dist == 'exp':
        return exp_prob(w=0.25)
    if dist == 'sexp':
        return exp_prob(w=0.05)
    if dist == 'mid_exp':
        return min(1.0, 0.3 + exp_prob(w=0.025))
    if dist == '0':
        return 0.0
    if dist == '50':
        return 0.5
    if dist == '90':
        return 0.9
    if dist == 'linear':
        return linear_prob()
    if dist == 'id':
        return 0
    else:
        raise NameError('dist: {}'.format(dist))


def exp_prob(w=0.16666):
    while True:
        do = np.random.exponential(w)
        if do <= 1.0:
            break
    return do


def linear_prob(w=10):
    w += 1
    weights = np.linspace(0, 1, w) / np.sum(np.linspace(0, 1, w))
    return 1 - np.random.choice(range(w), p=weights) / float(w)


def get_acc(model, dataset_tuple, ret_param='acc', batchsize=1024, gpu=0):
    xp = np if gpu < 0 else cuda.cupy
    x, y = dataset_tuple._datasets[0], dataset_tuple._datasets[1]
    accs = 0
    model.train = False
    for i in range(0, len(x), batchsize):
        x_batch = xp.array(x[i:i + batchsize])
        y_batch = xp.array(y[i:i + batchsize])
        acc_data = model(x_batch, y_batch, ret_param=ret_param)
        acc_data.to_cpu()
        acc = acc_data.data
        accs += acc * len(x_batch)
    return (accs / len(x)) * 100.


def get_idp_acc(model, dataset_tuple, comp_ratio, batchsize=1024, gpu=0):
    xp = np if gpu < 0 else cuda.cupy
    x, y = dataset_tuple._datasets[0], dataset_tuple._datasets[1]
    accs = 0
    model.train = False
    for i in range(0, len(x), batchsize):
        x_batch = xp.array(x[i:i + batchsize])
        y_batch = xp.array(y[i:i + batchsize])
        acc_data = model(x_batch, y_batch, comp_ratio=comp_ratio,
                         ret_param='acc')
        acc_data.to_cpu()
        acc = acc_data.data
        accs += acc * len(x_batch)
    return (accs / len(x)) * 100.


def sweep_idp(model, dataset, comp_ratios, args):
    accs = []
    for cr in comp_ratios:
        accs.append(get_idp_acc(model, dataset, comp_ratio=cr,
                                batchsize=args.batchsize, gpu=args.gpu))
    return accs


def train_model(model, train, test, args):
    chainer.config.train = True
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    xp = np if args.gpu < 0 else cuda.cupy
    if args.opt == 'momentum':
        opt = chainer.optimizers.MomentumSGD(args.learnrate)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.WeightDecay(5e-4))
    elif args.opt == 'adam':
        opt = optimizers.Adam(args.learnrate)
        opt.setup(model)
    else:
        raise NameError('Invalid opt: {}'.format(args.opt))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, opt, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    if args.opt == 'momentum':
        trainer.extend(extensions.ExponentialShift('lr', 0.5),
                       trigger=(25, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss'] +
        model.report_params() +
        ['elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

    name = model.param_names()
    save_model(model, os.path.join(args.model_path, name))
    chainer.config.train = False
    with open(os.path.join(args.out, 'log'), 'r') as fp:
        return eval(fp.read().replace('\n', ''))


def save_model(model, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_hdf5(os.path.join(folder, 'model.hdf5'), model)
    return model


def load_model(model, folder, gpu=0):
    load_hdf5(os.path.join(folder, 'model.hdf5'), model)
    if gpu >= 0:
        model = model.to_gpu(gpu)
    return model


def load_or_train_model(model, train, test, args):
    name = model.param_names()
    model_folder = os.path.join(args.model_path, name)
    if not os.path.exists(model_folder) or args.overwrite_models:
        train_model(model, train, test, args)
    else:
        load_model(model, model_folder, gpu=args.gpu)


def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        return get_mnist(ndim=3)
    if dataset_name == 'cifar10':
        return get_cifar10(ndim=3)
    raise NameError('{}'.format(dataset_name))


# small vs large only from device perspective
# def get_net_settings(dataset_name, size='large'):
#     if dataset_name == 'mnist' and size == 'large':
#         return (8, 4), 16, None
#     if dataset_name == 'mnist' and size == 'small':
#         return (8, 2), 16, None
#     if dataset_name == 'cifar10' and size == 'large':
#         return 16, 64, 128
#     if dataset_name == 'cifar10' and size == 'small':
#         return (16, 8), 32, 64
#     raise NameError('{}'.format(dataset_name))


def default_parser(description=''):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dataset', '-d', default='mnist',
                        choices=['mnist', 'cifar10'], help='dataset name')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='_output',
                        help='Directory to output the result')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--opt',  default='momentum',
                        choices=['momentum', 'adam'], help='optimizer')
    parser.add_argument('--model_path',  default='_models/',
                        help='Directory to store the models (for later use)')
    parser.add_argument('--figure_path',  default='_figures/',
                        help='Directory to store the generated figures')
    parser.add_argument('--overwrite_models', action='store_true',
                        help='If true, reruns a setting and overwrites old models')
    parser.add_argument('--ext', default='png', choices=['png', 'pdf'])
    return parser
