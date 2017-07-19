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
from idp.binary import weight_clip


def zero_end(coefs, coef_ratio):
    if coef_ratio is None:
        return np.array(coefs)
    coefs = np.array(coefs)
    coefs[int(coef_ratio * len(coefs)):] = 0
    return coefs


def get_acc(model, dataset_tuple, ret_param='acc', batchsize=1024, gpu=0):
    chainer.config.train = False
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


def get_idp_acc(model, dataset_tuple, comp_ratio, profile=None, batchsize=1024, gpu=0):
    chainer.config.train = True
    xp = np if gpu < 0 else cuda.cupy
    x, y = dataset_tuple._datasets[0], dataset_tuple._datasets[1]
    accs = 0
    model.train = False
    for i in range(0, len(x), batchsize):
        x_batch = xp.array(x[i:i + batchsize])
        y_batch = xp.array(y[i:i + batchsize])
        if profile == None:
            acc_data = model(x_batch, y_batch, comp_ratio=comp_ratio,
                            ret_param='acc')
        else:
            acc_data = model(x_batch, y_batch, comp_ratio=comp_ratio,
                            ret_param='acc', profile=profile)
        acc_data.to_cpu()
        acc = acc_data.data
        accs += acc * len(x_batch)
    return (accs / len(x)) * 100.


def init_model(model, dataset_tuple, profile=0, batchsize=1024, gpu=0):
    xp = np if gpu < 0 else cuda.cupy
    x, y = dataset_tuple._datasets[0], dataset_tuple._datasets[1]
    x_batch = xp.array(x[:batchsize])
    y_batch = xp.array(y[:batchsize])
    _ = model(x_batch, y_batch, profile=profile)


def sweep_idp(model, dataset, comp_ratios, args, profile=None):
    chainer.config.train = False
    accs = []
    for cr in comp_ratios:
        accs.append(get_idp_acc(model, dataset, comp_ratio=cr,
                                batchsize=args.batchsize, gpu=args.gpu,
                                profile=profile))
    return accs


def train_model_profiles(model, train, test, args):
    chainer.config.train = True
    name = model.param_names()
    model_folder = os.path.join(args.model_path, name)

    print(model_folder)
    # load model
    if os.path.exists(model_folder) or args.overwrite_models:
        load_model(model, model_folder, gpu=args.gpu)
        return

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # init the params for each profiles
    for profile in range(len(model.profiles)):
        init_model(model, train, profile=profile)

    # train each model
    for profile in range(len(model.profiles)):
        model.profile = profile
        train_model(model, train, test, args)


def train_model(model, train, test, args):
    chainer.config.train = True
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    xp = np if args.gpu < 0 else cuda.cupy
    if args.opt == 'sgd':
        opt = chainer.optimizers.SGD(args.learnrate)
        opt.setup(model)
        if hasattr(args, 'decay'):
            opt.add_hook(chainer.optimizer.WeightDecay(5e-4))
    elif args.opt == 'momentum':
        opt = chainer.optimizers.MomentumSGD(args.learnrate)
        opt.setup(model)
        if hasattr(args, 'decay'):
            opt.add_hook(chainer.optimizer.WeightDecay(5e-4))
    elif args.opt == 'adam':
        opt = optimizers.Adam(args.learnrate)
        opt.setup(model)
    else:
        raise NameError('Invalid opt: {}'.format(args.opt))
    if hasattr(args, 'clip'):
        opt.add_hook(weight_clip.WeightClip(-args.clip, args.clip))

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
