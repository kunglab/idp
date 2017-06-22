import os
from random import shuffle

from functools import partial
from itertools import combinations
import numpy
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
import numpy as np
from numpy.random import RandomState

from chainer import cuda
from chainer import function
from chainer.utils import type_check

def pct_alike(x, y):
    x, y = x.flatten(), y.flatten()
    return len(np.where(x == y)[0]) / float(len(x))

def get_acc(model, dataset_tuple, ret_param='acc0', batchsize=128, gpu=0):
    xp = np if gpu < 0 else cuda.cupy
    x, y = dataset_tuple._datasets[0], dataset_tuple._datasets[1]
    accs = 0
    model.train = False
    for i in range(0, len(x), batchsize):
        x_batch = xp.array(x[i:i+batchsize])
        y_batch = xp.array(y[i:i+batchsize])
        acc_data = model(x_batch, y_batch, ret_param=ret_param)
        acc_data.to_cpu()
        acc = acc_data.data
        accs += acc*len(x_batch)
    return (accs / len(x)) * 100.

def get_approx_acc(model, dataset_tuple, ratio, do_type, batchsize=128, gpu=0):
    xp = np if gpu < 0 else cuda.cupy
    x, y = dataset_tuple._datasets[0], dataset_tuple._datasets[1]
    accs = 0
    model.train = False
    for i in range(0, len(x), batchsize):
        x_batch = xp.array(x[i:i+batchsize])
        y_batch = xp.array(y[i:i+batchsize])
        acc_data = model.approx(x_batch, y_batch, ratio=ratio, do_type=do_type)
        acc_data.to_cpu()
        acc = acc_data.data
        accs += acc*len(x_batch)
    return (accs / len(x)) * 100.

def get_layer(model, dataset_tuple, layer, batchsize=1024, gpu=0):
    xp = np if gpu < 0 else cuda.cupy
    x, _ = dataset_tuple._datasets[0], dataset_tuple._datasets[1]
    for i in range(0, len(x), batchsize):
        x_batch = xp.array(x[i:i+batchsize])
        return model.layer(x_batch, layer)

def get_class_acc(model, dataset_tuple, batchsize=128, gpu=0):
    xp = np if gpu < 0 else cuda.cupy
    x, y = dataset_tuple._datasets[0], dataset_tuple._datasets[1]
    model.train = False
    for i in range(0, len(x), batchsize):
        x_batch = xp.array(x[i:i+batchsize])
        y_batch = xp.array(y[i:i+batchsize])
        y_hat = model(x_batch, y_batch, ret_param='y_hat')
        return y_hat

def train_model(model, train, test, args, out=None):
    if not out:
        out = args.out
    batchsize = args.batchsize
    n_epoch = args.epoch
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    xp = np if args.gpu < 0 else cuda.cupy
    lr_start = 0.00003
    opt = optimizers.Adam(lr_start)
    opt.setup(model)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, opt, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss'] +
        model.report_params() +
        ['elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

    name = model.param_names()
    save_model(model, os.path.join(out, name))
    with open(os.path.join(out, 'log'), 'r') as fp:
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

def load_or_train_model(model, train, test, args, gpu=0):
    name = model.param_names()
    if not os.path.exists(os.path.join(args.out, name)):
        train_model(model, train, test, args, out=args.out)
    else:
        load_model(model, os.path.join(args.out, name), gpu=gpu)