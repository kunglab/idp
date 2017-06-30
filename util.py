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

class FilterDropout(function.Function):
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        if not hasattr(self, 'mask'):
            num_x = x[0].shape[0]
            num_f = x[0].shape[1]
            num_trim = int(self.dropout_ratio*num_f)
            scale = x[0].dtype.type(1. / (1 - self.dropout_ratio))
            xp = cuda.get_array_module(*x)
            a = np.arange(num_f)
            perms = np.argsort(np.random.rand(a.shape[0], num_x-1), axis=0)
            x_idxs = np.repeat(np.arange(x[0].shape[0]), num_trim)
            y_idxs = np.hstack((a[:, np.newaxis], a[perms])).T[:, :int(self.dropout_ratio*num_f)].flatten()
            if xp == np:
                flag = xp.ones(x[0].shape)
            else:
                flag = xp.ones(x[0].shape, dtype=np.float32)
            flag[x_idxs, y_idxs] = 0.0
            self.mask = scale * flag
        return x[0] * self.mask,

    def backward(self, x, gy):
        return gy[0] * self.mask,

def filter_dropout(x, ratio=.5, train=True):
    if train:
        return FilterDropout(ratio)(x)
    return x

def gen_prob(dist):
    if dist == 'exp':
        # return exp_prob(w=0.1666)
        return exp_prob(w=0.1)
    if dist == 'sexp':
        return exp_prob(w=0.025)
    if dist == 'linear':
        return linear_prob()
    if dist == 'id':
        return 0
    else:
        raise NameError('dist: {}'.format(dist))

def exp_prob(w=0.16666, bins=10):
    while True:
        do = np.random.exponential(w)
        if do <= 1.0:
            break
    # do = round(do*bins)/bins
    # do = min(max(1e-5, do), 0.99999)
    return do

def linear_prob(w=10):
    w += 1
    weights = np.linspace(0, 1, w)/np.sum(np.linspace(0, 1, w))
    return 1 - np.random.choice(range(w), p=weights)/float(w)

def pct_alike(x, y):
    x, y = x.flatten(), y.flatten()
    return len(np.where(x == y)[0]) / float(len(x))

def get_acc(model, dataset_tuple, ret_param='acc', batchsize=1024, gpu=0):
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

def get_approx_acc(model, dataset_tuple, comp_ratio, filter_ratio, batchsize=1024, gpu=0):
    xp = np if gpu < 0 else cuda.cupy
    x, y = dataset_tuple._datasets[0], dataset_tuple._datasets[1]
    accs = 0
    model.train = False
    for i in range(0, len(x), batchsize):
        x_batch = xp.array(x[i:i+batchsize])
        y_batch = xp.array(y[i:i+batchsize])
        acc_data = model(x_batch, y_batch, comp_ratio=comp_ratio, 
                         filter_ratio=filter_ratio, ret_param='acc')
        acc_data.to_cpu()
        acc = acc_data.data
        accs += acc*len(x_batch)
    return (accs / len(x)) * 100.

def get_approx_features(model, dataset_tuple, ratio, do_type='random', batchsize=1024, gpu=0):
    xp = np if gpu < 0 else cuda.cupy
    x, _ = dataset_tuple._datasets[0], dataset_tuple._datasets[1]
    model.train = False
    x_batch = xp.array(x[:batchsize])
    features = model.approx_features(x_batch, ratio=ratio, do_type=do_type)
    features.to_cpu()
    return features.data

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
    chainer.config.train = True
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
    chainer.config.train = False

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