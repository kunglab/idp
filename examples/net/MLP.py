import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import reporter

from idp.links_linear import IncompleteLinear
import idp.coeffs_generator as cg
from idp.blackout_generator import gen_blackout
import util


class MLP(chainer.Chain):
    def __init__(self, class_labels, coeff_generator, n_units=100):
        super(MLP, self).__init__()
        self.coeff_generator = coeff_generator
        self.n_units = n_units
        self.profile = 0
        with self.init_scope():
            self.p0_l1 = IncompleteLinear(None,self.n_units)
            self.p1_l1 = IncompleteLinear(None,self.n_units)
            self.l2 = IncompleteLinear(self.n_units,self.n_units)
            self.p0_l3 = L.Linear(None,class_labels)
            self.p1_l3 = L.Linear(None,class_labels)

    def __call__(self, x, t, ret_param='loss', profile=None, comp_ratio=None):
        if profile == None:
            profile = self.profile

        def coeff_f(n):
            return util.zero_end(self.coeff_generator(n), comp_ratio)

        if profile == 0:
            h = F.relu(self.p0_l1(x))
            #h = self.p0_l1(x, coeff_f(784),
            #            cg.step(784, steps=[1, 1]),
            #            cg.step(self.n_units, steps=[1, 0]))
            #h = F.relu(h)
            # print('b',self.l2.W.data)
            h = self.l2(h, coeff_f(self.n_units),
                        cg.step(self.n_units, steps=[1, 0]),
                        cg.step(self.n_units, steps=[1, 0]))
            h = F.relu(h)
            h = self.p0_l3(h)
        elif profile == 1:
            h = F.relu(self.p1_l1(x))
            #h = self.p0_l1(x, coeff_f(784),
            #            cg.step(784, steps=[1, 1]),
            #            cg.step(self.n_units, steps=[0, 1]))
            #h = F.relu(h)
            h = self.l2(h, coeff_f(self.n_units),
                        cg.step(self.n_units, steps=[0, 1]),
                        cg.step(self.n_units, steps=[0, 1]))
            h = F.relu(h)
            h = self.p1_l3(h)
        else:
            raise ValueError('profile: {}'.format(profile))

        report = {
            'loss': F.softmax_cross_entropy(h, t),
            'acc': F.accuracy(h, t)
        }

        reporter.report(report, self)
        return report[ret_param]

    def profiles(self):
        return [0, 1]

    def report_params(self):
        return ['validation/main/acc']

    def param_names(self):
        return 'MLP_{}_{}'.format(self.n_units, self.coeff_generator.__name__)
