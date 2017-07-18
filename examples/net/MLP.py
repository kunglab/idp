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
from idp.layer_profile import layer_profile
import util


class MLP(chainer.Chain):
    def __init__(self, class_labels, coeff_generator, profiles, n_units=100):
        super(MLP, self).__init__()
        self.coeff_generator = coeff_generator
        self.n_units = n_units
        self.profiles = profiles
        self.profile = 0
        with self.init_scope():
            # self.p0_l1 = IncompleteLinear(None, self.n_units)
            # self.p1_l1 = IncompleteLinear(None, self.n_units)
            # self.p2_l1 = IncompleteLinear(None, self.n_units)
            self.l1 = IncompleteLinear(None, self.n_units)
            self.l2 = IncompleteLinear(self.n_units, self.n_units)
            self.p0_l3 = IncompleteLinear(None, class_labels)
            self.p1_l3 = IncompleteLinear(None, class_labels)
            self.p2_l3 = IncompleteLinear(None, class_labels)

    def __call__(self, x, t, ret_param='loss', profile=None, comp_ratio=None):
        if profile == None:
            profile = self.profile

        import numpy as np
        params = layer_profile(self.coeff_generator,
                               *self.profiles[profile], self.n_units,
                               self.n_units, comp_ratio)
        if profile == 0:
            # h = F.relu(self.p0_l1(x))
            h = F.relu(self.l1(x, [1], [1], np.ones(self.n_units)))
            h = self.l2(h, *params)
            h = F.relu(h)
            h = self.p0_l3(h)
        elif profile == 1:
            # h = F.relu(self.p1_l1(x))
            h = F.relu(self.l1(x, [1], [0], np.zeros(self.n_units)))
            h = self.l2(h, *params)
            h = F.relu(h)
            h = self.p1_l3(h)
        elif profile == 2:
            # h = F.relu(self.p2_l1(x))
            h = F.relu(self.l1(x, [1], [0], np.zeros(self.n_units)))
            h = self.l2(h, *params)
            h = F.relu(h)
            h = self.p2_l3(h)


        report = {
            'loss': F.softmax_cross_entropy(h, t),
            'acc': F.accuracy(h, t)
        }

        reporter.report(report, self)
        return report[ret_param]

    def report_params(self):
        return ['validation/main/acc']

    def param_names(self):
        return 'MLP_{}_{}_p{}'.format(self.n_units, self.coeff_generator.__name__, len(self.profiles))
