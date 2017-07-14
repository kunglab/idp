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
    def __init__(self, class_labels, coeff_generator, bo_generator, n_units=100):
        super(MLP, self).__init__()
        self.coeff_generator = coeff_generator
        self.bo_generator = bo_generator
        self.n_units = n_units
        with self.init_scope():
            self.l1 = L.Linear(self.n_units)
            self.l2 = IncompleteLinear(self.n_units)
            self.l3 = L.Linear(class_labels)

    def __call__(self, x, t, ret_param='loss', comp_ratio=None):
        if comp_ratio == None:
            comp_ratio = 1 - gen_blackout(self.bo_generator)

        def coeff_f(n):
            return util.zero_end(self.coeff_generator(n), comp_ratio)

        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h, coeff_f(self.n_units)))
        h = self.l3(h)

        report = {
            'loss': F.softmax_cross_entropy(h, t),
            'acc': F.accuracy(h, t)
        }

        reporter.report(report, self)
        return report[ret_param]

    def report_params(self):
        return ['validation/main/acc']

    def param_names(self):
        return 'MLP_{}_{}_{}'.format(self.n_units, self.coeff_generator.__name__,
                                     self.bo_generator)
