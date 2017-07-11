import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import reporter

from idp.links_linear import IncompleteLinear
from idp.coeffs_generator import uniform_seq, harmonic_seq, linear_seq, exp_seq, uniform_exp_seq
import util

class MLP(chainer.Chain):
    def __init__(self, n_units, n_out, coeff_generator):
        super(MLP, self).__init__()
        self.n_units = n_units
        self.n_out = n_out
        self.coeff_generator = coeff_generator
        with self.init_scope():
            self.l1 = L.Linear(n_units)
            self.l2 = IncompleteLinear(n_units)
            self.l3 = L.Linear(n_out)

    def __call__(self, x, t, ret_param='loss', comp_ratio=None):
        if comp_ratio == None:
            comp_ratio = 1 - util.gen_prob('sexp')

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
        return 'MLP_{}_{}'.format(self.n_units, self.coeff_generator.__name__)
