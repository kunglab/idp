import numpy

from chainer import cuda
from chainer.cuda import cupy
from chainer import function
from chainer.utils import type_check
from chainer import functions as F

class BST(function.Function):
    """Binary with Straight Thourgh estimator Unit."""
    def __init__(self):
        pass

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        y = x[0]
        y = numpy.where(y>=0, 1, -1).astype(numpy.float32, copy=False)
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
            'T x', 'T y',
            'y = x >= 0 ? 1 : -1', 'bst_fwd')(
                x[0])
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        zero_indices = numpy.abs(x[0]) > 1
        gx[zero_indices] = 0
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx',
            'gx = abs(x) > 1 ? 0 : gy', 'bst_bwd')(
                x[0], gy[0])
        return gx,
    
class SBST(function.Function):
    """Stochastic Binary with Straight Thourgh estimator Unit."""
    def __init__(self):
        pass

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        xp = cuda.get_array_module(*x)
        
        y = x[0]
        
        p = F.hard_sigmoid(y)
        ys = xp.random.choice([1,0],numpy.prod(y.shape),p=[p,1-p]).reshape(y.shape)        
        y = numpy.where(ys>0, 1, -1).astype(numpy.float32, copy=False)
        return y,

    def forward_gpu(self, x):
        xp = cuda.get_array_module(*x)
        y = x[0]
        p = F.hard_sigmoid(y)
        ys = xp.random.choice([1,0],numpy.prod(y.shape),p=[p,1-p]).reshape(y.shape)        
        y = cuda.elementwise(
            'T x', 'T y',
            'y = x > 0 ? 1 : -1', 'bst_fwd')(
                ys)
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        zero_indices = numpy.abs(x[0]) > 1
        gx[zero_indices] = 0
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx',
            'gx = abs(x) > 1 ? 0 : gy', 'bst_bwd')(
                x[0], gy[0])
        return gx,


class ZBST(function.Function):
    """Zero + Binary with Straight Thourgh estimator Unit."""
    def __init__(self):
        pass

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        return np.sign(x[0]),

    def forward_gpu(self, x):
        return cupy.sign(x[0]),

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        zero_indices = numpy.abs(x[0]) > 1
        gx[zero_indices] = 0
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx',
            'gx = abs(x) > 1 ? 0 : gy', 'zbst_bwd')(
                x[0], gy[0])
        return gx,

class MBST(function.Function):
    """Magnitude + Binary with Straight Thourgh estimator Unit."""
    def __init__(self, m):
        self.m = m

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        gt = x[0] > self.m
        lt = x[0] < -self.m
        x[0][gt] = 1
        x[0][lt] = -1
        x[0][~(gt | lt)] = 0
        return x

    def forward_gpu(self, x):
        gt = x[0] > self.m
        lt = x[0] < -self.m
        x[0][gt] = 1
        x[0][lt] = -1
        x[0][~(gt | lt)] = 0
        return x

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        zero_indices = numpy.abs(x[0]) > 1
        gx[zero_indices] = 0
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy', 'T gx',
            'gx = abs(x) > 1 ? 0 : gy', 'mbst_bwd')(
                x[0], gy[0])
        return gx,

class MBST_BP(function.Function):
    """Magnitude + Binary with Straight Thourgh estimator Unit w/ backward margin as well"""
    def __init__(self, m):
        self.m = m

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        gt = x[0] > self.m
        lt = x[0] < -self.m
        x[0][gt] = 1
        x[0][lt] = -1
        x[0][~(gt | lt)] = 0
        return x

    def forward_gpu(self, x):
        gt = x[0] > self.m
        lt = x[0] < -self.m
        x[0][gt] = 1
        x[0][lt] = -1
        x[0][~(gt | lt)] = 0
        return x

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        zero_indices = numpy.abs(x[0]) > self.m
        gx[zero_indices] = 0
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy, T m', 'T gx',
            'gx = abs(x) > m ? 0 : gy', 'mbst2_bwd')(
                x[0], gy[0], self.m)
        return gx,


def mbst_bp(x, m):
    return MBST_BP(m)(x)

def mbst(x, m):
    return MBST(m)(x)

def zbst(x):
    return ZBST()(x)

def bst(x):
    return BST()(x)

def sbst(x):
    return SBST()(x)