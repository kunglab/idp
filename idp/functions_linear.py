from chainer import function
from chainer.utils import type_check
from chainer import cuda
import numpy

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


class LinearFunction(function.Function):
    
    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def __init__(self, coeffs, bcoeffs=None, ocoeffs=None):
        self.coeffs = coeffs
        self.bcoeffs = bcoeffs
        self.ocoeffs = ocoeffs
        
    def forward(self, inputs):
        x = _as_mat(inputs[0])
        W = inputs[1]

        xp = cuda.get_array_module(*x)

        olen, ilen = W.shape
        if self.coeffs is None:
            self.coeffs = numpy.ones(ilen)
        coeffs = numpy.copy(self.coeffs)
        coeffs = numpy.expand_dims(coeffs, 0)        
        coeffs = numpy.broadcast_to(coeffs, W.shape)
        M = xp.asarray(coeffs,numpy.float32).reshape(W.shape)
        self.M = M
        W = self.M*W
        
        if not type_check.same_types(*inputs):
            raise ValueError('numpy and cupy must not be used together\n'
                             'type(W): {0}, type(x): {1}'
                             .format(type(W), type(x)))

        y = x.dot(W.T).astype(x.dtype, copy=False)
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,

    def backward(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        
        if self.bcoeffs is not None:
            xp = cuda.get_array_module(*x)
            coeffs = numpy.copy(self.bcoeffs)
            coeffs = numpy.expand_dims(coeffs, 0)
            coeffs = numpy.broadcast_to(coeffs, W.shape)
            self.mW = xp.asarray(coeffs,numpy.float32).reshape(W.shape)
        if self.ocoeffs is not None:
            xp = cuda.get_array_module(*x)
            coeffs = numpy.copy(self.ocoeffs)
            self.mb = xp.asarray(coeffs,numpy.float32)

        W = self.M * W
        gy = grad_outputs[0]

        if not type_check.same_types(*inputs):
            raise ValueError('numpy and cupy must not be used together\n'
                             'type(W): {0}, type(x): {1}'
                             .format(type(W), type(x)))

        gx = gy.dot(W).astype(x.dtype, copy=False).reshape(inputs[0].shape)
        gW = gy.T.dot(x).astype(W.dtype, copy=False)
        
        if hasattr(self,'mW'):
            gW = self.mW * gW
            if hasattr(self,'mb'):
                xp = cuda.get_array_module(*x)
                gW = xp.broadcast_to(xp.expand_dims(self.mb,1),gW.shape) * gW
        # print('gW',gW.sum(0).sum(0))
        if len(inputs) == 3:
            gb = gy.sum(0)
            if hasattr(self,'mb'):
                gb = self.mb * gb
            return gx, gW, gb
        else:
            return gx, gW

def linear(x, coeffs, W, b=None, bcoeffs=None, ocoeffs=None):
    """Linear function, or affine transformation.

    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes
    .. math:: Y = xW^\\top + b.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable, which is a :math:`(s_B, s_1, \
            s_2, ..., s_n)`-shaped float array. Its first dimension
            :math:`(s_B)` is assumed to be the *minibatch dimension*. The
            other dimensions are treated as concatenated one dimension whose
            size must be :math:`(s_1 * ... * s_n = N)`.
        W (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Weight variable of shape :math:`(M, N)`,
            where :math:`(N = s_1 * ... * s_n)`.
        b (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Bias variable (optional) of shape
            :math:`(M,)`.

    Returns:
        ~chainer.Variable: Output variable. A float array with shape
        of :math:`(s_B, M)`.

    .. seealso:: :class:`~chainer.links.Linear`

    .. admonition:: Example

        >>> x = np.random.uniform(0, 1, (3, 4)).astype('f')
        >>> W = np.random.uniform(0, 1, (5, 4)).astype('f')
        >>> b = np.random.uniform(0, 1, (5,)).astype('f')
        >>> y = F.linear(x, W, b)
        >>> y.shape
        (3, 5)

    """
    if b is None:
        return LinearFunction(coeffs, bcoeffs=bcoeffs, ocoeffs=ocoeffs)(x, W)
    else:
        return LinearFunction(coeffs, bcoeffs=bcoeffs, ocoeffs=ocoeffs)(x, W, b)
