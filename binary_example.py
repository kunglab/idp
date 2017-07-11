from approx.binary.links_depthwise_convolution_2d import IncompleteDepthwiseBinaryConvolution2D
from approx.binary.links_convolution_2d import IncompleteBinaryConvolution2D
from approx.binary.links_linear import IncompleteBinaryLinear
conv1 = IncompleteDepthwiseBinaryConvolution2D(16,1,3,1,1)
conv2 = IncompleteBinaryConvolution2D(16,16,1,1,0)
layer = IncompleteBinaryLinear(None,16)

h = np.random.randn(10,16,30,30).astype(np.float32)
h = conv1(h, [1]*16)
h = conv2(h, [1]*16)
h = layer(h)
l = F.mean_squared_error(h,np.random.randn(*h.shape).astype(np.float32))
l.backward()