import numpy as np
import chainer.functions as F

from approx.links_depthwise_convolution_2d import IncompleteDepthwiseConvolution2D
from approx.links_convolution_2d import IncompleteConvolution2D

conv = IncompleteDepthwiseConvolution2D(16,1,3,1,1)
v = conv(np.random.randn(10,16,30,30).astype(np.float32), [1]*16)
l = F.mean_squared_error(v,np.random.randn(10,16,30,30).astype(np.float32))
l.backward()
conv = IncompleteConvolution2D(16,16,1,1,0)
v = conv(np.random.randn(10,16,30,30).astype(np.float32), [1]*16)
l = F.mean_squared_error(v,np.random.randn(10,16,30,30).astype(np.float32))
l.backward()
conv1 = IncompleteDepthwiseConvolution2D(16,1,3,1,1)
conv2 = IncompleteConvolution2D(16,16,1,1,0)
h = np.random.randn(10,16,30,30).astype(np.float32)
h = conv1(h, [1]*16)
h = conv2(h, [1]*16)