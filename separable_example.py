
# coding: utf-8

# In[1]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[2]:

from approx.links_depthwise_convolution_2d import IncompleteDepthwiseConvolution2D


# In[3]:

conv = IncompleteDepthwiseConvolution2D(16,1,3,1,1)


# In[5]:

import numpy as np
v = conv(np.random.randn(10,16,30,30).astype(np.float32), [1]*16)


# In[6]:

import chainer.functions as F

l = F.mean_squared_error(v,np.random.randn(10,16,30,30).astype(np.float32))
# v.shape


# In[7]:

l.backward()


# In[8]:

from approx.links_convolution_2d import IncompleteConvolution2D


# In[10]:

conv = IncompleteConvolution2D(16,16,1,1,0)


# In[11]:

import numpy as np
v = conv(np.random.randn(10,16,30,30).astype(np.float32), [1]*16)


# In[13]:

import chainer.functions as F

l = F.mean_squared_error(v,np.random.randn(10,16,30,30).astype(np.float32))
# v.shape


# In[14]:

l.backward()


# In[16]:

conv1 = IncompleteDepthwiseConvolution2D(16,1,3,1,1)
conv2 = IncompleteConvolution2D(16,16,1,1,0)

h = np.random.randn(10,16,30,30).astype(np.float32)
h = conv1(h, [1]*16)
h = conv2(h, [1]*16)


# In[ ]:



