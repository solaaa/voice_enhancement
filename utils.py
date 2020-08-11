from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
import random



def conv_layer(x, kernel_size, stride, pad_pattern, dilations = [1,1,1,1], is_wn=False, name='c'):
    '''
    -----------------------------------------------
    x:           [batch, w, h, c]
    kernel_size: [w_k, h_k, input_channel(x_c), output_channel]
    stride:      [1, w_s, h_s, 1]
    pad_pattern: 'SAME' or 'VALID'
    dilations:   [1, d, d, 1]
    is_wn:       bool, weights normalization 
    '''
    
    w = tf.Variable(
                tf.random.truncated_normal(
                kernel_size,
                stddev = np.sqrt(1/(kernel_size[0]*kernel_size[1]*kernel_size[2]))),
                name=name+'_w')
    b = tf.Variable(
                tf.random.truncated_normal(
                [kernel_size[3],], 
                stddev=0.001),
                name=name+'_b')

    if is_wn:
        g = tf.Variable(
             tf.random.truncated_normal(
                [kernel_size[3]], 
                stddev=np.sqrt(0.001)),
                 name=name+'_g',trainable=True)

        w = tf.reshape(g/1.,[1,1,1,kernel_size[3]])*tf.nn.l2_normalize(w, axis=[0,1,2])

    c_out = tf.nn.conv2d(x, w, stride, 
                             dilations=dilations, 
                             padding=pad_pattern)+ b
    print(name+'x shape'+str(x.shape)+'  kernel_size shape'+ str(kernel_size))

    print('w: ' + str(w.shape) + 'b: ' + str(b.shape))
    print('c_out: ' + str(c_out.shape))



    return c_out


def fc_layer_chip(x, last_layer_element_count, unit_num, is_wn=False, name='fc'):
    w = tf.Variable(
             tf.random.truncated_normal(
                [last_layer_element_count, unit_num], 
                stddev=np.sqrt(1/last_layer_element_count)),
                name=name+'_w')
    b = tf.Variable(
             tf.random.truncated_normal(
                [unit_num], 
                stddev=0.001),
                name=name+'_b')

    if is_wn:
        g = tf.Variable(
             tf.random.truncated_normal(
                [unit_num], 
                stddev=np.sqrt(0.001)),
                 name=name+'_g',trainable=True)
        w = g/1. * tf.nn.l2_normalize(w, dim=0)

    fc_out = tf.matmul(x, w) + b
    return fc_out


def identity_block(inp, kernel_size, stride, pad_pattern, dilations, name='b'):
    x = conv_layer(inp, kernel_size, stride, pad_pattern, dilations, name=name+'_c1')
    x = tf.nn.relu(x)
    x = conv_layer(inp, kernel_size, stride, pad_pattern, dilations, name=name+'_c2')

    x = tf.add(x, inp)
    x = tf.nn.relu(x)
    return x

