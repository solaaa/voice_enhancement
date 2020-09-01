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


def complex_conv_v1(x_r, x_i, kernel_size, stride, pad_pattern, dilations = [1,1,1,1], name='c'):
    w_r = tf.Variable(
                tf.random.truncated_normal(
                kernel_size,
                stddev = np.sqrt(1/(kernel_size[0]*kernel_size[1]*kernel_size[2]))),
                name=name+'_w_r')
    w_i = tf.Variable(
                tf.random.truncated_normal(
                kernel_size,
                stddev = np.sqrt(1/(kernel_size[0]*kernel_size[1]*kernel_size[2]))),
                name=name+'_w_i')

    c_rr = tf.nn.conv2d(x_r, w_r, stride, 
                            dilations=dilations, 
                            padding=pad_pattern)
    c_ri = tf.nn.conv2d(x_r, w_i, stride, 
                            dilations=dilations, 
                            padding=pad_pattern)
    c_ii = tf.nn.conv2d(x_i, w_i, stride, 
                            dilations=dilations, 
                            padding=pad_pattern)
    c_ir = tf.nn.conv2d(x_i, w_r, stride, 
                            dilations=dilations, 
                            padding=pad_pattern)

    
    c_real = c_rr - c_ii
    c_imag = c_ri + c_ir
    return c_real, c_imag

def complex_conv_v2(x_r, x_i, kernel_size, stride, pad_pattern, dilations = [1,1,1,1], name='c'):
    w_rr = tf.Variable(
                tf.random.truncated_normal(
                kernel_size,
                stddev = np.sqrt(1/(kernel_size[0]*kernel_size[1]*kernel_size[2]))),
                name=name+'_w_rr')
    w_ii = tf.Variable(
                tf.random.truncated_normal(
                kernel_size,
                stddev = np.sqrt(1/(kernel_size[0]*kernel_size[1]*kernel_size[2]))),
                name=name+'_w_ii')
    w_ri = tf.Variable(
                tf.random.truncated_normal(
                kernel_size,
                stddev = np.sqrt(1/(kernel_size[0]*kernel_size[1]*kernel_size[2]))),
                name=name+'_w_ri')
    w_ir = tf.Variable(
                tf.random.truncated_normal(
                kernel_size,
                stddev = np.sqrt(1/(kernel_size[0]*kernel_size[1]*kernel_size[2]))),
                name=name+'_w_ir')

    c_rr = tf.nn.conv2d(x_r, w_rr, stride, 
                            dilations=dilations, 
                            padding=pad_pattern)
    c_ri = tf.nn.conv2d(x_r, w_ri, stride, 
                            dilations=dilations, 
                            padding=pad_pattern)
    c_ii = tf.nn.conv2d(x_i, w_ii, stride, 
                            dilations=dilations, 
                            padding=pad_pattern)
    c_ir = tf.nn.conv2d(x_i, w_ir, stride, 
                            dilations=dilations, 
                            padding=pad_pattern)

    
    c_real = c_rr - c_ii
    c_imag = c_ri + c_ir
    return c_real, c_imag


def complex_dconv(x_r, x_i, kernel_size, stride, batch_size=1, name='dc'):
    w_r = tf.Variable(
                tf.random.truncated_normal(
                kernel_size,
                stddev = np.sqrt(1/(kernel_size[0]*kernel_size[1]*kernel_size[2]))),
                name=name+'_w_r')
    w_i = tf.Variable(
                tf.random.truncated_normal(
                kernel_size,
                stddev = np.sqrt(1/(kernel_size[0]*kernel_size[1]*kernel_size[2]))),
                name=name+'_w_i')

    c_rr = tf.nn.conv2d_transpose(x_r, w_r, 
                                  output_shape=[batch_size, x_r.get_shape().as_list()[1], 2*x_r.get_shape().as_list()[2],kernel_size[2]], 
                                  strides=stride)
    c_ri = tf.nn.conv2d_transpose(x_r, w_i, 
                                  output_shape=[batch_size, x_r.get_shape().as_list()[1], 2*x_r.get_shape().as_list()[2],kernel_size[2]], 
                                  strides=stride)
    c_ir = tf.nn.conv2d_transpose(x_i, w_r, 
                                  output_shape=[batch_size, x_r.get_shape().as_list()[1], 2*x_r.get_shape().as_list()[2],kernel_size[2]], 
                                  strides=stride)
    c_ii = tf.nn.conv2d_transpose(x_i, w_i, 
                                  output_shape=[batch_size, x_r.get_shape().as_list()[1], 2*x_r.get_shape().as_list()[2],kernel_size[2]], 
                                  strides=stride)
    
    c_real = c_rr - c_ii
    c_imag = c_ri + c_ir
    return c_real, c_imag


class moving_average:
    def __init__(self, init_avg=0.5, beta=0.99):
        self.val = 0.
        self.beta = beta
        self.accumulate = init_avg
    def update(self, x):
        self.val = x
        self.accumulate = self.accumulate*self.beta + (1-self.beta)*self.val


def activation(x):
    if 1:
        alpha = 0.3
        x = tf.keras.layers.LeakyReLU(alpha)(x)
    elif 0:
        alpha = 1.
        x = tf.clip_by_value(x,-alpha,alpha)
    return x

def SI_SNR(clean_wav_, estimated_wav_):
    s_target = tf.reshape(tf.reduce_sum(clean_wav_*estimated_wav_, axis=1), [-1,1])*clean_wav_/tf.reshape(tf.reduce_sum(tf.square(clean_wav_),axis=1), [-1,1])
    e_noise = estimated_wav_-clean_wav_
    loss_ = tf.reduce_sum(tf.square(s_target),axis=1)/tf.reduce_sum(tf.square(e_noise),axis=1)
    loss_ = 10.*tf.log(loss_)/tf.log(10.)
    loss_ = -1. * tf.reduce_mean(loss_, axis=0) # the bigger SNR is, the better res is, so to minimize -1 * loss
    return loss_