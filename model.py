import math
import numpy as np

import tensorflow as tf

def get_model(inp, name):
    if name == 'cnn_version_1':
        return cnn_v1(inp)
    else:
        print('[{0}] does not exsist'.format(name))

def cnn_v1(inp):
    # inp shape = [batch, time]
    x = tf.reshape(inp, [-1, inp.get_shape().as_list()[1], 1, 1]) # [batch, time , 1, 1]
    # config
    conv_layer_num = 2
    kernel = 16
    stride = [1,1,1,1]
    pad_pattern = 'SAME'
    dilation = [1,1,1,1]
    channel = [16,16]


    for i in range(conv_layer_num):
        if i == 0:
            x = conv_layer(x, [kernel,1,1,channel[i]], 
                           stride, pad_pattern, dilations = dilation, 
                           is_wn=False, name='c%d'%(i))
        else:
            x = conv_layer(x, [kernel,1,channel[i-1],channel[i]], 
                           stride, pad_pattern, dilations = dilation, 
                           is_wn=False, name='c%d'%(i))
        x = tf.nn.relu(x)

    # output layer
    # make [batch, time, 1, channel] --> [batch, time, 1, 1] --> [batch, time]

    out = conv_layer(x, [kernel, 1, channel[-1], 1], 
                      [1,1,1,1], pad_pattern, dilations = [1,1,1,1], 
                      is_wn=False, name='c%d'%(i))
    out = tf.reshape(out, [-1, inp.get_shape().as_list()[1]])

    return out