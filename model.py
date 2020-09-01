import math
import numpy as np
from utils import *
import tensorflow as tf

def get_model(inp, settings):
    if settings['model_selection'] == 'cnn_version_1':
        return cnn_v1(inp)
    elif settings['model_selection']  == 'resnet_version_1':
        return cnn_v2(inp)
    elif settings['model_selection']  == 'complex_cnn_v1':
        return complex_cnn_v1(inp, settings)
    elif settings['model_selection']  == 'complex_cnn_v2':
        return complex_cnn_v2(inp, settings)
    elif settings['model_selection']  == 'complex_unet':
        return complex_unet(inp, settings)
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

    out = conv_layer(x, [1, 1, channel[-1], 1], 
                      [1,1,1,1], pad_pattern, dilations = [1,1,1,1], 
                      is_wn=False, name='c_out')
    out = tf.reshape(out, [-1, inp.get_shape().as_list()[1]])

    return out

def cnn_v2(inp):
    # inp shape = [batch, time]
    x = tf.reshape(inp, [-1, inp.get_shape().as_list()[1], 1, 1]) # [batch, time , 1, 1]
    # config
    block = 5
    kernel = 16
    stride = [1,1,1,1]
    pad_pattern = 'SAME'
    dilation = [1,1,1,1]
    channel = 16

    # c0
    x = conv_layer(x, [kernel,1,1,channel], 
                           stride, pad_pattern, dilations = dilation, 
                           name='c%d'%(0))

    for i in range(1, block+1):
        x = identity_block(x, [kernel,1,channel,channel], 
                           stride, pad_pattern, dilations = dilation, 
                           name='block%d'%(i))


    # output layer
    # make [batch, time, 1, channel] --> [batch, time, 1, 1] --> [batch, time]

    out = conv_layer(x, [1, 1, channel, 1], 
                      [1,1,1,1], pad_pattern, dilations = [1,1,1,1], 
                      is_wn=False, name='c_out')
    out = tf.reshape(out, [-1, inp.get_shape().as_list()[1]])
    if 1: # global skip
        out = tf.add(inp, out)
    return out

def complex_cnn_v1(inp, settings):
    # inp: [batch, time, freq(257)], complex
    inp_r, inp_i = tf.real(inp), tf.imag(inp)
    x_r = tf.reshape(inp_r, [-1, inp_r.get_shape().as_list()[1], 1, inp_r.get_shape().as_list()[2]])
    x_i = tf.reshape(inp_i, [-1, inp_i.get_shape().as_list()[1], 1, inp_i.get_shape().as_list()[2]])

    # setting
    kernel = 8
    stride = [1,1,1,1]
    pad_pattern = 'SAME'
    dilation = [1,1,1,1]
    channel = [128, 64, 128, 257]
    layer_num = len(channel)

    for i in range(layer_num):
        if i == 0:
            x_r, x_i = complex_conv_v2(x_r, x_i,
                                    [kernel,1,128,channel[i]],
                                    stride, pad_pattern, dilations = dilation, name='c%d'%(i))
        else:
            x_r, x_i = complex_conv_v2(x_r, x_i,
                                    [kernel,1,channel[i-1],channel[i]],
                                    stride, pad_pattern, dilations = dilation, name='c%d'%(i))
        x_r = activation(x_r)
        x_i = activation(x_i)

    ### MASK?
    x_r = tf.reshape(x_r, [-1, x_r.get_shape().as_list()[1], x_r.get_shape().as_list()[3]])
    x_i = tf.reshape(x_i, [-1, x_r.get_shape().as_list()[1], x_i.get_shape().as_list()[3]])
    if 0: # weiner
        s_r = x_r * inp_r - x_i * inp_i
        s_i = x_r * inp_i + x_i * inp_r 
    elif 1:
        s_r = x_r + inp_r
        s_i = x_i + inp_i

    estimated_stft = tf.complex(s_r, s_i)
    
    # inverse stft
    estimated_wav = tf.signal.inverse_stft(estimated_stft,
                                      frame_length = settings['window_size_samples'],
                                      frame_step = settings['window_stride_samples'],
                                      fft_length = 512)

    return estimated_wav


def complex_cnn_v2(inp, settings):
    # inp: [batch, time, freq(257)], complex
    inp_r, inp_i = tf.real(inp), tf.imag(inp)
    x_r = tf.reshape(inp_r, [-1, inp_r.get_shape().as_list()[1], inp_r.get_shape().as_list()[2], 1])
    x_i = tf.reshape(inp_i, [-1, inp_i.get_shape().as_list()[1], inp_i.get_shape().as_list()[2], 1])

    # setting
    kernel = [4, 8]
    stride = [1,1,1,1]
    pad_pattern = 'SAME'
    dilation = [1,1,1,1]
    channel = [8,8,16,16, 1]
    layer_num = len(channel)

    for i in range(layer_num):
        if i == 0:
            x_r, x_i = complex_conv_v2(x_r, x_i,
                                    [kernel[0], kernel[1], 1, channel[i]],
                                    stride, pad_pattern, dilations = dilation, name='c%d'%(i))
        else:
            x_r, x_i = complex_conv_v2(x_r, x_i,
                                    [kernel[0], kernel[1], channel[i-1], channel[i]],
                                    stride, pad_pattern, dilations = dilation, name='c%d'%(i))
        if i != layer_num - 1:
            x_r = activation(x_r)
            x_i = activation(x_i)
        else:
            # bound?
            x_am = tf.sqrt(tf.square(x_r) + tf.square(x_i))
            x_r = tf.tanh(x_am)*(x_r/x_am)
            x_i = tf.tanh(x_am)*(x_i/x_am)


    ### MASK?
    x_r = tf.reshape(x_r, [-1, x_r.get_shape().as_list()[1], x_r.get_shape().as_list()[2]])
    x_i = tf.reshape(x_i, [-1, x_r.get_shape().as_list()[1], x_i.get_shape().as_list()[2]])
    if 1: # weiner
        s_r = x_r * inp_r - x_i * inp_i
        s_i = x_r * inp_i + x_i * inp_r 

        #s_r = x_r * inp_r
        #s_i = x_i * inp_i

    elif 0:
        s_r = x_r + inp_r
        s_i = x_i + inp_i

    estimated_stft = tf.complex(s_r, s_i)
    
    # inverse stft
    estimated_wav = tf.signal.inverse_stft(estimated_stft,
                                      frame_length = settings['window_size_samples'],
                                      frame_step = settings['window_stride_samples'],
                                      fft_length = 512)

    return estimated_wav, estimated_stft


def complex_unet(inp, settings):
    # inp: [batch, time, freq(257)], complex
    inp_r, inp_i = tf.real(inp), tf.imag(inp)
    print('inp shape: {0}'.format(inp_r.shape))
    x_r = tf.reshape(inp_r, [-1, inp_r.get_shape().as_list()[1], inp_r.get_shape().as_list()[2], 1])
    x_i = tf.reshape(inp_i, [-1, inp_i.get_shape().as_list()[1], inp_i.get_shape().as_list()[2], 1])
    print('x shape: {0}'.format(x_r.shape))
    # setting
    kernel = [2, 4]
    stride = [1,1,1,1]
    pad_pattern = 'SAME'
    dilation = [1,1,1,1]
    channel = [1, 16, 32, 64]
    #Encoder_num = 2
    ################### Encoder ###################
    e1_r, e1_i = complex_conv_v1(x_r, x_i,
                                    [kernel[0], kernel[1], channel[0], channel[1]],
                                    [1,1,2,1], pad_pattern, dilations = dilation, name='e1')
    print('e1 shape: {0}'.format(e1_r.shape))
    e1_r = activation(e1_r)
    e1_i = activation(e1_i)



    e2_r, e2_i = complex_conv_v1(e1_r, e1_i,
                                    [kernel[0], kernel[1], channel[1], channel[2]],
                                    [1,1,2,1], pad_pattern, dilations = dilation, name='e2')
    print('e2 shape: {0}'.format(e2_r.shape))
    e2_r = activation(e2_r)
    e2_i = activation(e2_i)



    e3_r, e3_i = complex_conv_v1(e2_r, e2_i,
                                    [kernel[0], kernel[1], channel[2], channel[3]],
                                    [1,1,2,1], pad_pattern, dilations = dilation, name='e3')
    print('e3 shape: {0}'.format(e3_r.shape))
    e3_r = activation(e3_r)
    e3_i = activation(e3_i)

    e4_r, e4_i = complex_conv_v1(e3_r, e3_i,
                                    [kernel[0], kernel[1], channel[3], channel[2]],
                                    [1,1,1,1], pad_pattern, dilations = dilation, name='e4')
    print('e4 shape: {0}'.format(e4_r.shape))
    e4_r = activation(e4_r)
    e4_i = activation(e4_i)
    ################### Decoder ###################
    # dconv
    d2_r, d2_i = complex_dconv(e4_r, e4_i, 
                               kernel_size=[kernel[0], kernel[1], channel[2], channel[2]],
                               stride = [1,1,2,1],
                               name = 'd2')
    print('d2 shape: {0}'.format(d2_r.shape))

    c2_r = tf.concat([d2_r, e2_r], axis=-1)
    c2_i = tf.concat([d2_i, e2_i], axis=-1) # channel = 8 + 8
    c2_r, c2_i = complex_conv_v1(c2_r, c2_i,
                                    [2, 4, channel[2]*2, channel[1]],
                                    [1,1,1,1], pad_pattern, dilations = dilation, name='d2c')
    c2_r = activation(c2_r)
    c2_i = activation(c2_i)
    print('c2 shape: {0}'.format(c2_r.shape))
    # 
    d1_r, d1_i = complex_dconv(c2_r, c2_i, 
                               kernel_size=[kernel[0], kernel[1], channel[1], channel[1]],
                               stride = [1,1,2,1],
                               name = 'd1')
    print('d1 shape: {0}'.format(d1_r.shape))
    c1_r = tf.concat([d1_r, e1_r], axis=-1)
    c1_i = tf.concat([d1_i, e1_i], axis=-1)  # channel = 4 + 4
    c1_r, c1_i = complex_conv_v1(c1_r, c1_i,
                                    [2, 4, channel[1]*2, channel[0]],
                                    [1,1,1,1], pad_pattern, dilations = dilation, name='d1c')
    c1_r = activation(c1_r)
    c1_i = activation(c1_i)
    print('c1 shape: {0}'.format(c1_r.shape))
    #
    d0_r, d0_i = complex_dconv(c1_r, c1_i,
                               kernel_size=[kernel[0], kernel[1], channel[0], channel[0]],
                               stride = [1,1,2,1],
                               name = 'd0')
    print('d0 shape: {0}'.format(d0_r.shape))
    c0_r = tf.concat([d0_r, x_r], axis=-1)
    c0_i = tf.concat([d0_i, x_i], axis=-1)  # channel = 1 + 1

    c0_r, c0_i = complex_conv_v1(c0_r, c0_i,
                                    [2, 4, channel[0]*2, channel[0]],
                                    [1,1,1,1], pad_pattern, dilations = dilation, name='d0c')
    c0_r = activation(c0_r)
    c0_i = activation(c0_i)
    print('c0 shape: {0}'.format(c0_r.shape))


    ###############################################
    x_r , x_i = c0_r, c0_i
    x_r = tf.reshape(x_r, [-1, x_r.get_shape().as_list()[1], x_r.get_shape().as_list()[2]])
    x_i = tf.reshape(x_i, [-1, x_i.get_shape().as_list()[1], x_i.get_shape().as_list()[2]])
    if 0: # weiner
        if 0: # DCCRN-E
            x_am = tf.sqrt(tf.square(x_r) + tf.square(x_i))
            x_am = tf.clip_by_value(x_am, 0.00001, 255.)
            M_mag = tf.tanh(x_am)
            M_phase_r = x_r/x_am
            M_phase_i = x_i/x_am

            M_r = M_mag * M_phase_r
            M_i = M_mag * M_phase_i
    
            s_r = M_r * inp_r - M_i * inp_i
            s_i = M_r * inp_i + M_i * inp_r
        elif 1: # DCCRN-C:
            s_r = x_r * inp_r - x_i * inp_i
            s_i = x_r * inp_i + x_i * inp_r
        elif 0: # DCCRN-R:
            s_r = x_r * inp_r
            s_i = x_i * inp_i

    elif 1: # spec-subtraction
        s_r = x_r + inp_r
        s_i = x_i + inp_i

    estimated_stft = tf.complex(s_r, s_i)
    
    # inverse stft
    estimated_wav = tf.signal.inverse_stft(estimated_stft,
                                      frame_length = settings['window_size_samples'],
                                      frame_step = settings['window_stride_samples'],
                                      fft_length = 512)

    return estimated_wav, estimated_stft