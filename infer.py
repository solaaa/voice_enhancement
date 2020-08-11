
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import numpy as np
import os
import tensorflow as tf
import scipy.io as sio
import model
from pathlib import Path
import datetime
import frontend

sess = tf.compat.v1.InteractiveSession()
# training config
settings = {}
settings['model_selection']  = 'cnn_version_1'
#settings['model_selection']  = 'resnet_version_1'

settings['time_len'] = 1024

# tf computing graph

noised_inp_ = tf.placeholder(dtype=tf.float32, shape=[None, settings['time_len']], name = 'model_input')
clear_ground_truth_ = tf.placeholder(dtype=tf.float32, shape=[None, settings['time_len']], name = 'ground_truth')
denoised_out_ = model.get_model(noised_inp_, settings['model_selection'])


# check_point
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=100)
#################################################
# load check point (load weights)
if 1:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    data_str = 'd2020-08-11_t1624'
    checkpoint_str = 'E:\\dnn_denoising\\traing_log\\'+settings['model_selection']+'\\'+data_str+'\\chech_point-19'
    #checkpoint_str = r'E:\dnn_denoising\traing_log'+'/'+settings['model_selection']+'/'+ data_str+'r\chech_point-19'
    print('### checkpoint_str: {0}'.format(checkpoint_str))
    saver.restore(sess, checkpoint_str)           
#################################################


####################################################################################################

path = r'E:\dnn_denoising\data\learning_m\12.mat'
data = sio.loadmat(path)

clean, noised = data['clean_s'], data['noise_s']
noised = noised.reshape([1, -1])
denoised = sess.run(denoised_out_, feed_dict={noised_inp_:noised})

clean = clean.reshape([-1])
noised = noised.reshape([-1])
denoised = denoised.reshape([-1])

import matplotlib.pyplot as p

p.figure(1)
p.subplot(4,1,1)
p.plot(clean, label='clean')
p.subplot(4,1,2)
p.plot(noised, label='noised')
p.subplot(4,1,3)
p.plot(denoised, label='denoised')
p.subplot(4,1,4)
p.plot(clean, label='clean')
p.plot(noised, label='noised')
p.plot(denoised, label='denoised')
p.legend()
p.show()