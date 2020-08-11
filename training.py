
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


settings['dataset_path'] = r'E:\dnn_denoising\data\learning_m'
settings['batch_size'] = 16
settings['epoch'] = 20
settings['step_per_epoch'] = 500
settings['init_learning_rate'] = 0.01
settings['learning_rate_decay'] = 0.9
settings['time_len'] = 1024
train_generator = frontend.DataGenerator(settings)

# tf computing graph

noised_inp_ = tf.placeholder(dtype=tf.float32, shape=[None, settings['time_len']], name = 'model_input')
clear_ground_truth_ = tf.placeholder(dtype=tf.float32, shape=[None, settings['time_len']], name = 'ground_truth')
denoised_out_ = model.get_model(noised_inp_, settings['model_selection'])

learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name = 'lr')

# loss
if 1:
    #loss = tf.losses.mean_squared_error(clear_ground_truth, denoised_out)
    loss_ = tf.reduce_mean(tf.square(clear_ground_truth_ - denoised_out_), axis=[0,1])
elif 0:
    #loss = tf.losses.absolute_difference(clear_ground_truth, denoised_out)
    loss_ = tf.reduce_mean(tf.abs(clear_ground_truth_ - denoised_out_), axis=[0,1])


if 0:
    train_step =tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_)
elif 0:
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.99).minimize(loss_)
elif 1:
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_)

# log and chekpoint
train_log_path = r'.\traing_log'
if (Path(train_log_path).exists()==False):
    os.mkdir(train_log_path)
if (Path(os.path.join(train_log_path, settings['model_selection'])).exists()==False):
    os.mkdir(os.path.join(train_log_path, settings['model_selection']))
timenow=datetime.datetime.now()
date_str="d%s_t%02d%02d" %(str(timenow.date()), timenow.hour, timenow.minute)
save_path = os.path.join(train_log_path, settings['model_selection'],date_str)
if (Path(save_path).exists()==False):
    os.mkdir(save_path)
if (Path(os.path.join(save_path, 'chech_point')).exists()==False):
    os.mkdir(os.path.join(save_path, 'chech_point'))

# save settings
with open(os.path.join(save_path, 'training_info.txt'), 'w') as f:
    setting_keys = list(settings.keys())
    info = ''
    for key in setting_keys:
        info_line = '{0} = {1}/n'.format(key, settings[key])
        info += info_line
    f.write(info)

# check_point
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=100)
#################################################
# load check point (load weights)
if 0:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())    
    saver.restore(sess, r'')           
#################################################

sess.run(tf.initialize_all_variables())
##################################################
# start training
##################################################
loss_hist = []
for e in range(settings['epoch']):
    cur_lr = settings['init_learning_rate'] * settings['learning_rate_decay']**e
    loss_sum = 0
    for step in range(settings['step_per_epoch']):
        x_train, y_train = train_generator.get_batch_data(settings['batch_size'])
        loss, _ = sess.run([loss_, train_step], 
                       feed_dict={
                           noised_inp_: x_train,
                           clear_ground_truth_: y_train,
                           learning_rate: cur_lr})
        loss_sum += loss
        print('epoch: {0}/{1}, step: {2}/{3}, loss: {4:.8f}'.format(e,settings['epoch'],
                                                                    step, settings['step_per_epoch'],
                                                                    loss))
    loss_hist.append(loss_sum/settings['step_per_epoch'])
    saver.save(sess, os.path.join(save_path, 'chech_point'), global_step=e)


import matplotlib.pyplot as p
p.figure(1)
p.plot(loss_hist)
p.show()
    


