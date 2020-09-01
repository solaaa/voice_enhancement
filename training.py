
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import moving_average, SI_SNR
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
#settings['model_selection']  = 'cnn_version_1'
#settings['model_selection']  = 'resnet_version_1'
settings['model_selection']  = 'complex_unet'


settings['dataset_path'] = r'F:\DNS\DNS-Challenge\datasets'
settings['clean_path'] = os.path.join(settings['dataset_path'], 'clean')
settings['noise_path'] = os.path.join(settings['dataset_path'], 'noise')

settings['train_percentage'] = 0.9

settings['batch_size'] = 8
settings['epoch'] = 10
settings['step_per_epoch'] = 800
settings['step_val_per_epoch'] = 10

settings['init_learning_rate'] = 0.001
settings['learning_rate_decay'] = 0.5

settings['window_size_samples'] = 480
settings['window_stride_samples'] = 240
settings['snrs_train'] = [15,20,25]
settings['snrs_val'] = [15,20,25]

settings['use_reverb'] = False

Fs = 16000
clip_duration_ms = 31*1000
settings['desired_samples'] = int(Fs * clip_duration_ms / 1000)
settings['train_stft_time_len'] = 1+int((settings['desired_samples'] - settings['window_size_samples']) / settings['window_stride_samples'])

data_generator = frontend.DataGenerator_v2(settings)

# tf computing graph

noised_inp_stft_ = tf.placeholder(dtype=tf.complex64, shape=[None, settings['train_stft_time_len'], 128], name = 'model_input')

clean_wav_ = tf.placeholder(dtype=tf.float32, shape=[None, settings['desired_samples']-(settings['desired_samples']%settings['window_stride_samples'])], name = 'ground_truth')

clean_stft_ = tf.placeholder(dtype=tf.complex64, shape=[None, settings['train_stft_time_len'], 128], name = 'clean_stft')

estimated_wav_, estimated_stft_= model.get_model(noised_inp_stft_, settings)

learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name = 'lr')

# loss
if 0:
    #loss = tf.losses.mean_squared_error(clear_ground_truth, denoised_out)
    loss_r = tf.reduce_mean(tf.reduce_sum(tf.square(tf.real(estimated_stft_ - clean_stft_)), axis=[1,2]),axis=0)
    loss_i = tf.reduce_mean(tf.reduce_sum(tf.square(tf.imag(estimated_stft_ - clean_stft_)), axis=[1,2]),axis=0)
    loss_ = (loss_r + loss_i)/100.
elif 0:
    #loss = tf.losses.absolute_difference(clear_ground_truth, denoised_out)
    loss_ = tf.reduce_mean(tf.reduce_sum(tf.abs(estimated_stft_ - clean_stft_), axis=[1,2]),axis=0)
elif 0:
    loss_ = tf.reduce_sum(tf.abs(estimated_wav_),axis=1)/tf.reduce_sum(tf.abs(estimated_wav_-clean_wav_),axis=1)
    loss_ = 20*tf.log(loss_)/tf.log(10.)
    loss_ = -1. * tf.reduce_mean(loss_,axis = 0)
elif 1:
    # SI-SNR loss
    #loss_ = SI_SNR(clean_wav_, estimated_wav_) + 20
    dot_product_ = tf.reshape(tf.reduce_sum(clean_wav_*estimated_wav_, axis=1), [-1,1])
    s_L2_ = tf.reshape(tf.reduce_sum(tf.square(clean_wav_),axis=1), [-1,1])

    s_target = dot_product_*clean_wav_/s_L2_
    e_noise = estimated_wav_-clean_wav_

    s_target_L2_ = tf.reduce_sum(tf.square(s_target),axis=1)
    e_noise_L2_ = tf.reduce_sum(tf.square(e_noise),axis=1)
    cost_ = s_target_L2_/e_noise_L2_
    #loss_ = 10.*tf.log(loss_)/tf.log(10.)
    cost_ = 10.*tf.log(cost_)/tf.log(10.)
    loss_ = -1. * tf.reduce_mean(cost_, axis=0) # the bigger SNR is, the better res is, so to minimize -1 * loss

    


if 1:
    train_step =tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_)
elif 0:
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.99).minimize(loss_)
elif 0:
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
        info_line = '{0} = {1}\n'.format(key, settings[key])
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
loss_ema = moving_average(init_avg = 0.009)
loss_hist = []
val_loss_hist = []
for e in range(settings['epoch']):
    cur_lr = settings['init_learning_rate'] * settings['learning_rate_decay']**e
    loss_sum = 0
    for step in range(settings['step_per_epoch']):
        c_stft, n_stft, c_wav, _ = data_generator.get_batch_data(settings['batch_size'],
                                                         clean_percentage=0.,
                                                         is_noise=True,is_reverb=settings['use_reverb'],
                                                         mode='train', sess=sess)

        loss, _= sess.run([loss_, train_step], 
                       feed_dict={
                           noised_inp_stft_: n_stft,
                           clean_wav_: c_wav,
                           #clean_stft_: c_stft,
                           learning_rate: cur_lr})
        if loss is np.nan:
            raise Exception('')
        #print('mr: max: {0}  min: {1}'.format(np.max(mr), np.min(mr)))
        #print('mi: max: {0}  min: {1}'.format(np.max(mi), np.min(mi)))
        #print('e_noise_L2: {0}'.format(e_noise_L2.reshape([-1,])))

        loss_sum += loss
        loss_ema.update(loss)
        if step == settings['step_per_epoch']-1:
            print('epoch: {0}/{1}, step: {2}/{3}, lr:{4:.6f}, loss: {5:.4f}({6:.4f}) '.format(e,settings['epoch'],
                                                                    step, settings['step_per_epoch'],
                                                                    cur_lr,
                                                                    loss, loss_ema.accumulate), end='\n')
        else:
            print('epoch: {0}/{1}, step: {2}/{3}, lr:{4:.6f}, loss: {5:.4f}({6:.4f}) '.format(e,settings['epoch'],
                                                                    step, settings['step_per_epoch'],
                                                                    cur_lr,
                                                                    loss, loss_ema.accumulate), end='\n')
    # val per epoch
    print('*'*60)
    print('* start to val ...')
    val_loss_sum = 0
    for step in range(settings['step_val_per_epoch']):
        c_stft, n_stft, c_wav, _ = data_generator.get_batch_data(settings['batch_size'],
                                                         clean_percentage=0.2,
                                                         is_noise=True,is_reverb=settings['use_reverb'],
                                                         mode='val', sess=sess)
        loss = sess.run(loss_, 
                       feed_dict={
                           noised_inp_stft_: n_stft,
                           #clean_stft_: c_stft,
                           clean_wav_: c_wav,})
        val_loss_sum += loss
    print('* epoch: {0}/{1}, loss: {2:.4f}, val_loss: {3:.4f} '.format(e, settings['epoch'],
                                                                    loss_sum/settings['step_per_epoch'],
                                                                    val_loss_sum/settings['step_val_per_epoch']))
    print('*'*60)
    val_loss_hist.append(val_loss_sum/settings['step_val_per_epoch'])
    loss_hist.append(loss_sum/settings['step_per_epoch'])
    saver.save(sess, os.path.join(save_path, 'chech_point'), global_step=e)


import matplotlib.pyplot as p
p.figure(1)
p.plot(loss_hist, 'b')
p.plot(val_loss_hist, 'r')
p.show()
    


