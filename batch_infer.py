

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
import audio_lib as audio

sess = tf.compat.v1.InteractiveSession()
# training config
settings = {}
settings['model_selection']  = 'complex_unet'
#settings['model_selection']  = 'resnet_version_1'
settings['window_size_samples'] = 480
settings['window_stride_samples'] = 240
MAX_WAV_LEN = 16000*20

# data config
noised_path = r'E:\dnn_denoising\data\test_data\synthetic\noise'
save_path = r'E:\dnn_denoising\data\test_data\synthetic\denoise'

file_list = os.listdir(noised_path)

# tf computing graph

noised_wav_ph = tf.placeholder(dtype=tf.float32, shape=[1, MAX_WAV_LEN])
noised_inp_stft_, _  = frontend.get_stft(noised_wav_ph, 
                                settings['window_size_samples'], 
                                settings['window_stride_samples'])
noised_inp_stft_ = noised_inp_stft_[:, :, :128]
print('noised_inp_stft_ shape: {0}'.format(noised_inp_stft_.shape))
#clean_wav_ = tf.placeholder(dtype=tf.float32, shape=[None, None], name = 'ground_truth')
estimated_wav_, _ = model.get_model(noised_inp_stft_, settings)


# check_point
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=100)
#################################################
# load check point (load weights)
if 1:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    data_str = 'd2020-08-31_t1814'
    checkpoint_str = 'E:\\dnn_denoising\\traing_log\\'+settings['model_selection']+'\\'+data_str+'\\chech_point-3'
    #checkpoint_str = r'E:\dnn_denoising\traing_log'+'/'+settings['model_selection']+'/'+ data_str+'r\chech_point-19'
    print('### checkpoint_str: {0}'.format(checkpoint_str))
    saver.restore(sess, checkpoint_str)           
#################################################


####################################################################################################

for i, file_name in enumerate(file_list):
    # get STFT
    wav_q15, par = audio.open_wave(os.path.join(noised_path, file_name))
    wav_flt = wav_q15*2**(-15)
    wav_len = len(wav_flt)
    if wav_len < MAX_WAV_LEN:
        wav_flt = np.concatenate([wav_flt, np.zeros([MAX_WAV_LEN-wav_len])], axis=0)
    else:
        wav_flt = wav_flt[:MAX_WAV_LEN]
    wav_flt = wav_flt.reshape([1,-1])

    estimated_wav_flt = sess.run(estimated_wav_, 
                             feed_dict={noised_wav_ph: wav_flt})
    print('estimated_wav_flt shape: {0}'.format(estimated_wav_flt.shape))
    estimated_wav_q15 = np.round(estimated_wav_flt*2**15)
    estimated_wav_q15 = estimated_wav_q15.astype(np.int16)
    estimated_wav_q15 = estimated_wav_q15.reshape([-1,])
    if wav_len < MAX_WAV_LEN:
        estimated_wav_q15 = estimated_wav_q15[:wav_len]
    print(len(estimated_wav_q15))
    audio.save_wave(estimated_wav_q15, par,
                    os.path.join(save_path, file_name))

