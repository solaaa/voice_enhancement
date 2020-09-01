import numpy as np
import scipy.io as sio
import os
import random
import audio_lib as audio
import random
import tensorflow as tf

def get_stft(wave_data_flt, window_len, window_stride, fft_len=512):
    stft_ = tf.contrib.signal.stft(wave_data_flt,
                                  frame_length = window_len,
                                  frame_step = window_stride,
                                  fft_length = fft_len)
    stft_Am_ = tf.abs(stft_)
    return stft_, stft_Am_

def get_istft(stft_, window_len, window_stride, fft_len=512):
    inverse_stft_ = tf.signal.inverse_stft(stft_,
                                      frame_length = window_len,
                                      frame_step = window_stride,
                                      fft_length = fft_len)
    return inverse_stft_


    

class DataGenerator_v1:
    # v1 is for time domain
    # now is deleted
    def __init__(self, settings):
        self.dataset_path = settings['dataset_path']
        

        #
        self.data_file_list = os.listdir(self.dataset_path)


    def get_batch_data(self, num):
        max_num = len(self.data_file_list)
        assert max_num >= num
        
        rd_idx = random.sample(range(len(self.data_file_list)), num)
        for i, idx in enumerate(rd_idx):
            path = os.path.join(self.dataset_path, self.data_file_list[idx])
            data = sio.loadmat(path)
            x, y = data['noise_s'], data['clean_s'] # x for noised, y for clear
            x = x.reshape([1, -1])
            y = y.reshape([1, -1])
            if i == 0:
                x_train = x
                y_train = y
            else:
                x_train = np.concatenate([x_train, x], axis=0)
                y_train = np.concatenate([y_train, y], axis=0)
        return x_train, y_train


class DataGenerator_v2:
    def __init__(self, settings):
        # clean_path is set of clean voice
        # noise_path is set of noise, not noised voice
        self.settings = settings
        self.clean_path = settings['clean_path']
        self.noise_path = settings['noise_path']

        self.clean_list = os.listdir(self.clean_path)
        self.noise_list = os.listdir(self.noise_path)

        clean_num = len(self.clean_list)
        noise_num = len(self.noise_list)

        ### small set test ###
        clean_num = int(clean_num*0.1)
        noise_num = int(noise_num*0.1)
        self.clean_list = self.clean_list[:clean_num]
        self.noise_list = self.noise_list[:noise_num]
        ######################

        self.clean_list_train = self.clean_list[:int(settings['train_percentage']*clean_num)]
        self.clean_list_val = self.clean_list[int(settings['train_percentage']*clean_num):]
        self.noise_list_train = self.noise_list[:int(settings['train_percentage']*noise_num)]
        self.noise_list_val = self.noise_list[int(settings['train_percentage']*noise_num):]

        self.snrs_train = settings['snrs_train']
        self.snrs_val = settings['snrs_val']

        # computing graph
        self.wav_ph = tf.placeholder(dtype=tf.float32, shape=[None, None])       # [batch, time]
        self.stft_ph = tf.placeholder(dtype=tf.complex64, shape=[None, None, 128]) # [batch, time, freq]
        self.stft_, self.stft_am_ = get_stft(self.wav_ph, 
                                             settings['window_size_samples'], 
                                             settings['window_stride_samples'])
        self.istft_ = get_istft(self.stft_ph,
                                settings['window_size_samples'], 
                                settings['window_stride_samples'])
    def reverb(self, wav):
        reverbed_wav = wav
        return reverbed_wav

    def add_noise(self, clean, noise, snr=10):
        # from DCCRN, test snr [0, 20], train snr [-5, 20] dB
        while len(noise) <= len(clean):
            noise = np.concatenate([noise, noise])
        if len(noise) > len(clean):
            rd_idx = np.random.randint(0, len(noise) - len(clean))
            noise = noise[rd_idx:rd_idx+len(clean)]

        A_clean = np.mean(np.abs(clean))
        A_noise = np.mean(np.abs(noise))

        noise = noise * A_clean / (A_noise*10**(snr/20))

        mixed = clean + noise
        return mixed

    def time_domain_process(self, wav_path, noise_path, is_noise, is_reverb, snr=10):
        clean_wav, _ = audio.open_wave(wav_path)
        if len(clean_wav) > self.settings['desired_samples']:
            rd = np.random.randint(0, len(clean_wav)-self.settings['desired_samples'] -1)
            clean_wav = clean_wav[rd : rd+self.settings['desired_samples']]
        wav = clean_wav
        if is_noise:
            noise, _ = audio.open_wave(noise_path)
            A_clean = np.mean(np.abs(wav))
            A_noise = np.mean(np.abs(noise))
            if A_clean < 1. or A_noise < 1:
                print(wav_path)
                print(noise_path)

        if is_reverb:
            wav = self.reverb(wav)
            noise = self.reverb(noise)
        if is_noise:
            noised_wav = self.add_noise(wav, noise, snr)
        else:
            noised_wav = wav
        return clean_wav, noised_wav

    def get_batch_data(self, batch_num, clean_percentage, is_noise, is_reverb, mode, sess):
        assert batch_num > 0
        clean_num = int(batch_num*clean_percentage)
        noise_num = batch_num - clean_num

        if mode=='train':
            clean_list = self.clean_list_train
            noise_list = self.noise_list_train
            snrs = self.snrs_train
        elif mode == 'val':
            clean_list = self.clean_list_val
            noise_list = self.noise_list_val
            snrs = self.snrs_val

        sampled_file_list = random.sample(clean_list, batch_num)
        for i, file_name in enumerate(sampled_file_list):

            snr = snrs[np.random.randint(len(snrs))]
            noise_name = random.sample(noise_list, 1)
            noise_name = noise_name[0]

            file_path = os.path.join(self.clean_path, file_name)
            noise_path = os.path.join(self.noise_path, noise_name)
            #print(file_name)
            #print(noise_name)
             
            try:
                clean_wav_q15, noised_wav_q15 = self.time_domain_process(file_path,
                                                                 noise_path,
                                                                 is_noise, is_reverb, snr)
            except:
                # wave.Error: file does not start with RIFF id
                print(' ')
                print('-*'*20)
                print('wave.Error: file does not start with RIFF id')
                print('filename: {0}'.format(file_name))
                print('-*'*20)
                resampled_file = random.sample(clean_list, 1)
                resampled_noise = random.sample(noise_list, 1)
                file_path = os.path.join(self.clean_path, resampled_file[0])
                noise_path = os.path.join(self.noise_path, resampled_noise[0])
                clean_wav_q15, noised_wav_q15 = self.time_domain_process(file_path,
                                                                 noise_path,
                                                                 is_noise, is_reverb, snr)

            if i < clean_num:
                noised_wav_q15 = clean_wav_q15

            clean_wav_flt = clean_wav_q15*2**(-15)
            noised_wav_flt = noised_wav_q15*2**(-15)
            clean_wav_flt = clean_wav_flt.reshape([1,-1])
            noised_wav_flt = noised_wav_flt.reshape([1,-1])
            #noised_stft = sess.run(self.stft_, feed_dict={self.wav_ph: noised_wav_flt})

            if i==0:
                batch_clean_wav = clean_wav_flt
                batch_noised_wav = noised_wav_flt
            else:
                batch_clean_wav = np.concatenate([batch_clean_wav, clean_wav_flt], axis=0)
                batch_noised_wav = np.concatenate([batch_noised_wav, noised_wav_flt], axis=0)

        batch_noised_stft = sess.run(self.stft_, feed_dict={self.wav_ph: batch_noised_wav})
        batch_clean_stft = sess.run(self.stft_, feed_dict={self.wav_ph: batch_clean_wav})
        # cut
        batch_noised_stft = batch_noised_stft[:, :, :128]
        batch_clean_stft = batch_clean_stft[:, :, :128]
        # clip end
        batch_clean_wav = batch_clean_wav[:,:-(self.settings['desired_samples']%self.settings['window_stride_samples'])]

        return batch_clean_stft, batch_noised_stft, batch_clean_wav, batch_noised_wav

    def get_batch_istft(self, batch_stft, sess):
        # batch_stft: [-1, none, 257]
        batch_wav = sess.run(self.istft_, feed_dict={self.stft_ph: batch_stft})
        return batch_wav
 
