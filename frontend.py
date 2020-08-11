import numpy as np
import scipy.io as sio
import os
import random

class DataGenerator:
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
            x, y = data['x'], data['y'] # x for noised, y for clear
            if i == 0:
                x_train = x.reshape([1, -1])
                y_train = y.reshape([1, -1])
            else:
                x_train = np.concatenate([x_train, x], axis=0)
                y_train = np.concatenate([y_train, y], axis=0)
        return x_train, y_train


        
