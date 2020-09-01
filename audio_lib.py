
import wave
import numpy as np


import os
import matplotlib.pyplot as p
import random
import time
import scipy.io as sio
from pathlib import Path
import os

import numpy as np
def open_wave(path):
    with wave.open(path, 'rb') as f:
        param = f.getparams()
        nchannels, sampwidth, framerate, nframes = param[:4]
        str_data = f.readframes(nframes)
        wave_data = np.fromstring(str_data, dtype=np.short)
    return wave_data, {'channel': nchannels, 'width':sampwidth, 'sample_rate':framerate}

def save_wave(data, param, path):
        with wave.open(path, 'wb') as w:
            w.setnchannels(param['channel'])
            w.setsampwidth(2) # 16bits?
            w.setframerate(param['sample_rate'])
            w.writeframes(b''.join(data))

def vedio_to_wav(src, dst, sample_rate=16000):
    ffmpeg = r'D:\ffmpeg\ffmpeg-20191028-68f623d-win64-static\bin\ffmpeg.exe'
    cmd = ffmpeg + " -i " + src + " -ar 16000 -vn " + dst
    res = os.system(cmd)
    time.sleep(0.01)




