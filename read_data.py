import os
import numpy as np
# np.random.seed(1337)
import scipy.io.wavfile as wave


def read_datasets(filepath):
    data = []
    for fn in os.listdir(filepath):
        if fn.endswith('.wav'):
            fd1 = os.path.join(filepath, fn)
            r1, x1 = wave.read(fd1)
            data.append(x1)
    X = np.array(data)
    return X

def read_label(filename):
    y = np.loadtxt(filename)
    return y

def get_label(shape,positive=0.5):
    label_p =  np.ones(int(shape*positive))
    label_n = np.zeros(int(shape*(1-positive)))
    return np.hstack((label_p,label_n))
