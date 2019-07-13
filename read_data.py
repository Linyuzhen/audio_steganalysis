import os
import random
import numpy as np
import scipy.io.wavfile as wave

def read_datasets(filepath):
    data = []
    filenames = os.listdir(filepath)
    filenames.sort(key=lambda x: int(x[:-4]))
    for fn in filenames:
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
    return np.hstack((label_n,label_p))

def shuffle(a,b):
    rand_seed = random.randint(0,100)
    random.seed(rand_seed)
    random.shuffle(a)
    random.seed(rand_seed)
    random.shuffle(b)
    return a,b

def data_split(Xc,Xs,train_size=600,vaild_size=200):
    shuffle(Xc,Xs)

    Xc_train = Xc[0:train_size,:]
    Xc_vaild = Xc[train_size:train_size+vaild_size,:]
    Xc_test = Xc[train_size+vaild_size:len(Xc),:]

    Xs_train = Xs[0:train_size,:]
    Xs_vaild = Xs[train_size:train_size + vaild_size,:]
    Xs_test = Xs[train_size + vaild_size:len(Xs),:]

    train_data = np.concatenate((Xc_train,Xs_train))
    vaild_data = np.concatenate((Xc_vaild,Xs_vaild))
    test_data = np.concatenate((Xc_test,Xs_test))
    return train_data,vaild_data,test_data


def pair_batch_generator(x,y,batch_size, categorical=True):
    num_samples = y.shape[0]
    num_classes = y.shape[1]
    batch_xpair_shape = (batch_size//2, *x.shape[1:])
    batch_ypair_shape = (batch_size//2, num_classes) if categorical else (batch_size//2, )
    indexes = [0 for _ in range(num_classes)]
    samples = [[] for _ in range(num_classes)]
    for i in range(num_samples):
        samples[np.argmax(y[i])].append(x[i])
    while True:
        batch_x1 = np.ndarray(shape=batch_xpair_shape, dtype=x.dtype)
        batch_x2 = np.ndarray(shape=batch_xpair_shape, dtype=x.dtype)
        batch_y1 = np.zeros(shape=batch_ypair_shape, dtype=y.dtype)
        batch_y2 = np.zeros(shape=batch_ypair_shape, dtype=y.dtype)
        for i in range(batch_size//2):
            random_class = random.randrange(num_classes)
            current_index = indexes[random_class]
            indexes[random_class] = (current_index + 1) % len(samples[random_class])
            if current_index == 0:
                shuffle(samples[random_class],samples[1-random_class])
            batch_x1[i] = samples[random_class][current_index]
            batch_x2[i] = samples[1-random_class][current_index]
            if categorical:
                batch_y1[i][random_class] = 1
                batch_y2[i][1-random_class] = 1
            else:
                batch_y1[i] = random_class
                batch_y2[i] = 1-random_class
        batch_x = np.concatenate((batch_x1,batch_x2))
        batch_y = np.concatenate((batch_y1,batch_y2))
        yield (batch_x, batch_y)
