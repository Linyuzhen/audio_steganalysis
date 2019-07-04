import numpy as np
import tensorflow as tf
from tensorflow import keras

class HPF_Layer(keras.layers.Layer):
    def __init__(self,filters,kernel_size,hpf_kernel,is_train=True,**kwargs):
        super(HPF_Layer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.hpf_kernel = hpf_kernel
        self.is_train = is_train

    def build(self, input_shape):
        k = np.load(self.hpf_kernel)
        k = k.reshape([self.filters,self.kernel_size,1])
        # print(k)
        self.hpf = keras.layers.Conv1D(self.filters,self.kernel_size,
                                  kernel_initializer=keras.initializers.constant(k),
                                  trainable=self.is_train,
                                  padding='same')
        super(HPF_Layer, self).build(input_shape)


    def call(self, inputs, **kwargs):
        return self.hpf(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


def Lin_Net(X):
    inputs = keras.layers.Input(shape=(X.shape[1],1))

    # HPF
    hpf = HPF_Layer(filters=4,kernel_size=5,hpf_kernel='SRM_k.npy',is_train=True)(inputs)

    # group 1
    x = keras.layers.Conv1D(8, 1, padding='same')(hpf)
    x = keras.layers.ThresholdedReLU(theta=3.0)(x)
    x = keras.layers.Conv1D(8, 5, padding='same')(x)
    x = keras.layers.Conv1D(16, 1, padding='same')(x)

    # group 2
    x = keras.layers.Conv1D(16, 5, padding='same',activation='relu')(x)
    x = keras.layers.Conv1D(32, 1, padding='same',activation='relu')(x)
    x = keras.layers.AveragePooling1D(pool_size=3,strides=2,padding='same')(x)

    # group 3
    x = keras.layers.Conv1D(32, 5, padding='same', activation='relu')(x)
    x = keras.layers.Conv1D(64, 1, padding='same', activation='relu')(x)
    x = keras.layers.AveragePooling1D(pool_size=3, strides=2, padding='same')(x)

    # group 4
    x = keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = keras.layers.Conv1D(128, 1, padding='same', activation='relu')(x)
    x = keras.layers.AveragePooling1D(pool_size=3, strides=2, padding='same')(x)

    # group 5
    x = keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = keras.layers.Conv1D(256, 1, padding='same', activation='relu')(x)
    x = keras.layers.AveragePooling1D(pool_size=3, strides=2, padding='same')(x)

    # group 4
    x = keras.layers.Conv1D(256, 5, padding='same', activation='relu')(x)
    x = keras.layers.Conv1D(512, 1, padding='same', activation='relu')(x)
    x = keras.layers.GlobalAveragePooling1D()(x)

    # classifier
    prob = keras.layers.Dense(2,activation='softmax')(x)

    # Build model
    model = keras.models.Model(inputs,prob)
    model.summary()

    return model









