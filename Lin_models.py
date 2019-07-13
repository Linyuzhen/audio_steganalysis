import numpy as np
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

def TLU(x,th=3.0):
    return keras.backend.minimum(keras.backend.maximum(x,-th),th)


def Lin_Net(X):
    inputs = keras.layers.Input(shape=(X.shape[1],1))

    # HPF
    hpf = HPF_Layer(filters=4,kernel_size=5,hpf_kernel='SRM_k.npy',is_train=True,name='HPF_layer')(inputs)

    # Group 1
    gp1 = keras.layers.Conv1D(8, 1, padding='same',activation=TLU,name='Group1_conv1')(hpf)
    gp1 = keras.layers.Conv1D(8, 5, padding='same',name='Group1_conv2')(gp1)
    gp1 = keras.layers.Conv1D(16, 1, padding='same',name='Group1_output')(gp1)

    # Group 2
    gp2 = keras.layers.Conv1D(16, 5, padding='same',activation='relu',name='Group2_conv1')(gp1)
    gp2 = keras.layers.Conv1D(32, 1, padding='same',activation='relu',name='Group2_conv2')(gp2)
    gp2 = keras.layers.AveragePooling1D(pool_size=3,strides=2,padding='same',name='Group2_output')(gp2)

    # Group 3
    gp3 = keras.layers.Conv1D(32, 5, padding='same', activation='relu',name='Group3_conv1')(gp2)
    gp3 = keras.layers.Conv1D(64, 1, padding='same', activation='relu',name='Group3_conv2')(gp3)
    gp3 = keras.layers.AveragePooling1D(pool_size=3, strides=2, padding='same',name='Group3_output')(gp3)

    # Group 4
    gp4 = keras.layers.Conv1D(64, 5, padding='same', activation='relu',name='Group4_conv1')(gp3)
    gp4 = keras.layers.Conv1D(128, 1, padding='same', activation='relu',name='Group4_conv2')(gp4)
    gp4 = keras.layers.AveragePooling1D(pool_size=3, strides=2, padding='same',name='Group4_output')(gp4)

    # Group 5
    gp5 = keras.layers.Conv1D(128, 5, padding='same', activation='relu',name='Group5_conv1')(gp4)
    gp5 = keras.layers.Conv1D(256, 1, padding='same', activation='relu',name='Group5_conv2')(gp5)
    gp5 = keras.layers.AveragePooling1D(pool_size=3, strides=2, padding='same',name='Group5_output')(gp5)

    # Group 6
    gp6 = keras.layers.Conv1D(256, 5, padding='same', activation='relu',name='Group6_conv1')(gp5)
    gp6 = keras.layers.Conv1D(512, 1, padding='same', activation='relu',name='Group6_conv2')(gp6)
    gp6 = keras.layers.GlobalAveragePooling1D(name='Group6_output')(gp6)

    # Classifier
    prob = keras.layers.Dense(2,activation='softmax',name='Classifier')(gp6)

    # Build model
    model = keras.models.Model(inputs,prob)
    model.summary()

    return model















