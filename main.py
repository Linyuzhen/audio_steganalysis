import math
import matplotlib.pyplot as plt
from tensorflow import keras
# from sklearn.model_selection import train_test_split
from read_data import read_datasets,get_label,data_split,pair_batch_generator
from Lin_models import Lin_Net

# Load data&label
cover_dir = r'.\cover_dir'
stego_dir = r'.\stego_dir'
X_c = read_datasets(cover_dir)
X_s = read_datasets(stego_dir)

# Data preprocessing
X_train,X_val,X_test = data_split(X_c,X_s)
y_train = get_label(X_train.shape[0],positive=0.5)
y_val = get_label(X_val.shape[0],positive=0.5)
y_test = get_label(X_test.shape[0],positive=0.5)

X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_val = X_val.reshape(-1, X_val.shape[1], 1)
X_test = X_test.reshape(-1, X_test.shape[1], 1)

y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_val = keras.utils.to_categorical(y_val, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)

# Build model
model = Lin_Net(X_train)

# Before fine-tuning the network, you should change the names of the HPF_layer and Dense(2) layers.
model.load_weights('model_weights.h5',by_name=True)
# model = keras.utils.multi_gpu_model(model,gpus=4)

adam = keras.optimizers.Adam(1e-4)
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training model
batch_size = 64
epochs = 10
steps = math.ceil(len(X_train)//batch_size)
train_batch = pair_batch_generator(X_train, y_train, batch_size=batch_size)

early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, mode='auto', verbose=0)
lr_reduce=keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,mode='auto',verbose=0)
print('Training--------------------------------')
history = model.fit_generator(train_batch,steps_per_epoch=steps,epochs=epochs,
                    validation_data=(X_val,y_val),verbose=1,callbacks=[lr_reduce, early_stopping])

# Testing model
print("\nTesting------------------------------------")
loss,accuracy=model.evaluate(X_test,y_test)

model.save_weights('model_weights.h5')

# Figure acc
history_dict = history.history
acc=history_dict['acc']
val_acc=history_dict['val_acc']
epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'r',linewidth=5.0,label='Training acc')
plt.plot(epochs,val_acc,'b',linewidth=5.0,label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Figure loss
loss_value=history_dict['loss']
val_loss_value=history_dict['val_loss']

plt.plot(epochs,loss_value,'r',linewidth=5.0,label='Training loss')
plt.plot(epochs,val_loss_value,'b',linewidth=5.0,label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
