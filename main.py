import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from read_data import read_datasets,get_label
from Lin_models import Lin_Net
# Load data&label
cover_dir = r'E:\林昱臻\实验\audioCNN\Libs\TIMITcut'
stego_dir = r'E:\林昱臻\实验\audioCNN\Libs\TIMIT_1'
X = np.vstack((read_datasets(cover_dir),read_datasets(stego_dir)))
y = get_label(X.shape[0],positive=0.5)

# Data preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)

X_train = X_train.reshape(-1, X.shape[1], 1)
X_test = X_test.reshape(-1, X.shape[1], 1)
y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)

model = Lin_Net(X)
model = keras.utils.multi_gpu_model(model,gpus=4)

adam = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])

print('Training--------------------------------')
# checkpoint=keras.callbacks.ModelCheckpoint('./Checkpoint/best.h5', monitor='val_loss', save_best_only=True, mode='auto',period=5)
early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, mode='auto', verbose=0)
lr_reduce=keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,mode='auto',verbose=0)
history = model.fit(X_train,y_train,epochs=500,verbose=1,validation_split=0.25,
                    callbacks=[lr_reduce, early_stopping])
# history = model.fit(X_train,y_train,batch_size=64,epochs=20,validation_data=(X_test,y_test))

history_dict = history.history

# test model
print("\nTesting------------------------------------")
loss,accuracy=model.evaluate(X_test,y_test)

print('test loss:',loss)
print('test accuracy:',accuracy)

# model.save_weights('ROC/STC_2/Chen.h5')
pd.DataFrame(history_dict).to_csv('save_data/HPF_fix.csv')

# figure acc
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

# figure loss
loss_value=history_dict['loss']
val_loss_value=history_dict['val_loss']

plt.plot(epochs,loss_value,'r',linewidth=5.0,label='Training loss')
plt.plot(epochs,val_loss_value,'b',linewidth=5.0,label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()







