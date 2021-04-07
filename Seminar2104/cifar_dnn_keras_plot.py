"""
OK
Python 3.7
"""

from keras.datasets import cifar10
from matplotlib import pyplot as plt
from keras.utils import np_utils
from tensorflow import keras

# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
 
# csv을 위한 라이브러리
import time
import csv


batch_size = 100
num_classes = 10
epochs = 50
trainingDataSize = 50000
batch_size = 100
display_step = 1
 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# One hot Encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train[0:trainingDataSize, :, :] ###############
y_train = y_train[0:trainingDataSize] ###############
 
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=x_train.shape[1:]),
#    keras.layers.Dense(10, activation='softmax')
#])
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=x_train.shape[1:]),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(10, activation='softmax')
#])
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=x_train.shape[1:]),
#    keras.layers.Dense(256, activation='relu'),
#    keras.layers.Dense(10, activation='softmax')
#])
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=x_train.shape[1:]),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(10, activation='softmax')
#])
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=x_train.shape[1:]),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(10, activation='softmax')
#])
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=x_train.shape[1:]),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(10, activation='softmax')
#])
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=x_train.shape[1:]),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(10, activation='softmax')
#])
model = keras.Sequential([
    keras.layers.Flatten(input_shape=x_train.shape[1:]),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=x_train.shape[1:]),
#    keras.layers.Dense(256, activation='relu'),
#    keras.layers.Dense(256, activation='relu'),
#    keras.layers.Dense(256, activation='relu'),
#    keras.layers.Dense(10, activation='softmax')
#])
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=x_train.shape[1:]),
#    keras.layers.Dense(256, activation='relu'),
#    keras.layers.Dense(256, activation='relu'),
#    keras.layers.Dense(256, activation='relu'),
#    keras.layers.Dense(256, activation='relu'),
#    keras.layers.Dense(10, activation='softmax')
#])
 
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
 

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)



write_enable = 1
if write_enable == 1:
    sheet_file = open('C:/Users/inslab/Desktop/Soomin/cifar_dnn_relu_adam.csv', 'w+', newline='')
    wr = csv.writer(sheet_file)
time_accumulate = []
for i in range(epochs):
    time_accumulate.append(0)
time_callback = TimeHistory()
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[time_callback])

for i in range(epochs):
    for j in range(0, i+1):
        time_accumulate[i] += time_callback.times[j]
print(time_callback.times)
print(time_accumulate)
with open('./result_test', 'wb') as f:
    for i in range(epochs):
        wr.writerow([i, time_accumulate[i], 100*hist.history['val_acc'][i], hist.history['loss'][i]])    
print('====================')


if write_enable == 1:
            sheet_file.close()