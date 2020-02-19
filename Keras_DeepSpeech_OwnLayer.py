# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:37:51 2020

@author: Ganyu Wang


This script is trying to implement the generator 



"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM, Softmax, TimeDistributed, Flatten, Masking
from keras.utils import to_categorical
from keras import backend as K
from keras.layers import Layer
import tensorflow as tf

from Data_generator_draft import DataGenerator

#%%    network parameter..... should be optimized.
BATCH_SIZE = 64
time_step_len = 431
window_len = 19
n_mfcc = 30
NUM_CHARACTERS = 28


#%%
Batch_X = np.random.rand(64, 431, 30)
# 0~26, 0 [space], 1~26, [a-z]
y_train1 = np.random.randint(low=0, high= NUM_CHARACTERS, size=(BATCH_SIZE, time_step_len))
# y_train1[0, 5:9] = 0   # mark some point to 0 or []
Batch_y = to_categorical(y_train1, num_classes=NUM_CHARACTERS)




#%% not good.  do not use this.

class OverlapTimeWindowLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(OverlapTimeWindowLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        
        super(OverlapTimeWindowLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        eye_filter = (np.eye(window_len * n_mfcc)
                           .reshape(window_len, n_mfcc, window_len * n_mfcc), tf.float64)
        #eye_filter = tf.constant(value=eye_filter)

        # Create overlapping windows
        #batch_x = K.conv1d(x=x, kernel=eye_filter, strides=1, padding='same')
        
        # Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
        #batch_x = K.reshape(batch_x, [BATCH_SIZE, -1, window_len, n_mfcc])

        # Create overlapping windows
        batch_x = tf.nn.conv1d(input=batch_x, filters=eye_filter, stride=1, padding='SAME')
        
        # Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
        batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels])

        return batch_x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



#%% CTC loss

def ctc_loss(y_true, y_pred):
    y_true = keras.backend.ctc_label_dense_to_sparse(y_true, time_step_len)
    return tf.nn.ctc_loss(y_true, y_pred, input_length=time_step_len, label_length=time_step_len)

#%% Keras DeepSpeech Model Draft

# network parameters
n_hidden = 1024
rate_dropout = 0.05

# build model 
model = Sequential()

#model.add(TimeDistributed(Flatten()))
model.add(OverlapTimeWindowLayer((time_step_len, window_len, n_mfcc)))
model.add(Masking(mask_value=0., input_shape=(time_step_len, window_len*n_mfcc)))
model.add(TimeDistributed(Dense(n_hidden, activation='relu', input_dim=(window_len* n_mfcc), )))
model.add(TimeDistributed(Dropout(rate_dropout)))
model.add(TimeDistributed(Dense(n_hidden, activation='relu', input_dim=(window_len* n_mfcc), )))
model.add(TimeDistributed(Dropout(rate_dropout)))
model.add(TimeDistributed(Dense(n_hidden, activation='relu', input_dim=(window_len* n_mfcc), )))
model.add(TimeDistributed(Dropout(rate_dropout)))
model.add(Bidirectional(LSTM(n_hidden, return_sequences=True)))
model.add(TimeDistributed(Dropout(rate_dropout)))
model.add(TimeDistributed(Dense(NUM_CHARACTERS)))
model.add(TimeDistributed(Softmax(axis=-1)))

#optimizer = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
#model.compile(optimizer='rmsprop', loss=ctc_loss)
model.compile(optimizer='rmsprop', loss='mean_squared_error')

model.fit(Batch_X, Batch_y)




