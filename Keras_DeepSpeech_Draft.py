# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 20:00:39 2020

@author: Ganyu Wang

This script is a draft for building the model of DeepSpeech


"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM, Softmax, TimeDistributed, Flatten, Masking
from keras.utils import to_categorical


import tensorflow.compat.v1 as tf

import numpy as np

BATCH_SIZE = 2
time_step_len = 4
window_len = 3
n_mfcc = 4
NUM_CHARACTERS = 28

# 30 is n_mfcc, 200 is number of frames. 
x_train = np.random.rand(BATCH_SIZE, time_step_len , window_len, n_mfcc)\
                .reshape(BATCH_SIZE, time_step_len, window_len*n_mfcc) #flatten the input to (Batch, time_step, flatten Feature)
                
# 0~26, 0 [space], 1~26, [a-z]
y_train1 = np.random.randint(low=0, high= NUM_CHARACTERS, size=(BATCH_SIZE, time_step_len))

y_train = to_categorical(y_train1, num_classes=NUM_CHARACTERS)


#%% CTC loss

############
def ctc_loss(y_true, y_pred):
    print(y_true)
    print(y_pred)
    
    # 1 ,keras ctc_batch_cost, using y_train1 as input. 
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, np.array([[4], [4]]), np.array([[4], [4]]) )
    

    # try this 
    # 4    
    # np.array([4, 4])
    # Shape (?, ?) must have rank 1
    # np.array(4)
    # np.array([[4], [4]])



#%% Draft

# input parameters


# network parameters
n_hidden = 2048
rate_dropout = 0.05

# build model 
model = Sequential()

#model.add(TimeDistributed(Flatten()))

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


#optimizer = keras.optimizers.Adam( beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer='rmsprop', loss=ctc_loss)

#model.compile(optimizer='rmsprop', loss='mean_squared_error')


model.fit(x_train, y_train1)



