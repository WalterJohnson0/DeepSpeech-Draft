# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 19:39:27 2020

@author: wang_


this is the draft to train a network for softmax function learning. 



Only Dense layer and soft max

x_train : (batch_size, num_of_dim)
y_train : (batch_size, num_of_classes)  (in one hot label. )


"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM, Softmax, TimeDistributed
from keras.utils import to_categorical

import tensorflow as tf

import numpy as np



#%%
# random a dataset to learn how to use ctc loss
BATCH_SIZE = 10
max_label_seq_length = 200

n_mfcc = 30

NUM_CHARACTERS = 27

# 30 is n_mfcc, 200 is number of frames. 
x_train = np.random.rand(BATCH_SIZE , max_label_seq_length)
# 0~26, 0 [space], 1~26, [a-z]
y_train = to_categorical(np.random.randint(low=0, high= NUM_CHARACTERS, size=BATCH_SIZE), num_classes=27)


#%% input parameters
n_input_dim = n_mfcc*max_label_seq_length 


# network parameters
n_hidden = 2048
rate_dropout = 0.05

# build model 
model = Sequential()
model.add(Dense(n_hidden, activation='relu', input_dim = 200))
model.add(Dropout(rate_dropout))

model.add(Dense(n_hidden, activation='relu'))
model.add(Dropout(rate_dropout))

model.add(Dense(n_hidden, activation='relu'))
model.add(Dropout(rate_dropout))

model.add(Dense(NUM_CHARACTERS))
model.add(Softmax(axis=-1))


#optimizer = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
#model.compile(optimizer='rmsprop', loss=ctc_loss)
model.compile(optimizer='rmsprop', loss='mean_squared_error')


model.fit(x_train, y_train)
