# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 20:00:39 2020

@author:  Ganyu Wang


This is the draft for masking.

input and output can be vary in length.

The input and output should using the max length, but with 0 padding. 
for x use "0." for padding, for y use '0' for padding. 


Masking is done.     
    
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM, Softmax, TimeDistributed, Masking
from keras.utils import to_categorical
import tensorflow as tf

import numpy as np
import pandas as pd

from Data_generator_draft import DataGenerator

BATCH_SIZE = 32
time_step_len = 431
window_len = 3
n_mfcc = 30
NUM_CHARACTERS = 28

#%% first draft for train set.
x_train = np.random.rand(BATCH_SIZE, time_step_len, window_len, n_mfcc)\
    .reshape(BATCH_SIZE, time_step_len, window_len*n_mfcc) 
                #use a flatten layer flatten the input to (Batch, time_step, flatten Feature)

#x_train[0, 7:9, :] = 0.  # mark some point to 0.

# 0~26, 0 [space], 1~26, [a-z]
y_train1 = np.random.randint(low=0, high= NUM_CHARACTERS, size=(BATCH_SIZE, time_step_len))
# y_train1[0, 5:9] = 0   # mark some point to 0 or []
y_train = to_categorical(y_train1, num_classes=NUM_CHARACTERS)



#%% second draft with the data generator. 

train_df = pd.read_csv("data/CV_EN/train.csv")
training_generator = DataGenerator(train_df['wav_filename'] , train_df['transcript'])

valid_df = pd.read_csv("data/CV_EN/dev.csv")
validation_generator = DataGenerator(valid_df['wav_filename'] , valid_df['transcript'])



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


#optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer='rmsprop', loss=ctc_loss)


# model.fit_generator(generator=training_generator,
#                     validation_data=validation_generator,
#                     use_multiprocessing=True,
#                     workers=6)

model.train_on_batch(x_train, y_train)
