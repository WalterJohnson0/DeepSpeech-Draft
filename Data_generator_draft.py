# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 21:30:03 2020

@author: Ganyu Wang


This is the script to learn how to use data generator. 





"""
import os

import pandas as pd
import numpy as np
import keras


import librosa
import librosa.display
from librosa.feature import mfcc


#%% main

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df_wav_filename, df_transcript, batch_size=32, time_step=431, window_len=3, n_mfcc=30 , n_classes=28):
        'Initialization'
        self.df_wav_filename = df_wav_filename
        self.df_transcript = df_transcript
        
        self.batch_size = batch_size
        self.time_step = time_step
        self.n_window = time_step
        self.window_len = window_len
        self.n_mfcc = n_mfcc
        self.n_classes = n_classes
        #
        self.alphabet = {}
        self.create_alphabet_dict("data/CV_EN/alphabet.txt")
        
        
    def create_alphabet_dict(self, alphabet_path):
        # load alphabet to program
        alphabet_dict = {}
        
        with open(alphabet_path, 'r') as alphabet_file:
            lines = alphabet_file.readlines()
            ind = 0
            for line in lines:
                if line[0] == '#':
                    continue
                alphabet_dict[line[0]] = ind
                ind += 1
                
        self.alphabet = alphabet_dict
        
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df_wav_filename) / self.batch_size))

    def __getitem__(self, idx):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        # Initialization (这是没做，overlap window的。方案将第一层做成overlap window。或者测试后重新做，用np的conv1d。)
        batch_X_before_window = np.zeros((self.batch_size, self.time_step, self.n_mfcc))
        
        batch_X = np.zeros((self.batch_size, self.n_window, self.window_len * self.n_mfcc))
        batch_y = np.zeros((self.batch_size, self.time_step, self.n_classes), dtype=int)
    
        for sample_idx in range(self.batch_size):
             # x
             tmp_x = self.get_mfccs_from_file_name(self.df_wav_filename[idx*self.batch_size+sample_idx])
             tmp_x_shape0 = tmp_x.shape[0]
             batch_X_before_window[sample_idx, :tmp_x_shape0,] = tmp_x
             
             window_num = self.n_window - self.window_len + 1
             for window_idx in range(window_num):
                 batch_X[sample_idx, window_idx, :] =\
                     batch_X_before_window[sample_idx, window_idx: window_idx+self.window_len, :]\
                         .reshape(self.window_len * self.n_mfcc) 
             
             # y
             tmp_y = np.array(self.character2idx(self.df_transcript[idx*self.batch_size+sample_idx]))
             cat_y = keras.utils.to_categorical(tmp_y, self.n_classes)
             tmp_y_shape0 = tmp_y.shape[0]
             batch_y[sample_idx, :tmp_y_shape0, ] = cat_y
             
             
        return batch_X, batch_y
    
    def get_mfccs_from_file_name(self, file_name):
        
        pre, ext = os.path.splitext(file_name)
        y, sr = librosa.load('data/CV_EN/clips/' + pre + '.mp3')  # FLAGE Audio achive path
        mfccs = mfcc(y=y, sr=sr, n_mfcc= 30).T
        mfccs = np.array(mfccs)
        return mfccs
    
    def character2idx(self, sentence):
        idx_vect = []
        for character in sentence:
            idx_vect.append(self.alphabet[character])
        
        return idx_vect



if __name__ == "__main__":
    
        # Parameters
    params = {'batch_size': 32,
              'n_classes': 28}
    
    
    train_df = pd.read_csv("data/CV_EN/train.csv")
    training_generator = DataGenerator(train_df['wav_filename'] , train_df['transcript'], **params)
    


