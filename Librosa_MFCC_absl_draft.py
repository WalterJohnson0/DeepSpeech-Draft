# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:45:00 2020

@author: wang_

import file
do mfcc
get a matrix

"""

import librosa
import librosa.display
from librosa.feature import mfcc
import numpy as np

from util.flags import create_flags, FLAGS
import absl.app

#import matplotlib.pyplot as plt


#%% draft

n_mfcc = 40

y1, sr1 = librosa.load('data/clips/common_voice_en_1.mp3')
y2, sr2 = librosa.load('data/clips/common_voice_en_10.mp3')


mfccs1 = mfcc(y=y1, sr=sr1, n_mfcc=n_mfcc)
mfccs2 = mfcc(y=y2, sr=sr2, n_mfcc=n_mfcc)




#%% Max number length: (n_mfcc, 431)

y3 = np.random.rand(220500)
sr3 = sr1

mfccs3 = mfcc(y=y3, sr=sr3, n_mfcc=n_mfcc)
mfccs3 = mfccs3.T


#%% function
def get_mfccs_from_file_name(file_name):
    
    y, sr = librosa.load(FLAGS.audio_archive + '/' + file_name)  # FLAGE Audio achive path
    mfccs = mfcc(y=y, sr=sr, n_mfcc= FLAGS.n_mfcc).T
    
    return mfccs



def main(_):
    a = get_mfccs_from_file_name("common_voice_en_1.mp3") 
    print(a)



#%% main test

if __name__ == '__main__':
    
    create_flags()
    absl.app.run(main)
    

