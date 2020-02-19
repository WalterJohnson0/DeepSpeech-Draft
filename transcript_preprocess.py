# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:58:04 2020

@author: wang_
"""

import numpy as np
import pandas as pd
import re
from keras.utils import to_categorical


ALPHABET_PATH = "data/CV_EN/alphabet.txt"

transcript_train = pd.read_csv("data/CV_EN/train.csv")


def create_alphabet_dict(alphabet_path):
    # load alphabet to program
    alphabet_dict = {}
    
    with open(alphabet_path, 'r') as alphabet_file:
        lines = alphabet_file.readlines()
        ind = 1
        for line in lines:
            if line[0] == '#':
                continue
            alphabet_dict[line[0]] = ind
            ind += 1
            
    return alphabet_dict
        
 
#%% draft dict to do Character2idx
        

def character2idx(sentence, dictionary):
    idx_vect = []
    for character in sentence:
        idx_vect.append(alphabet_dict[character])
    
    return idx_vect



#%% main

alphabet_dict = create_alphabet_dict(ALPHABET_PATH)

#%% 

idx_vect_list = []

for index, row in transcript_train.iterrows():
    idx_vect = character2idx(transcript_train.iloc[index, 2], alphabet_dict)
    idx_vect_list.append(idx_vect)
    
transcript_train['character_idx_vect'] = idx_vect_list

#%%
#transcript_train.to_csv('train_idx.csv', index=False)

