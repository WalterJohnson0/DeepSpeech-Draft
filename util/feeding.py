# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:11:34 2020

@author: wang_


This is the draft for feeding package

default input is 

"data/CV_EN/clips/train.csv"
"data/CV_EN/clips/train.csv"
"data/CV_EN/clips/train.csv"


the main function is create_dataset()
    it will return a keras data generator. 
    

"""


#%% sequence generater draft

from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

data = np.array([[i] for i in range(50)])
targets = np.array([[i] for i in range(50)])

data_gen = TimeseriesGenerator(data, targets,
                               length=10, sampling_rate=2,
                               batch_size=2)
assert len(data_gen) == 20

batch_0 = data_gen[0]
x, y = batch_0
assert np.array_equal(x,
                      np.array([[[0], [2], [4], [6], [8]],
                                [[1], [3], [5], [7], [9]]]))
assert np.array_equal(y,
                      np.array([[10], [11]]))



#%%





