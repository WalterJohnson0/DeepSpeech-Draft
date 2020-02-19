# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:26:53 2020

@author: wang_


FLAGEs

All the parameter that need to be set


"""


from absl import flags


FLAGS = flags.FLAGS


def create_flags():
    
    f = flags
    
    f.DEFINE_string('audio_archive', "data/CV_EN/clips", 'the folder path that save all of the audio files')

    f.DEFINE_integer('n_mfcc', 30, "the number of the MFCC")

    f.DEFINE_integer('batch_size', 10, "Batch size for training")

