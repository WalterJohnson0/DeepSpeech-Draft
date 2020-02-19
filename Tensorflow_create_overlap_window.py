# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 23:59:04 2020

@author: Ganyu Wang


done

\
"""


import tensorflow as tf
import numpy as np


batch_size = 1
mfccs_max_time_len = 431
n_mfcc = 30

n_context = 9

window_width = 2 * n_context + 1
num_channels = n_mfcc


# make a batch
batch_x = np.random.rand(batch_size, mfccs_max_time_len, n_mfcc)

batch_size = tf.shape(input=batch_x)[0]

#%%
# Create a constant convolution filter using an identity matrix, so that the
# convolution returns patches of the input tensor as is, and we can create
# overlapping windows over the MFCCs.
eye_filter = tf.constant(np.eye(window_width * num_channels)
                           .reshape(window_width, num_channels, window_width * num_channels), tf.float64) # pylint: disable=bad-continuation

# Create overlapping windows
batch_x = tf.nn.conv1d(input=batch_x, filters=eye_filter, stride=1, padding='SAME')

# Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels])


with tf.compat.v1.Session() as sess:
    print(sess.run(batch_x))
    sess.close

