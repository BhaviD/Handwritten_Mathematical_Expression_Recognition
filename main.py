import tensorflow as tf
import numpy as np
import numpy
import sys
import os
import time
import math

class Watcher_train():
    def __init__(self, blocks,             # number of dense blocks
                                level,                     # number of levels in each blocks
                                growth_rate,               # growth rate as mentioned in DenseNet paper: k
                                training,
                                dropout_rate=0.2,          # Dropout layer's keep-rate
                                dense_channels=0,          # Number of filters in transition layer's input
                                transition=0.5,            # Compression rate
                                input_conv_filters=48,     # Number of filters of conv2d before dense blocks
                                input_conv_stride=2,       # Stride of conv2d placed before dense blocks
                                input_conv_kernel=[7,7]):  # Size of kernel of conv2d placed before dense blocks
        self.blocks = blocks
        self.level = level
        self.growth_rate = growth_rate
        self.training = training
        self.dense_channels = dense_channels
        self.dropout_rate = dropout_rate
        self.transition = transition
        self.input_conv_filters = input_conv_filters
        self.input_conv_stride = input_conv_stride
        self.input_conv_kernel = input_conv_kernel

    def bound(self, nin, nout, kernel):
        kernel_dim_1 = kernel[0]
        kernel_dim_2 = kernel[1]
        mul = kernel_dim_1  * kernel_dim_2
        fin = nin * mul
        fout = nout * mul
        result = (6. / (fin + fout))
        result = np.sqrt(result)
        return result
