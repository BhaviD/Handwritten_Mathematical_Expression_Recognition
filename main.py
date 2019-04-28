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

    def before_dense_net(self,input_x,mask_x):
        #### before flowing into dense blocks ####
        x = input_x
        limit = self.bound(1, self.input_conv_filters, self.input_conv_kernel)
        x = tf.layers.conv2d(x, filters=self.input_conv_filters, strides=self.input_conv_stride,
        kernel_size=self.input_conv_kernel, padding='SAME', data_format='channels_last', use_bias=False, kernel_initializer=tf.random_uniform_initializer(-limit, limit, dtype=tf.float32))
        mask_x = mask_x[:, 0::2, 0::2]
        x = tf.layers.batch_normalization(x, training=self.training, momentum=0.9, scale=True, gamma_initializer=tf.random_uniform_initializer(-1.0/math.sqrt(self.input_conv_filters),
            1.0/math.sqrt(self.input_conv_filters), dtype=tf.float32), epsilon=0.0001)
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2, padding='SAME')
        # input_pre = x
        mask_x = mask_x[:, 0::2, 0::2]
        self.dense_channels += self.input_conv_filters
        dense_out = x
        return mask_x , dense_out
