import tensorflow as tf
import numpy as np
import numpy
import sys
import os
import time
import math

rng = np.random.RandomState(int(time.time()))

def norm_weight(fan_in, fan_out):
    W_bound = np.sqrt(6.0 / (fan_in + fan_out))
    return np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(fan_in, fan_out)), dtype=np.float32)
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

    #Bound function for weight initialisation
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
    
    def bottleneck(self,x):
        ##---------------------Bottleneck layer to improve computational efficiency,i.e.,to reduce the input to 4k feature maps.(k=24)------------------##
        #### [1, 1] convolution part for bottleneck ####
        filter_size = [1,1]
        limit = self.bound(self.dense_channels, 4 * self.growth_rate, filter_size)
        x = tf.layers.conv2d(x, filters=4 * self.growth_rate, kernel_size=filter_size,
            strides=1, padding='VALID', data_format='channels_last', use_bias=False, kernel_initializer=tf.random_uniform_initializer(-limit, limit, dtype=tf.float32))
        x = tf.layers.batch_normalization(inputs=x,  training=self.training, momentum=0.9, scale=True, gamma_initializer=tf.random_uniform_initializer(-1.0/math.sqrt(4 * self.growth_rate),
            1.0/math.sqrt(4 * self.growth_rate), dtype=tf.float32), epsilon=0.0001)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(inputs=x, rate=self.dropout_rate, training=self.training)
        return x
