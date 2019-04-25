import tensorflow as tf
import numpy as np
import numpy
import sys
import os
import time
import math

#Bound function for weight initialisation
def bound(self, nin, nout, kernel):
    #bound(1, self.input_conv_filters, self.input_conv_kernel)
    kernel_dim_1 = kernel[0]
    kernel_dim_2 = kernel[1]
    mul = kernel_dim_1  * kernel_dim_2
    fin = nin * mul
    fout = nout * mul
    result = (6. / (fin + fout)
    result = np.sqrt(result)
    return result

rng = np.random.RandomState(int(time.time()))

def norm_weight(fan_in, fan_out):
    W_bound = np.sqrt(6.0 / (fan_in + fan_out))
    return np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(fan_in, fan_out)), dtype=np.float32)