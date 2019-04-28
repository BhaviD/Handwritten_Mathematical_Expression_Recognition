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

def conv_norm_weight(nin, nout, kernel_size):
    filter_shape = (kernel_size[0], kernel_size[1], nin, nout)
    fan_in = kernel_size[0] * kernel_size[1] * nin
    fan_out = kernel_size[0] * kernel_size[1] * nout
    W_bound = np.sqrt(6. / (fan_in + fan_out))
    W = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=np.float32)
    return W.astype('float32')

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
    
    def convolution_layer_in_DenseB(self,x):
        #### [3, 3] filter for regular convolution layer
        filter_size = [3,3]
        limit = self.bound(4 * self.growth_rate, self.growth_rate,filter_size )
        x = tf.layers.conv2d(x, filters=self.growth_rate, kernel_size=filter_size,
            strides=1, padding='SAME', data_format='channels_last', use_bias=False, kernel_initializer=tf.random_uniform_initializer(-limit, limit, dtype=tf.float32))
        return x
      
    def transition_layer(self,x,mask_x):
        ####There is no transition layer after last DenseB layer,so this module is not run for the last block.####
        compressed_channels = int(self.dense_channels * self.transition)
        #### new dense channels for new dense block ####
        self.dense_channels = compressed_channels
        limit = self.bound(self.dense_channels, compressed_channels, [1,1])
        x = tf.layers.conv2d(x, filters=compressed_channels, kernel_size=[1,1],
            strides=1, padding='VALID', data_format='channels_last', use_bias=False, kernel_initializer=tf.random_uniform_initializer(-limit, limit, dtype=tf.float32))
        x = tf.layers.batch_normalization(x, training=self.training, momentum=0.9, scale=True, gamma_initializer=tf.random_uniform_initializer(-1.0/math.sqrt(self.dense_channels),
                1.0/math.sqrt(self.dense_channels), dtype=tf.float32), epsilon=0.0001)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(inputs=x, rate=self.dropout_rate, training=self.training)
        x = tf.layers.average_pooling2d(inputs=x, pool_size=[2,2], strides=2, padding='SAME')
        dense_out = x
        mask_x = mask_x[:, 0::2, 0::2]
        return x,dense_out,mask_x

    def DenseB_and_transition_layer(self,x,mask_x,dense_out):
        #### flowing into dense blocks and transition_layer ####
        for i in range(self.blocks):
            for j in range(self.level):
                ##----------------------------------------------------------DenseB Layer---------------------------------------------------------------------------##
                #### Bottleneck layer ####
                x = self.bottleneck(x)
                #### 3x3 Convolution Layer ####
                x = self.convolution_layer_in_DenseB(x)
                #### Batch Normalisation Layer ####
                x = tf.layers.batch_normalization(inputs=x, training=self.training, momentum=0.9, scale=True, gamma_initializer=tf.random_uniform_initializer(-1.0/math.sqrt(self.growth_rate),
                    1.0/math.sqrt(self.growth_rate), dtype=tf.float32), epsilon=0.0001)
                #### Relu Activation Layer ####
                x = tf.nn.relu(x)
                x = tf.layers.dropout(inputs=x, rate=self.dropout_rate, training=self.training)
                dense_out = tf.concat([dense_out, x], axis=3)
                x = dense_out
                #### calculate the filter number of dense block's output ####
                self.dense_channels += self.growth_rate

            if i < self.blocks - 1:
                ##---------------------------------------------------------Transition Layer------------------------------------------------------------------------##
                x,dense_out,mask_x = self.transition_layer(x,mask_x)

        return mask_x ,dense_out
      
      

class Attender():
    def __init__(self, channels,                                # output of Watcher | [batch, h, w, channels]
                dim_decoder, dim_attend):                       # decoder hidden state:$h_{t-1}$ | [batch, dec_dim]

        self.channels = channels

        self.coverage_kernel = [11,11]                          # kernel size of $Q$
        self.coverage_filters = dim_attend                      # filter numbers of $Q$ | 512

        self.dim_decoder = dim_decoder                          # 256
        self.dim_attend = dim_attend                            # unified dim of three parts calculating $e_ti$ i.e.
                                                                # $Q*beta_t$, $U_a * a_i$, $W_a x h_{t-1}$ | 512

        self.U_f = tf.Variable(norm_weight(self.coverage_filters, self.dim_attend), name='U_f') # $U_f x f_i$ | [cov_filters, dim_attend]
        self.U_f_b = tf.Variable(np.zeros((self.dim_attend,)).astype('float32'), name='U_f_b')  # $U_f x f_i + U_f_b$ | [dim_attend, ]

        self.U_a = tf.Variable(norm_weight(self.channels,
            self.dim_attend), name='U_a')                                                      # $U_a x a_i$ | [annotatin_channels, dim_attend]
        self.U_a_b = tf.Variable(np.zeros((self.dim_attend,)).astype('float32'), name='U_a_b') # $U_a x a_i + U_a_b$ | [dim_attend, ]

        self.W_a = tf.Variable(norm_weight(self.dim_decoder,
            self.dim_attend), name='W_a')                                                      # $W_a x h_{t_1}$ | [dec_dim, dim_attend]
        self.W_a_b = tf.Variable(np.zeros((self.dim_attend,)).astype('float32'), name='W_a_b') # $W_a x h_{t-1} + W_a_b$ | [dim_attend, ]

        self.V_a = tf.Variable(norm_weight(self.dim_attend, 1), name='V_a')                    # $V_a x tanh(A + B + C)$ | [dim_attend, 1]
        self.V_a_b = tf.Variable(np.zeros((1,)).astype('float32'), name='V_a_b')               # $V_a x tanh(A + B + C) + V_a_b$ | [1, ]

        self.alpha_past_filter = tf.Variable(conv_norm_weight(1, self.dim_attend, self.coverage_kernel), name='alpha_past_filter')


class WAP():
    def __init__(self, watcher, attender, parser, hidden_dim, word_dim, context_dim, target_dim, training):
        # self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.word_dim = word_dim
        self.context_dim = context_dim
        self.target_dim = target_dim
        self.embed_matrix = tf.Variable(norm_weight(self.target_dim, self.word_dim), name='embed')

        self.watcher = watcher
        self.attender = attender
        self.parser = parser
        self.Wa2h = tf.Variable(norm_weight(self.context_dim, self.hidden_dim), name='Wa2h')
        self.ba2h = tf.Variable(np.zeros((self.hidden_dim,)).astype('float32'), name='ba2h')
        self.Wc = tf.Variable(norm_weight(self.context_dim, self.word_dim), name='Wc')
        self.bc = tf.Variable(np.zeros((self.word_dim,)).astype('float32'), name='bc')
        self.Wh = tf.Variable(norm_weight(self.hidden_dim, self.word_dim), name='Wh')
        self.bh = tf.Variable(np.zeros((self.word_dim,)).astype('float32'), name='bh')
        self.Wy = tf.Variable(norm_weight(self.word_dim, self.word_dim), name='Wy')
        self.by = tf.Variable(np.zeros((self.word_dim,)).astype('float32'), name='by')
        self.Wo = tf.Variable(norm_weight(self.word_dim//2, self.target_dim), name='Wo')
        self.bo = tf.Variable(np.zeros((self.target_dim,)).astype('float32'), name='bo')
        self.training = training

    def get_word(self, sample_y, sample_h_pre, alpha_past_pre, sample_annotation):

        emb = tf.cond(sample_y[0] < 0,
            lambda: tf.fill((1, self.word_dim), 0.0),
            lambda: tf.nn.embedding_lookup(wap.embed_matrix, sample_y)
            )

        #ret = self.parser.one_time_step((h_pre, None, None, alpha_past_pre, annotation, None), (emb, None))
        emb_y_z_r_vector = tf.tensordot(emb, self.parser.W_yz_yr, axes=1) + \
        self.parser.b_yz_yr                                            # [batch, 2 * dim_decoder]
        hidden_z_r_vector = tf.tensordot(sample_h_pre,
        self.parser.U_hz_hr, axes=1)                                   # [batch, 2 * dim_decoder]
        pre_z_r_vector = tf.sigmoid(emb_y_z_r_vector + \
        hidden_z_r_vector)                                             # [batch, 2 * dim_decoder]

        r1 = pre_z_r_vector[:, :self.parser.hidden_dim]                # [batch, dim_decoder]
        z1 = pre_z_r_vector[:, self.parser.hidden_dim:]                # [batch, dim_decoder]

        emb_y_h_vector = tf.tensordot(emb, self.parser.W_yh, axes=1) + \
        self.parser.b_yh                                               # [batch, dim_decoder]
        hidden_r_h_vector = tf.tensordot(sample_h_pre,
        self.parser.U_rh, axes=1)                                      # [batch, dim_decoder]
        hidden_r_h_vector *= r1
        pre_h_proposal = tf.tanh(hidden_r_h_vector + emb_y_h_vector)

        pre_h = z1 * sample_h_pre + (1. - z1) * pre_h_proposal

        context, _, alpha_past = self.parser.attender.get_context(sample_annotation, pre_h, alpha_past_pre, None)  # [batch, dim_ctx]
        emb_y_z_r_nl_vector = tf.tensordot(pre_h, self.parser.U_hz_hr_nl, axes=1) + self.parser.b_hz_hr_nl
        context_z_r_vector = tf.tensordot(context, self.parser.W_c_z_r, axes=1)
        z_r_vector = tf.sigmoid(emb_y_z_r_nl_vector + context_z_r_vector)

        r2 = z_r_vector[:, :self.parser.hidden_dim]
        z2 = z_r_vector[:, self.parser.hidden_dim:]

        emb_y_h_nl_vector = tf.tensordot(pre_h, self.parser.U_rh_nl, axes=1) + self.parser.b_rh_nl
        emb_y_h_nl_vector *= r2
        context_h_vector = tf.tensordot(context, self.parser.W_c_h_nl, axes=1)
        h_proposal = tf.tanh(emb_y_h_nl_vector + context_h_vector)
        h = z2 * pre_h + (1. - z2) * h_proposal

        h_t = h
        c_t = context
        alpha_past_t = alpha_past
        y_t_1 = emb
        logit_gru = tf.tensordot(h_t, self.Wh, axes=1) + self.bh
        logit_ctx = tf.tensordot(c_t, self.Wc, axes=1) + self.bc
        logit_pre = tf.tensordot(y_t_1, self.Wy, axes=1) + self.by
        logit = logit_pre + logit_ctx + logit_gru   # batch x word_dim

        shape = tf.shape(logit)
        logit = tf.reshape(logit, [-1, shape[1]//2, 2])
        logit = tf.reduce_max(logit, axis=2)

        logit = tf.layers.dropout(inputs=logit, rate=0.2, training=self.training)

        logit = tf.tensordot(logit, self.Wo, axes=1) + self.bo

        next_probs = tf.nn.softmax(logits=logit)
        next_word  = tf.reduce_max(tf.multinomial(next_probs, num_samples=1), axis=1)
        return next_probs, next_word, h_t, alpha_past_t

    def get_cost(self, cost_annotation, cost_y, a_m, y_m):
        timesteps = tf.shape(cost_y)[0]
        batch_size = tf.shape(cost_y)[1]
        emb_y = tf.nn.embedding_lookup(self.embed_matrix, tf.reshape(cost_y, [-1]))
        emb_y = tf.reshape(emb_y, [timesteps, batch_size, self.word_dim])
        emb_pad = tf.fill((1, batch_size, self.word_dim), 0.0)
        emb_shift = tf.concat([emb_pad ,tf.strided_slice(emb_y, [0, 0, 0], [-1, batch_size, self.word_dim], [1, 1, 1])], axis=0)
        new_emb_y = emb_shift
        anno_mean = tf.reduce_sum(cost_annotation * a_m[:, :, :, None], axis=[1, 2]) / tf.reduce_sum(a_m, axis=[1, 2])[:, None]
        h_0 = tf.tensordot(anno_mean, self.Wa2h, axes=1) + self.ba2h  # [batch, hidden_dim]
        h_0 = tf.tanh(h_0)

        ret = self.parser.get_ht_ctx(new_emb_y, h_0, cost_annotation, a_m, y_m)
        h_t = ret[0]                      # h_t of all timesteps [timesteps, batch, word_dim]
        c_t = ret[1]                      # c_t of all timesteps [timesteps, batch, context_dim]

        y_t_1 = new_emb_y                 # shifted y | [1:] = [:-1]
        logit_gru = tf.tensordot(h_t, self.Wh, axes=1) + self.bh
        logit_ctx = tf.tensordot(c_t, self.Wc, axes=1) + self.bc
        logit_pre = tf.tensordot(y_t_1, self.Wy, axes=1) + self.by
        logit = logit_pre + logit_ctx + logit_gru
        shape = tf.shape(logit)
        logit = tf.reshape(logit, [shape[0], -1, shape[2]//2, 2])
        logit = tf.reduce_max(logit, axis=3)

        logit = tf.layers.dropout(inputs=logit, rate=0.2, training=self.training)

        logit = tf.tensordot(logit, self.Wo, axes=1) + self.bo
        logit_shape = tf.shape(logit)
        logit = tf.reshape(logit, [-1,
            logit_shape[2]])
        cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=tf.one_hot(tf.reshape(cost_y, [-1]),
            depth=self.target_dim))

        cost = tf.multiply(cost, tf.reshape(y_m, [-1]))
        cost = tf.reshape(cost, [shape[0], shape[1]])
        cost = tf.reduce_sum(cost, axis=0)
        cost = tf.reduce_mean(cost)
        return cost

# wap_obj = WAP(None,None,None,1,1,1,1,False)
