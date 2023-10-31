from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import rnn as rnn_cell

import tensorflow as tf
from models.IA_Encoder import IA_Encoder
from models.GTA_Decoder import GTA_Decoder
import math
import numpy as np

class MRN_CSG:
    def __init__(self, config, scope='MRN_CSG'):
        self.config = config
        with tf.variable_scope(scope, reuse=False):
            # basic setting
            learning_rate = self.config.lr
            W = self.config.W
            batchsize = self.config.batch_size

            IAE_steps = self.config.IAE_steps
            GTAD_steps = self.config.GTAD_steps
            IAE_inputnum = self.config.IAE_inputnum
            GTAD_inputnum = self.config.GTAD_inputnum
            IAE_hidden = self.config.IAE_hidden
            GTAD_hidden = self.config.GTAD_hidden
            GTAD_outputnum = self.config.GTAD_outputnum

            # tf Graph input
            IAE_input = tf.placeholder("float", [None, IAE_steps, IAE_inputnum], name='IAE_input')
            GTAD_input = tf.placeholder("float", [None, GTAD_steps, GTAD_inputnum], name='GTAD_input')
            GTAD_gt = tf.placeholder("float", [None, GTAD_outputnum], name='GTAD_gt')
            IAE_attention_states = tf.placeholder("float", [None, IAE_inputnum, IAE_steps],
                                                      name='IAE_attention_states')
            # Define weights2
            weights = {'out1': tf.Variable(tf.random_normal([GTAD_hidden, GTAD_outputnum]), name='weights')}
            biases = {'out1': tf.Variable(tf.random_normal([GTAD_outputnum]), name='biases')}
            pred, attn_weights = self.MRNCSG(IAE_input, GTAD_input, weights, biases, IAE_attention_states, W)

        cost = tf.reduce_sum(tf.pow(tf.subtract(pred, GTAD_gt), 2))
        tf.summary.scalar('loss', cost)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


        ### assignment
        self.learning_rate = learning_rate
        self.W = W
        self.batchsize = batchsize
        self.IAE_steps = IAE_steps
        self.GTAD_steps = GTAD_steps
        self.IAE_inputnum = IAE_inputnum
        self.GTAD_inputnum = GTAD_inputnum
        self.IAE_hidden = IAE_hidden
        self.GTAD_hidden = GTAD_hidden
        self.GTAD_outputnum = GTAD_outputnum
        self.IAE_input = IAE_input
        self.GTAD_input = GTAD_input
        self.GTAD_gt = GTAD_gt
        self.IAE_attention_states = IAE_attention_states
        self.y_pred = pred
        self.attn_weights = attn_weights
        self.optimizer = optimizer
        self.cost = cost
        self.summ = tf.summary.merge_all()
        self.summary_updates = tf.get_collection('summary_ops')

    def get_wavelet_conv(self, x, filter, x_axis, name):
        # x_axis:1 represents the wavelet decomposition of the column; 2 represents the wavelet decomposition of the row
        filter_size = len(filter)
        if (x_axis == 1):
            x_size = x.get_shape().as_list()[x_axis]
            filter_V = tf.constant(np.array(filter)[:, np.newaxis, np.newaxis, np.newaxis], dtype=tf.float32)
            filter_V = tf.Variable(filter_V, name=name, trainable=True)

            x = tf.tile(x, [1, math.ceil(filter_size / (2 * x_size)) * 2 + 1, 1, 1])

            crop_index = math.ceil(filter_size / (2 * x_size)) * x_size - int((filter_size / 2))
            x = x[:, crop_index:crop_index + filter_size + x_size, :, :]
            x = tf.pad(x, [[0, 0], [filter_size - 1, filter_size - 1], [0, 0], [0, 0]])

            result = tf.nn.conv2d(x, filter_V, strides=(1, 1, 1, 1), padding='VALID')
            result = result[:, filter_size:filter_size + x_size, :, :]
        return result

    def MRNCSG(self, IAE_input, GTAD_input, weights, biases, IAE_attention_states,W):

        guide_state = []
        IAE_input = IAE_input[:, :, :, np.newaxis]
        IAE_input_wavelet = []

        filter_h = [-0.1294, -0.2241, 0.8365, -0.4830]
        filter_l = [0.4830, 0.8365, 0.2241, -0.1294]

        ori_filter_h = filter_h
        ori_filter_l = filter_l

        filter_size = len(filter_h)

        ### Decomposition.
        # When W=3, it means decomposing twice to obtain "low frequency-medium high frequency-high frequency" signal.
        for l in range(W - 1):
            # print(filter_h)
            # print(filter_l)
            IAE_input_h = self.get_wavelet_conv(IAE_input, filter_h, 1, ('filter_h%d' % l))
            IAE_input_l = self.get_wavelet_conv(IAE_input, filter_l, 1, ('filter_l%d' % l))
            IAE_input_wavelet.append(tf.squeeze(IAE_input_h, axis=3))  # Save the decomposed high-frequency information

            IAE_input = IAE_input_l  # Next decomposition of low-frequency information

            # Simulate SWT, calculate the convolution kernel parameters for the next decomposition.
            # For example: [1,2] -> [0,1,0,2]
            filter_newh = [0 for _ in range(filter_size * int(math.pow(2, l + 1)))]
            for temp in range(filter_size):
                filter_newh[(temp + 1) * int(math.pow(2, l + 1)) - 1] = ori_filter_h[temp]
            filter_h = filter_newh

            filter_newl = [0 for _ in range(filter_size * int(math.pow(2, l + 1)))]
            for temp in range(filter_size):
                filter_newl[(temp + 1) * int(math.pow(2, l + 1)) - 1] = ori_filter_l[temp]
            filter_l = filter_newl

        IAE_input_wavelet.append(tf.squeeze(IAE_input_l, axis=3))  # Save the last low frequency information
        IAE_input_wavelet.reverse()  # Reverse order. For example: [high,medium high,...,low]->[low,...,medium high,high]

        ### 1. Input the signal in the order of "low-medium high-high".
        # 2. Add the guide_state parameter in the decoding process for guidance
        for W_num in range(W):
            IAE_input_per = IAE_input_wavelet[W_num]
            IAE_attention_states_per = tf.transpose(IAE_input_per, [0, 2, 1])

            # MRNCSG MODEL(IAE-GTAD)
            IAE_input_per = tf.transpose(IAE_input_per, [1, 0, 2])
            IAE_input_per = tf.reshape(IAE_input_per, [-1, self.config.IAE_inputnum])
            IAE_input_per = tf.split(IAE_input_per, self.config.IAE_steps, 0, name='IAE_split')

            GTAD_input_per = tf.transpose(GTAD_input, [1, 0, 2])
            GTAD_input_per = tf.reshape(GTAD_input_per, [-1, self.config.GTAD_inputnum])
            GTAD_input_per = tf.split(GTAD_input_per, self.config.GTAD_steps, 0)
            # IAE.
            with tf.variable_scope('IA_Encoder_%d' % W_num) as scope:
                IAE_cell = rnn_cell.BasicLSTMCell(self.config.IAE_hidden, forget_bias=1.0)
                IAE_outputs, IAE_state, attn_weights, guides = IA_Encoder(
                    IAE_input_per,
                    IAE_attention_states_per,
                    IAE_cell)

            # GTAD.
            top_states = [tf.reshape(e, [-1, 1, IAE_cell.output_size]) for e in IAE_outputs]
            attention_states = tf.concat(top_states, 1)
            with tf.variable_scope('GTA_Decoder_%d' % W_num) as scope:
                GTAD_cell = rnn_cell.BasicLSTMCell(self.config.GTAD_hidden, forget_bias=1.0)
                outputs, states, guides = GTA_Decoder(GTAD_input_per, IAE_state,
                                                                            attention_states, guide_state,
                                                                            GTAD_cell)
            guide_state = guides
        return tf.matmul(outputs[-1], weights['out1']) + biases['out1'], attn_weights

    def train(self, one_batch, sess):
        fd = self.get_feed_dict(one_batch)
        [_,loss_per,sss] = sess.run([self.optimizer,self.cost,self.summ], feed_dict=fd)
        return loss_per,sss

    def validation(self, one_batch, sess):
        fd = self.get_feed_dict(one_batch)
        loss_val = sess.run(self.cost, feed_dict=fd)/len(one_batch[1])
        return loss_val

    def predict(self, one_batch, sess):
        fd = self.get_feed_dict(one_batch)
        pred_y = sess.run(self.y_pred, feed_dict=fd)
        loss_test = sess.run(self.cost, feed_dict=fd) / len(one_batch[1])
        return loss_test, pred_y

    def get_feed_dict(self, one_batch):
        fd = {self.IAE_input: one_batch[0],
              self.GTAD_gt: one_batch[1],
              self.GTAD_input: one_batch[2],
              self.IAE_attention_states: one_batch[3]}
        return fd












