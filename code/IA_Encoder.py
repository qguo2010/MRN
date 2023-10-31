from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
import tensorflow as tf

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = core_rnn_cell._linear  # pylint: disable=protected-access

def IA_Encoder(IAE_inputs, attention_states, cell, num_heads=1,
               dtype=dtypes.float32, scope=None):

    batch_size = array_ops.shape(IAE_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = array_ops.reshape(attention_states, [-1, attn_length, 1,
                                                  attn_size])
    attention_vec_size = attn_size  # Size of query vectors for attention.
    k = tf.Variable(tf.zeros([1, 1, attn_size, attention_vec_size]), name='IA_EncoderW')
    v = tf.Variable(tf.zeros([attention_vec_size]), name='IA_EncoderV')
    hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")

    # how to get the initial_state
    initial_state_size = array_ops.stack([batch_size, cell.output_size])
    initial_state = [array_ops.zeros(initial_state_size, dtype=dtype) for _ in xrange(2)]
    state = initial_state

    def attention(query):
        ds = []
        if nest.is_sequence(query):
            query_list = nest.flatten(query)
            query = array_ops.concat(query_list, 1)

        y = linear(query, attention_vec_size, True)  # y shape (batch_size, attention_vec_size)
        y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])  # y shape (batch_size, 1, 1, attention_vec_size)

        s = math_ops.reduce_sum(v * math_ops.tanh(hidden_features + y),
                                [2, 3])  # hidden_features shape (batch_size, attn_length, 1, attn_size)
        a = nn_ops.softmax(s)  # a shape (batch_size, attn_length)
        ds.append(a)
        return ds

    outputs = []
    attn_weights = []
    states_guide = []
    batch_attn_size = array_ops.stack([batch_size, attn_length])
    attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
             for _ in xrange(num_heads)]

    # i is the index of the which time step
    # inp is numpy.array and the shape of inp is (batch_size, n_feature)
    for i, inp in enumerate(IAE_inputs):
        if i > 0:
            variable_scope.get_variable_scope().reuse_variables()
        input_size = inp.get_shape().with_rank(2)[1]

        # multiply attention weights with the original input
        # get the newly input
        x = attns[0] * inp
        # Run the BasicLSTM with the newly input
        cell_output, state = cell(x, state)

        attns = attention(state)
        outputs.append(cell_output)
        attn_weights.append(attns)
        states_guide.append(state)

    return outputs, state, attn_weights, states_guide
