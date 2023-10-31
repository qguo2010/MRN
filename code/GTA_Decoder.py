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

# TODO(ebrevdo): Remove once _linear is fully deprecated.
Linear = core_rnn_cell._linear  # pylint: disable=protected-access


def GTA_Decoder(GTAD_inputs,
                initial_state,
                attention_states,
                guides,
                cell,
                output_size=None,
                num_heads=1,
                loop_function=None,
                dtype=None,
                scope=None,
                initial_state_attention=False):
    if not GTAD_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s" %
                         attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(scope or "GTA_Decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        batch_size = array_ops.shape(GTAD_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = array_ops.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states,
                                   [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = variable_scope.get_variable(
                "IA_W_%d" % a, [1, 1, attn_size, attention_vec_size], dtype=dtype)
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(
                variable_scope.get_variable(
                    "IA_V_%d" % a, [attention_vec_size], dtype=dtype))

        state = initial_state

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)
            for a in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    y = Linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    y = math_ops.cast(y, dtype)
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                            [2, 3])
                    a = nn_ops.softmax(math_ops.cast(s, dtype=dtypes.float32))
                    # Now calculate the attention-weighted vector d.
                    a = math_ops.cast(a, dtype)
                    d = math_ops.reduce_sum(
                        array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        states_state = []
        states_hidden = []
        prev = None
        batch_attn_size = array_ops.stack([batch_size, attn_size])
        attns = [
            array_ops.zeros(batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
        ]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)
        for i, inp in enumerate(GTAD_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of GTAD_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            inputs = [inp] + attns
            inputs = [math_ops.cast(e, dtype) for e in inputs]
            x = Linear(inputs, input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=True):
                    attns = attention(state)
            else:
                if isinstance(guides, list):

                    attns = attention(state)
                else:
                    guide_single = guides[i]

                    attns = attention((state[0] * guide_single, state[1]))

            with variable_scope.variable_scope("GTAOutput"):
                cell_output = math_ops.cast(cell_output, dtype)
                inputs = [cell_output] + attns
                output = Linear(inputs, output_size, True)
            if loop_function is not None:
                prev = output
            outputs.append(output)
            states_state.append(state[0])
            states_hidden.append(state[1])
        guides = math_ops.tanh(states_state)

    return outputs, state, guides
