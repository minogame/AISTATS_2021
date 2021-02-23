from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.util import nest
from tensorflow.contrib.distributions import Bernoulli
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

import numpy as np
import copy
from collections import deque
import pdb
from scipy.linalg import fractional_matrix_power

class TFCTensorLSTMCell(RNNCell):
    """LSTM cell with high order correlations with tensor contraction"""
    def __init__(self, num_units, num_lags, num_orders, rank_vals, is_branched, is_weighted, r_se, forget_bias=1.0, state_is_tuple=True, activation=tanh, reuse=None):
        super(TFCTensorLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._num_lags = num_lags
        self._rank_vals = rank_vals
        self._forget_bias = forget_bias
        self._state_is_tuple= state_is_tuple
        self._activation = activation
        self._num_orders = num_orders
        self._is_branched = is_branched
        self._is_weighted = is_weighted
        self._r_se = r_se
        
    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)
    
    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, states):
        sigmoid = tf.sigmoid
        if self._state_is_tuple:
            hs = ()
            for state in states:
            # every state is a tuple of (c,h)
                c, h = state
                hs += (h,)
        else:
            hs = ()
            for state in states:
                c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)
                hs += (h,)
 
        output_size = 4 * self._num_units
        concat, new_p = tensor_network_tt_einsum(inputs, hs, output_size, self._num_orders, self._rank_vals, True, self._is_branched, self._is_weighted, self._r_se)
        
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

        new_c = (
            c * sigmoid(f + self._forget_bias) + sigmoid(i) * j)
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state

    
def tensor_network_tt_einsum(inputs, states, output_size, num_orders, rank_vals, bias, is_branched, is_weighted, r_se, bias_start=0.0):

    """tensor train decomposition for the full tenosr """
    num_lags = len(states)
    batch_size = tf.shape(inputs)[0] 
    state_size = states[0].get_shape()[1].value #hidden layer size
    input_size= inputs.get_shape()[1].value
    total_state_size = (state_size * num_lags + 1 )

    states_vector = tf.concat(states, 1)
    states_vector = tf.concat([states_vector, tf.ones([batch_size, 1])], 1)
    
    states_vector_all = tf.concat([states_vector, inputs],1)
    weight_all = vs.get_variable("weight_all", [total_state_size+input_size, output_size, rank_vals[0]])
    
    branch_product = tf.tensordot(states_vector_all,weight_all,1)
    branch_save = tf.greater_equal(branch_product,0)
    branch_save = tf.to_float(branch_save) * 2 - 1
    branch_product = np.abs(branch_product)+0.01

    new_p = tf.layers.dense(inputs=states_vector_all, activation=tf.nn.sigmoid, use_bias=True, units=(1))
    new_p = tf.expand_dims(new_p,2)
    branch_result = tf.pow(branch_product, new_p)
    branch_result = tf.multiply(branch_result,branch_save)
    

    if is_weighted:
        # apply se-block
        squeeze = tf.reduce_mean(branch_result, 1)
        excitation = tf.layers.dense(inputs=squeeze, activation=tf.nn.relu, use_bias=True, units=(rank_vals[0])/r_se)
        excitation = tf.layers.dense(inputs=excitation, activation=tf.nn.sigmoid, use_bias=True, units=(rank_vals[0]))
        excitation = tf.expand_dims(excitation, 1)
        branch_result = tf.multiply(branch_result, excitation)
    
    res = tf.reduce_sum(branch_result, -1)

    if not bias:
        return res
    biases = vs.get_variable("biases", [output_size])
    return nn_ops.bias_add(res,biases), new_p
    