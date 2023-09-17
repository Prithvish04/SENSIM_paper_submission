# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 19:07:07 2020

@author: yousef21

This file contains simplified implementation of delta activation layer
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


@tf.custom_gradient
def round_custom(x):
  def grad(dy):
    return dy
  return tf.round(x), grad

@tf.custom_gradient
def no_grad_custom(x):
  def grad(dy):
    return dy * 0.0
  return x, grad

@tf.custom_gradient
def neg_grad_custom(x):
  def grad(dy):
    return dy * (-1.0)
  return x, grad

def quantize(x, q):
  return tf.multiply(round_custom(tf.divide(x, no_grad_custom(q))), neg_grad_custom(q))

class DAL(keras.layers.Layer): 
    '''
    Arguments:
    threshold_level='channel-wise', the granuality of qualtization level (threshold), can be channel-wise, neuron-wise or layer-wise
    sp_rate=1.0,  Sparsity factor for the loss function of sparsity, increasing this factor will push toward more sparsity with the cost of accuracy
    n_outputs=1.0, Number of outputs used in the same way as sparsity factor to increase this factor for high fan-out neurons
    thr_init=1e-1, Initial value of quantization level (threshold)
    trainabale_bias=False, Add traininable bias parameters (in the case of delta procesing, the linear operations, like conv and dense should not contain bias)
    activation=None, Define the Activation function here, currently only 'relu' and 'softmax' is added
    max_pooling=None, Define the max_pool kernel size if required
    sigma =False, If you don't want the layer to perform integration operation (sigma) leave this to False
    delta_level=0, If you don't want the layer to perform delta operation (delta) leave this to 0, 1 is when normal delta out is required. 2 is when the first frame is not required (DVS type without backgound)
    name='delta_activation'
    layer_name=None, 
    show_metric=False
    '''
    def __init__(self, threshold_level='layer-wise', sp_rate=1.0, n_outputs=1.0, thr_init=1e-1, show_metric=False, name='delta_activation', thr_trainable=True):
        super(DAL, self).__init__(name=name)
        self.rate = sp_rate
        self.n_outputs = n_outputs
        self.thr_init = thr_init
        self.layer_n = name
        self.threshold_level = threshold_level
        self.show_metric = show_metric
        self.thr_trainable = thr_trainable
        
    def build(self, input_shape):    #input shape is (batch,time,x,y,c)
        #One threshold per neuron
        if(self.threshold_level=='neuron-wise'):
          self.threshold = self.add_weight(shape=input_shape[2:], initializer=keras.initializers.Constant(value=self.thr_init), trainable=self.thr_trainable, name='threshold') #threshold shape: (x,y,c)
        
        #One threshold per channel
        if(self.threshold_level=='channel-wise'):
          self.threshold = self.add_weight(shape=input_shape[-1], initializer=keras.initializers.Constant(value=self.thr_init), trainable=self.thr_trainable, name='threshold') #threshold shape: (c)

        #One threshold per layer
        if(self.threshold_level=='layer-wise'):
          self.threshold = self.add_weight(shape=1, initializer=keras.initializers.Constant(value=self.thr_init), trainable=self.thr_trainable, name='threshold') #threshold shape: 1


    def call(self, inputs, layer_name=None):
        threshold = tf.math.abs(self.threshold)
        neuron_states = inputs
        output = quantize(neuron_states, threshold)

        #Delta calculation
        # Make the old output (with a blank first frame)  [same as output with a time shift]
        # To measure number of spikes, we do not consider the first wave of spikes from blank frame
        if len(output.shape)==5: #NNtype=='cnn'  
          spikes = tf.subtract(output[:,:-1,:,:,:], output[:,1:,:,:,:])
          output_old = tf.concat([tf.zeros_like(output[:,0:1,:,:,:]),output[:,:-1,:,:,:]],1)
        if len(output.shape)==3: #NNtype=='fc'  
          spikes = tf.subtract(output[:,:-1,:], output[:,1:,:])  
          output_old = tf.concat([tf.zeros_like(output[:,0:1,:]),output[:,:-1,:]],1)
        
        
        n_frames = tf.cast(tf.math.multiply(tf.shape(spikes)[0],tf.shape(spikes)[1]), dtype=tf.float32) 
        spikes_L0 = tf.cast(tf.math.count_nonzero(spikes), dtype=tf.float32) 
        spikes_L1 = tf.math.reduce_sum(tf.math.abs(spikes))

        n_neurons = tf.cast(tf.math.reduce_prod(tf.shape(spikes)), dtype=tf.float32) 
        spikes_L1_normalized = tf.divide(spikes_L1, n_neurons) 
        spikes_L0_normalized = tf.divide(spikes_L0, n_neurons) 
        
        self.add_loss(self.rate * self.n_outputs * spikes_L1_normalized) #L1 loss weighted with n_operations
        if self.show_metric:
          self.add_metric(tf.math.divide(spikes_L0_normalized,n_frames), name='n_spikes_'+self.layer_n, aggregation='mean') #moving average number of spikes (normalized)
        return output, spikes_L0, n_neurons