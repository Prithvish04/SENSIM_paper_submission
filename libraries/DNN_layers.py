import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import numpy as np
import tensorflow
from tensorflow.python.keras.backend_config import epsilon


class layer:
    def __init__(self, 
                ID=None,
                layer_type=None, 
                pooling=None, pooling_size=[1,1], padding='same', is_flatten=False,
                neuron_type=None, threshold=None, act_fun=None,
                shape=None, 
                name=None,
                # layerIdx=None,
                weight_tensor=[], bias_tensor=[], outputLayer=()):
        '''
        layer object definitions
        
        layer_type: layer operation type ('input', 'output', 'conv', 'dense', 'flatten', ...) + params
            - input and output layers are dummy (no energy, no latency) but must be implemented as input and output layers and inside individual cores
            layer_params: parameters specific to each layer type
                - conv layer: pooling type('stride', 'avgpool', None), pooling_size=[H, W], padding ('same','valid'), is_flatten
                - dense layer: -
        
        neuron_type: type of processing in the neuron('SigmaDelta', ...) + params
            - SigmaDelta params(Sigma+Delta):  list of channel-wise threshold [thr0,thr1,...]
            

        act_fun: activation function (None=linear, 'ReLU', 'sigmoid', ...)

        shape: shape of neuron state in this layer (calculate based on the input size, padding and pooling)
        name: optional layer name
        '''
        self.ID = ID
        self.layer_type = layer_type
        self.pooling = pooling
        self.pooling_size = pooling_size
        self.padding = padding
        self.is_flatten = is_flatten

        self.neuron_type = neuron_type
        self.threshold = threshold

        self.act_fun = act_fun

        self.shape = shape

        self.name = name
        # self.layerIdx = layerIdx
        
        if self.layer_type!='input' and self.layer_type!='output':
            self.weights = weight_tensor
            self.biases = bias_tensor
            # Initial states for neurons is the biases
            self.neuron_states = np.zeros(self.shape) + self.biases        
            # output old is required for sigma delta neuron types
            self.old_output= np.zeros(self.neuron_states.shape)
        
        if self.threshold is not None:
            if all(self.threshold)==False: self.threshold +=np.finfo(float).eps #thresholds cannot be zero

        self.output_layers = outputLayer
        
    def setOutputLayer(self, output_layers=()):   
        self.output_layers = output_layers

    def act(self, x):
        if self.act_fun==None:
            return x 
        elif self.act_fun=='ReLU':
            return np.maximum(x,0)
        elif self.act_fun=='Sigmoid':
            return 1/(1+np.exp(-x))
        elif self.act_fun=='Tanh':
            return np.tanh(x)
        else:
            raise Exception("ActFun ",self.act_fun," not implemented...")

    def act_time(self):
        if self.act_fun==None:
            return 0 
        elif self.act_fun=='ReLU':
            return 1
        elif self.act_fun=='Sigmoid':
            return 3
        elif self.act_fun=='Tanh':
            return 10

    def quantize(self, x):
        return np.round(x/self.threshold)*self.threshold


