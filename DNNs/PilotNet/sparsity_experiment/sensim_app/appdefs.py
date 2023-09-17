import os
import numpy as np
from libraries.DNN_layers import layer
from libraries.interconnect import interconnect


class PilotNet_app():
    def __init__(self, simObj, threshold_factor=1.0):
        self.threshold_factor = threshold_factor
        self.sim = simObj
        # Modify experiment specific parameters
        from DNNs.PilotNet.sparsity_experiment.sensim_app.parameters import parameters
        self.sim.output_dir = "DNNs/PilotNet/sparsity_experiment/outputs/outputs_map2/"

        os.makedirs(self.sim.output_dir, exist_ok=True)
        self.sim.param = parameters()
        
        #Timing
        self.sim.simulation_end_time = 1000e6
        self.sim.param.evaluation_time_step= 1e6
        self.sim.param.snapshot_log_time_step = 1e6
        self.sim.param.time_step= 1e4

        #HW
        self.sim.param.N_NPE = 8

        #Spike packet
        self.sim.param.flit_width = 32 #number of bits per flits        
        self.sim.param.max_event_flits = 1 # limit the max number of flits per event
        self.sim.param.spike_per_flits = 1 # number of [C,Value] per each flit 
        self.sim.param.header_flits = 0 # number of flits for the header [Source Layer, H, W] --> can be 0 for single flit AER events


        self.sim.fetchWholeDataSet = False
        self.sim.setHW()
        return

    def composeLayers(self):
        self.layers = {}
        # load SNN parameters
        weights    = np.load('./DNNs/PilotNet/keras/weights_delta.npy',  allow_pickle=True)
        thresholds = np.load('./DNNs/PilotNet/keras/thresholds_delta.npy',  allow_pickle=True)*self.threshold_factor
        
        # Define layers (layer_type=None, pooling=None, pooling_size=None, padding='same', is_flatten=False, neuron_type=None, threshold=None, shape=None, name=None)
        # Calculate the shape of the layer: ceil{(input_shape-padding)/pooling} [sorry that you need to do it manually]
        # input shape is [66,200,3]
        self.layers['O']  = layer(layer_type='output', name='output')
        self.layers['L10']= layer(layer_type='dense', neuron_type='SigmaDelta', threshold=np.ones([   1])*0,  act_fun='Tanh',  shape=[   1], name='dns5', weight_tensor=weights[9][0], bias_tensor=weights[9][1], outputLayer=(self.layers['O'],))
        self.layers['L9'] = layer(layer_type='dense', neuron_type='SigmaDelta', threshold=np.ones([  10])*thresholds[9],  act_fun='ReLU',  shape=[  10], name='dns4', weight_tensor=weights[8][0], bias_tensor=weights[8][1], outputLayer=(self.layers['L10'],))
        self.layers['L8'] = layer(layer_type='dense', neuron_type='SigmaDelta', threshold=np.ones([  50])*thresholds[8],  act_fun='ReLU',  shape=[  50], name='dns3', weight_tensor=weights[7][0], bias_tensor=weights[7][1], outputLayer=(self.layers['L9'],))
        self.layers['L7'] = layer(layer_type='dense', neuron_type='SigmaDelta', threshold=np.ones([ 100])*thresholds[7],  act_fun='ReLU',  shape=[ 100], name='dns2', weight_tensor=weights[6][0], bias_tensor=weights[6][1], outputLayer=(self.layers['L8'],))
        self.layers['L6'] = layer(layer_type='dense', neuron_type='SigmaDelta', threshold=np.ones([1164])*thresholds[6],  act_fun='ReLU',  shape=[1164], name='dns1', weight_tensor=weights[5][0], bias_tensor=weights[5][1], outputLayer=(self.layers['L7'],))

        self.layers['L5'] = layer(layer_type='conv', pooling='stride', pooling_size=[1,1], padding='valid', is_flatten=True,  neuron_type='SigmaDelta', threshold=np.ones([64])*thresholds[5],  act_fun='ReLU',  shape=[ 1,18,64], name='cnv5', weight_tensor=weights[4][0], bias_tensor=weights[4][1], outputLayer=(self.layers['L6'],))
        self.layers['L4'] = layer(layer_type='conv', pooling='stride', pooling_size=[1,1], padding='valid', is_flatten=False, neuron_type='SigmaDelta', threshold=np.ones([64])*thresholds[4],  act_fun='ReLU',  shape=[ 3,20,64], name='cnv4', weight_tensor=weights[3][0], bias_tensor=weights[3][1], outputLayer=(self.layers['L5'],))
        self.layers['L3'] = layer(layer_type='conv', pooling='stride', pooling_size=[2,2], padding='valid', is_flatten=False, neuron_type='SigmaDelta', threshold=np.ones([48])*thresholds[3],  act_fun='ReLU',  shape=[ 5,22,48], name='cnv3', weight_tensor=weights[2][0], bias_tensor=weights[2][1], outputLayer=(self.layers['L4'],))
        self.layers['L2'] = layer(layer_type='conv', pooling='stride', pooling_size=[2,2], padding='valid', is_flatten=False, neuron_type='SigmaDelta', threshold=np.ones([36])*thresholds[2],  act_fun='ReLU',  shape=[14,47,36], name='cnv2', weight_tensor=weights[1][0], bias_tensor=weights[1][1], outputLayer=(self.layers['L3'],))
        self.layers['L1'] = layer(layer_type='conv', pooling='stride', pooling_size=[2,2], padding='valid', is_flatten=False, neuron_type='SigmaDelta', threshold=np.ones([24])*thresholds[1],  act_fun='ReLU',  shape=[31,98,24], name='cnv1', weight_tensor=weights[0][0], bias_tensor=weights[0][1], outputLayer=(self.layers['L2'],))
        self.layers['I']  = layer(layer_type='input', name='input', outputLayer=(self.layers['L1'],))
        
        return self.layers

    def composeLayersCoresMap(self):
        # Layer2Core mapping rules:  layer: list of (Core, Neurons, Cache-Miss-Rate). Be carefull for neuron ranges when you have paddings='Valid'
        # assigned neurons can be in the form of [range H, range W, range C] or only [range C]
        # Cache-Miss-Rate = [neurons, weights], for example [1.0, 0.0] means neurons are located into external memory, but weights are internal. 0<Cache-Miss-Rate<1
        # Assumption is that CMR won't affect timing (only affects energy) due to a proper pre-fetching mechanism
        # CI is reserved for input layer. CO is reserved for output layer. These cores are not contributing in energy
        # {layerObj : (CoreObj, [multidimentional neuron def])}
        self.layer_core_map =  { self.layers['I'] : ((self.sim.CI,[range(0,66), range(0,200), range(0,3)], [0.0, 0.0]),),

                            # L1 : shape=[31,98,24]
                            self.layers['L1']:( (self.sim.coresList[0][0],[range(  0, 31), range(  0, 98), range(  0, 24)], [0.0, 0.0]), ), 
                            # L2 : shape=[14,47,36]
                            self.layers['L2']:( (self.sim.coresList[1][0],[range(  0, 14), range(  0, 47), range(  0, 36)], [0.0, 0.0]), ),
                            # L3 : shape=[ 5,22,48]
                            self.layers['L3']:( (self.sim.coresList[2][0],[range(  0,  5), range(  0, 22), range(  0, 48)], [0.0, 0.0]), ),
                            # L4 : shape=[ 3,20,64]
                            self.layers['L4']:( (self.sim.coresList[3][0],[range(  0,  3), range(  0, 20), range(  0, 64)], [0.0, 0.0]), ), 
                            # L5 : shape=[ 1,18,64]
                            self.layers['L5']:( (self.sim.coresList[4][0],[range(  0,  1), range(  0, 18), range(  0, 64)], [0.0, 0.0]), ),
                            # L6 : shape=[1164]
                            self.layers['L6']:( (self.sim.coresList[5][0],[range(  0,1164)], [0.0, 0.0]), ),
                            # L7 : shape=[100]
                            self.layers['L7']:( (self.sim.coresList[6][0],[range(  0, 100)], [0.0, 0.0]), ),
                            # L8 : shape=[50]
                            self.layers['L8']:( (self.sim.coresList[7][0],[range(  0,  50)], [0.0, 0.0]), ),
                            # L9 : shape=[10]
                            self.layers['L9']:( (self.sim.coresList[8][0],[range(  0,  10)], [0.0, 0.0]), ),
                            # L10: shape=[1]
                            self.layers['L10']:( (self.sim.coresList[9][0],[range(  0, 1)], [0.0, 0.0]), ),
                            # L out
                            self.layers['O']  :( (self.sim.CO              ,[range(  0, 1)], [0.0, 0.0]), )}

        return self.layer_core_map
    def composeBusSegmentsList(self):
        segI = interconnect(master_cores=[self.sim.CI],                 slave_cores=self.sim.coresList[0][0:1], name='segI')
        seg1 = interconnect(master_cores=self.sim.coresList[0][0:1],    slave_cores=self.sim.coresList[1][0:1], name='seg1')
        seg2 = interconnect(master_cores=self.sim.coresList[1][0:1],    slave_cores=self.sim.coresList[2][0:1], name='seg2')
        seg3 = interconnect(master_cores=self.sim.coresList[2][0:1],    slave_cores=self.sim.coresList[3][0:1], name='seg3')
        seg4 = interconnect(master_cores=self.sim.coresList[3][0:1],    slave_cores=self.sim.coresList[4][0:1], name='seg4')
        seg5 = interconnect(master_cores=self.sim.coresList[4][0:1],    slave_cores=self.sim.coresList[5][0:1], name='seg5')
        seg6 = interconnect(master_cores=self.sim.coresList[5][0:1],    slave_cores=self.sim.coresList[6][0:1], name='seg6')
        seg7 = interconnect(master_cores=self.sim.coresList[6][0:1],    slave_cores=self.sim.coresList[7][0:1], name='seg7')
        seg8 = interconnect(master_cores=self.sim.coresList[7][0:1],    slave_cores=self.sim.coresList[8][0:1], name='seg8')
        seg9 = interconnect(master_cores=self.sim.coresList[8][0:1],    slave_cores=self.sim.coresList[9][0:1], name='seg9')
        segO = interconnect(master_cores=self.sim.coresList[9][0:1],    slave_cores=[self.sim.CO],    name='segO')
        self.interconnectList = [segI, seg1, seg2, seg3, seg4, seg5, seg6, seg7, seg8, seg9, segO]
        
        return self.interconnectList   






















