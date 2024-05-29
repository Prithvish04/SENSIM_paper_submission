import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import libraries.utils as utils
from skimage.io import imread_collection
import scipy.io
import os

from libraries.utils import DEBUG_SIM

class mnistIO():
    def __init__(self, simObj, outputDir, start_sample=0,num_samples=9, plot_outputs=True, mnist_num=None):
        self.sim = simObj
        self.start_sample = start_sample
        self.num_samples = num_samples
        self.outputDir = outputDir
        self.sim.fetchWholeDataSet = True
        self.sim.plot_outputs = True
        self.plot_outputs = plot_outputs
        self.mnist_num = mnist_num
        self.output_frame_buffer = np.zeros([10])
        if self.sim.plot_outputs:
            self.output_snapshot_file = open(os.path.join(self.outputDir, "output_snapshot.csv"), 'w')
        else:
            self.output_snapshot_file = 0 
        return

    def doPlot(self):
        if(self.plot_outputs):
            time_axis = []
            out_membrane = []
            
            with open(os.path.join(self.outputDir, "output_snapshot.csv")) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    time_axis.append(float(row[0])*1e3/self.sim.param.clk_freq)
                    out_membrane.append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10])])        
            plt.plot(time_axis, out_membrane)
            plt.ylabel('membrane potential')
            plt.xlabel('time (k cycles)')
            plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            plt.xticks(ticks=np.arange(0, time_axis[-1], time_axis[-1]/10))
            plt.grid(b=True, axis='x')
            # plt.savefig('./DNNs/4layer_CONV_MNIST/output_mnist.png', bbox_inches='tight')
            plt.savefig(os.path.join(self.outputDir, "output_sensim.eps"),  format="eps",bbox_inches='tight', dpi=1200)
            plt.close()

        # time_axis = []
        # energy = []
        # with open("gui/snapshots.csv") as csv_file:
        #     csv_reader = csv.reader(csv_file, delimiter=',')
        #     for row in csv_reader:
        #         if row[2]=='0' and row[3]=='2':
        #             time_axis.append(float(row[0]))
        #             energy.append(float(row[9]))
        # plt.plot(time_axis, energy)
        # plt.ylabel('energy core [0,2]')
        # plt.xlabel('time')
        # plt.savefig('./DNNs/4layer_CONV_MNIST/core02_energy.png', bbox_inches='tight')


    def fetchWholeData(self):
        input_frame_buffer = np.zeros([28,28])
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        test_images = test_images/255
        # print(np.shape(test_labels[test_labels == 5]))
        if self.mnist_num != None:
            test_images = test_images[test_labels == self.mnist_num]
        # print(np.shape(test_images))
        if self.num_samples>1: 
            # send zero for last image (during debug with only 1 sample, I don't want blank image at the end)
            test_images[self.num_samples] = test_images[self.num_samples]*0 
            self.num_samples += 1
        for i in range(self.num_samples):
            time = 100e3*i
            time = utils.frame2event(input_frame=test_images[i], time_between_events=100, shuffle_pixels=True, queue=self.sim.CI.getQueue(), input_layer_ind=utils.findInputLayerFromLayerMap(self.sim.layer_core_map).ID, delta=True, start_time=time, frame_buffer=input_frame_buffer, threshold=0, parameters=self.sim.param)
            # print(test_labels[i], end=" ")
            # print("")
    
    def fetchData(self):        
        return

    def dumpData(self, time):
        utils.event2frame_mnist(queue=self.sim.CO.getQueue(), last_layer_ind=utils.findOutputLayerFromLayerMap(self.sim.layer_core_map).ID-1, frame_buffer=self.output_frame_buffer, parameters=self.sim.param, capture_intervals=self.sim.param.evaluation_time_step, snapshot_file=self.output_snapshot_file, time=time, capture_enable=self.sim.plot_outputs)
        return
    
    def close(self):
        if self.sim.plot_outputs:
            self.output_snapshot_file.close() 



class PilotNetIO():
    def __init__(self, simObj, outputDir, start_sample=0,num_samples=500, plot_outputs=True):
        self.sim = simObj
        self.fps = 0 # if FPS=0, there will be push back on input
        self.sim.plot_outputs = True
        self.plot_outputs = plot_outputs
        self.start_sample = start_sample
        self.num_samples = num_samples
        self.last_sample_converted = 0
        self.input_frame_buffer = np.zeros([66,200,3])
        self.time_input_event = 0

        self.input_images = scipy.io.loadmat('./DNNs/PilotNet/dataset/Pilot_0_250.mat')['images'][self.start_sample:self.num_samples]
        self.threshold = np.load('./DNNs/PilotNet/keras/thresholds_delta.npy',  allow_pickle=True)[0]
        
        self.outputDir = outputDir
        self.output_frame_buffer = 0

        if self.sim.plot_outputs:
            self.output_snapshot_file = open(os.path.join(self.outputDir,"output_snapshot.csv"), 'w') 
        else:
            self.output_snapshot_file = 0 
        return

    def doPlot(self):
        if(self.plot_outputs):
            degree_factor = (180/np.pi)*2 # convert the output to degree
            time_axis = []
            out_membrane = []
            try:
                with open(os.path.join(self.outputDir, "output_snapshot.csv")) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for row in csv_reader:
                        time_axis.append(float(row[0])*1e3/self.sim.param.clk_freq)
                        out_membrane.append(float(row[1])*degree_factor) 
            except FileNotFoundError:
                print(f'file output_snapshot.csv not found at {os.path.join(self.outputDir)}')
            
            Delta_PilotNet_output = (np.load('DNNs/PilotNet/keras/output_delta.npy')[self.start_sample:self.num_samples])*degree_factor
            
            plt.plot(time_axis, out_membrane)
            plt.ylabel('sensim output value')
            plt.xlabel('time (ms)')
            plt.legend(['SENSIM_output'])
            plt.xticks(ticks=np.arange(0, time_axis[-1], time_axis[-1]/10))
            plt.grid(b=True, axis='x')
            #plt.savefig('./DNNs/PilotNet/output_sensim.png', bbox_inches='tight', dpi=1200)
            plt.savefig(os.path.join(self.outputDir, "output_sensim.eps"),  format="eps",bbox_inches='tight', dpi=1200)
            plt.clf()
            plt.cla()
            plt.close('all')
            

            plt.plot(np.arange(self.start_sample,self.num_samples), Delta_PilotNet_output)
            plt.ylabel('keras output value')
            plt.xlabel('frame')
            plt.legend(['Keras_output'])
            plt.xticks(ticks=np.arange(self.start_sample, self.num_samples, (self.num_samples- self.start_sample)/10))
            plt.grid(b=True, axis='x')
            #plt.savefig('./DNNs/PilotNet/output_sensim_keras.png', bbox_inches='tight', dpi=1200)
            plt.savefig(os.path.join(self.outputDir, "output_sensim_keras.eps"), format="eps", bbox_inches='tight', dpi=1200)
            plt.clf()
            plt.cla()
            plt.close('all')

        # time_axis = []
        # energy = []
        # with open("gui/snapshots.csv") as csv_file:
        #     csv_reader = csv.reader(csv_file, delimiter=',')
        #     for row in csv_reader:
        #         if row[2]=='0' and row[3]=='2':
        #             time_axis.append(float(row[0]))
        #             energy.append(float(row[9]))
        # plt.plot(time_axis, energy)
        # plt.ylabel('energy core [0,2]')
        # plt.xlabel('time')
        # plt.savefig('./DNNs/4layer_CONV_MNIST/core02_energy.png', bbox_inches='tight')


    def fetchWholeData(self):
        for i in range(self.start_sample, self.num_samples):
            self.time_input_event = utils.frame2event(input_frame=self.input_images[i], time_between_events=10, shuffle_pixels=True, queue=self.sim.CI.getQueue(), input_layer_ind=utils.findInputLayerFromLayerMap(self.sim.layer_core_map).ID, delta=True, start_time=self.time_input_event, frame_buffer=self.input_frame_buffer, threshold=self.threshold, parameters=self.sim.param)
            if(DEBUG_SIM):
                print("Image ", str(i) , " conversion, CI output queue occupancy: ", self.sim.CI.getQueue()['out_queue_occupancy'].value, " end time:", self.time_input_event)
            #self.time_input_event = 200e6 # real time speed = self.sim.param.clk_freq/25fps

    def fetchData(self):
        #inject new frame
        if self.sim.CI.getQueue()['out_queue_occupancy'].value<self.sim.param.queue_depths[0] and self.last_sample_converted<(self.num_samples- self.start_sample):
            pre_occ = self.sim.CI.getQueue()['out_queue_occupancy'].value
            if(self.fps!=0): 
                self.time_input_event = max(int((self.last_sample_converted/self.fps)* self.sim.param.clk_freq), int(self.time_input_event))
            self.time_input_event = utils.frame2event(input_frame=self.input_images[self.last_sample_converted], time_between_events=10, shuffle_pixels=True, queue=self.sim.CI.getQueue(), input_layer_ind=utils.findInputLayerFromLayerMap(self.sim.layer_core_map).ID, delta=True, start_time=self.time_input_event, frame_buffer=self.input_frame_buffer, threshold=self.threshold, parameters=self.sim.param)
            if(DEBUG_SIM):
                print("\nImage ", str(self.last_sample_converted) ,'/',str(self.num_samples - self.start_sample) , " converted, CI output queue occupancy: ", self.sim.CI.getQueue()['out_queue_occupancy'].value, " number of added flits: ", self.sim.CI.getQueue()['out_queue_occupancy'].value-pre_occ)
            self.last_sample_converted += 1
            return 2
        #Queue not emply
        elif self.sim.CI.getQueue()['out_queue_occupancy'].value>0:
            return 1
        #Finished data injection 
        elif self.sim.CI.getQueue()['out_queue_occupancy'].value==0:
            return 0
        else:
            raise Exception("IO fetch data is in unknown state")

    def dumpData(self, time):
        self.output_frame_buffer = utils.event2frame_pilotnet(queue=self.sim.CO.getQueue(), last_layer_ind=utils.findOutputLayerFromLayerMap(self.sim.layer_core_map).ID-1, frame_buffer=self.output_frame_buffer, parameters=self.sim.param, capture_intervals=self.sim.param.snapshot_log_time_step, snapshot_file=self.output_snapshot_file, time=time, capture_enable=self.sim.plot_outputs)
        return
    
    def close(self):
        if self.sim.plot_outputs:
            self.output_snapshot_file.close() 















