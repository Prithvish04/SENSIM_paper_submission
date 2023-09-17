import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import libraries.utils as utils
from skimage.io import imread_collection
import scipy.io


class PilotNetIO():
    def __init__(self, simObj, threshold_factor):
        self.sim = simObj
        self.fps = 0 # if FPS=0, there will be push back on input
        self.sim.plot_outputs = True
        self.num_samples = 1000
        self.last_sample_converted = 0
        self.input_frame_buffer = np.zeros([66,200,3])
        self.time_input_event = 0

        self.input_images = scipy.io.loadmat("DNNs/PilotNet/dataset/PilotNet_images_part1.mat")['images'][0:self.num_samples]
        self.threshold = threshold_factor * np.load('DNNs/PilotNet/keras/thresholds_delta.npy',  allow_pickle=True)[0]

        self.output_frame_buffer = 0
        if self.sim.plot_outputs:
            self.output_snapshot_file = open(self.sim.output_dir+"snapshots_output.csv", 'w') 
        else:
            self.output_snapshot_file = 0 
        return
        

    def doPlot(self):
        degree_factor = (180/np.pi)*2 # convert the output to degree
        time_axis = []
        out_membrane = []
        with open(self.sim.output_dir+"snapshots_output.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                time_axis.append(float(row[0])*1e3/self.sim.param.clk_freq)
                out_membrane.append(float(row[1])*degree_factor) 

        
        plt.plot(time_axis, out_membrane)
        plt.ylabel('sensim output value')
        plt.xlabel('time (ms)')
        plt.legend(['SENSIM_output'])
        plt.xticks(ticks=np.arange(0, time_axis[-1], time_axis[-1]/10))
        plt.grid(b=True, axis='x')
        plt.savefig(self.sim.output_dir+'output_sensim.png', bbox_inches='tight', dpi=1200)
        plt.clf()

        # Delta_PilotNet_output = (np.load('DNNs/PilotNet/keras/output_delta.npy')[0:self.num_samples])*degree_factor
        # plt.plot(np.arange(self.num_samples), Delta_PilotNet_output)
        # plt.ylabel('keras output value')
        # plt.xlabel('frame')
        # plt.legend(['Keras_output'])
        # plt.xticks(ticks=np.arange(0, self.num_samples, self.num_samples/10))
        # plt.grid(b=True, axis='x')
        # plt.savefig('./DNNs/PilotNet/output_sensim_keras.png', bbox_inches='tight', dpi=1200)
        # plt.close()

    def fetchWholeData(self):
        for i in range(self.num_samples):
            self.time_input_event = utils.frame2event(input_frame=self.input_images[i], time_between_events=10, shuffle_pixels=True, queue=self.sim.CI.getQueue(), input_layer_ind=utils.findInputLayerFromLayerMap(self.sim.layer_core_map).ID, delta=True, start_time=self.time_input_event, frame_buffer=self.input_frame_buffer, threshold=self.threshold, parameters=self.sim.param)
            print("Image ", str(i) , " conversion, CI output queue occupancy: ", self.sim.CI.getQueue()['out_queue_occupancy'].value, " end time:", self.time_input_event)
            #self.time_input_event = 200e6 # real time speed = self.sim.param.clk_freq/25fps

    def fetchData(self): 
        if self.sim.CI.getQueue()['out_queue_occupancy'].value<self.sim.param.queue_depths[0] and self.last_sample_converted<self.num_samples:
            pre_occ = self.sim.CI.getQueue()['out_queue_occupancy'].value
            if(self.fps!=0): self.time_input_event = np.max((self.last_sample_converted/self.fps)*self.sim.param.clk_freq, self.time_input_event)
            self.time_input_event = utils.frame2event(input_frame=self.input_images[self.last_sample_converted], time_between_events=10, shuffle_pixels=True, queue=self.sim.CI.getQueue(), input_layer_ind=utils.findInputLayerFromLayerMap(self.sim.layer_core_map).ID, delta=True, start_time=self.time_input_event, frame_buffer=self.input_frame_buffer, threshold=self.threshold, parameters=self.sim.param)
            print("\nImage ", str(self.last_sample_converted) ,'/',str(self.num_samples) , " converted, CI output queue occupancy: ", self.sim.CI.getQueue()['out_queue_occupancy'].value, " number of added flits: ", self.sim.CI.getQueue()['out_queue_occupancy'].value-pre_occ)
            self.last_sample_converted += 1
        return

    def dumpData(self, time):
        self.output_frame_buffer = utils.event2frame_pilotnet(queue=self.sim.CO.getQueue(), last_layer_ind=utils.findOutputLayerFromLayerMap(self.sim.layer_core_map).ID-1, frame_buffer=self.output_frame_buffer, parameters=self.sim.param, capture_intervals=self.sim.param.snapshot_log_time_step, snapshot_file=self.output_snapshot_file, time=time, capture_enable=self.sim.plot_outputs)
        return
    
    def close(self):
        if self.sim.plot_outputs:
            self.output_snapshot_file.close() 















