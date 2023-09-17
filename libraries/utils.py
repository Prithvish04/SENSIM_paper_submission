import json
import numpy as np
from multiprocessing import Value

from numpy.core.numeric import NaN
from libraries.Queue import Queue as Queue

DEBUG_SIM = True

def ReadSettings(fileName):
    f = open(fileName)
    data = json.load(f)
    f.close()
    return data

def LoadSimulatorSettings():
    f = ReadSettings()
    data = json.load(f)
    f.close()
    return data

def WriteSettings(fileName, data):
    with open(fileName, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def NewSettingItem():
    setting = dict()
    setting['value'] = "empty"
    setting['gui'] = dict()
    return setting

def NewHeaderValue(header, size, dataList):
    setting = dict()
    setting['values'] = dict()
    setting['gui'] = dict()
    setting['gui']['type'] = 'header-value'
    setting['gui']['size'] = str(size)
    setting['gui']['header'] = header
    
    dummyCnt = 0
    color = 0
    oldFirstValue = ""
    for valuesList in dataList:
        if(not (oldFirstValue == valuesList[0])):
            color = (color+1)%2
        
        oldFirstValue = valuesList[0]

        newitem = NewSettingItem()        
        newitem['gui']['size'] = str(size)
        newitem['gui']['color'] = str(color)
        newitem['value'] = valuesList
        setting['values']["item%d"%dummyCnt] = newitem

        dummyCnt+= 1
    
    return setting

def findInputLayerFromLayerMap(layer_core_map):
    for layer in layer_core_map:
        if layer.layer_type == "input":
            return layer

def findOutputLayerFromLayerMap(layer_core_map):
    for layer in layer_core_map:
        if layer.layer_type == "output":
            return layer

def NrFilitsToMemorySize(NrFilits):
    # TODO For amir: fix this function 
    return  NrFilits * 10

def MemorySizeToNrFilits(MemSize):
    # TODO For amir: fix this function 
    return  MemSize / 10

def createEmptyQueueObj(sizeI, sizeO):
    return {'in_event_queue': Queue("inputQ", sizeI), 
            'in_queue_occupancy' : Value("i", 0, lock=False), 
            'out_event_queue' :  Queue("outputQ", sizeO), 
            'out_queue_occupancy' : Value("i", 0, lock=False), 
            'in_queue_occupancy_peak' : Value("i", 0, lock=False),
            'out_queue_occupancy_peak' : Value("i", 0, lock=False), 
            'packet_loss' : Value("i", 0, lock=False)}

def frame2event(input_frame=None, time_between_events=0, shuffle_pixels=False, queue=None, input_layer_ind=None, delta=False, start_time=0, frame_buffer=None, threshold=0, parameters=None):
    '''
    Convert input frame to events
        input_frame: a 3D or 2D frame tensor which should be converted to events
        time_between_events: add delay between events to reduce the rate
        shuffle_pixels: shuffle the frame pixels to not have the top->down order
        queue: the input core object which the events will be inijected to its output event queue
        delta: generate delta events (diffrent between input tensor and frame-buffer)
        frame_buffer, threshold: used for the case of Delta events, one threshold per input channel
        parameters: parameter file to find the format of events, ....
    '''
    # At this moment for 1D or 2D input shape
    time_stamp=start_time

    if delta:
        inp = quantize(input_frame,threshold) - quantize(frame_buffer,threshold)
        frame_buffer += inp
    else:
        inp = input_frame



    if len(inp.shape)==1: # flatten input 
        nz_pix = np.nonzero(inp)
        order = np.arange(len(nz_pix[0]))
        if shuffle_pixels:
            np.random.seed(len(nz_pix[0]))
            np.random.shuffle(order)   
        output_event = [time_stamp, input_layer_ind, [0,0]]
        for pixel in order:
            address = nz_pix[0][pixel]
            output_event.append([address, inp[address]])
            if parameters.cnt_event_flit(output_event)>=parameters.max_event_flits:                                
                queue['out_event_queue'].put(output_event) #event format:(time-stamp, Source Layer, [0,0], [C,Value])
                queue['out_queue_occupancy'].value += parameters.cnt_event_flit(output_event)
                time_stamp += time_between_events 
                output_event = [time_stamp, input_layer_ind, [0,0]]                
        if len(output_event)>3: 
            queue['out_event_queue'].put(output_event)
            queue['out_queue_occupancy'].value += parameters.cnt_event_flit(output_event)
            time_stamp += time_between_events
    elif len(inp.shape)==2: # 2D input GrayScale (1channel) --> events with only single spike
        nz_pix = np.nonzero(inp)
        order = np.arange(len(nz_pix[0]))
        if shuffle_pixels:
            np.random.seed(len(nz_pix[0]))
            np.random.shuffle(order)
        for pixel in order:
            [H,W] = [nz_pix[0][pixel],nz_pix[1][pixel]]
            output_event = [time_stamp, input_layer_ind, [H,W], [0,inp[H,W]]]   # event format:(time-stamp, Source Layer, [H,W], [C,Value])
            queue['out_event_queue'].put(output_event)    
            queue['out_queue_occupancy'].value += int(parameters.cnt_event_flit(output_event))
            time_stamp += time_between_events 
    elif len(inp.shape)==3: # 3D input RGB (3channel)
        inp_2D = np.sum(inp,axis=2)
        nz_pix = np.nonzero(inp_2D)
        order = np.arange(len(nz_pix[0]))
        if shuffle_pixels:
            np.random.seed(len(nz_pix[0]))
            np.random.shuffle(order)
        for pixel in order:
            [H,W] = [nz_pix[0][pixel],nz_pix[1][pixel]]
            if (parameters.max_event_flits-parameters.header_flits)*parameters.spike_per_flits >=3:
                output_event = [time_stamp, input_layer_ind, [H,W]]   # event format:(time-stamp, Source Layer, [H,W], [C,Value])
                if inp[H,W,0]!=0: output_event.append([0,inp[H,W,0]])
                if inp[H,W,1]!=0: output_event.append([1,inp[H,W,1]])
                if inp[H,W,2]!=0: output_event.append([2,inp[H,W,2]])
                queue['out_event_queue'].put(output_event)    
                queue['out_queue_occupancy'].value += int(parameters.cnt_event_flit(output_event))
                time_stamp += time_between_events 
            else:
                if inp[H,W,0]!=0: 
                    output_event = [time_stamp, input_layer_ind, [H,W], [0,inp[H,W,0]]]
                    queue['out_event_queue'].put(output_event)    
                    queue['out_queue_occupancy'].value += int(parameters.cnt_event_flit(output_event))
                    time_stamp += time_between_events 
                if inp[H,W,1]!=0: 
                    output_event = [time_stamp, input_layer_ind, [H,W], [1,inp[H,W,1]]]
                    queue['out_event_queue'].put(output_event)    
                    queue['out_queue_occupancy'].value += int(parameters.cnt_event_flit(output_event))
                    time_stamp += time_between_events 
                if inp[H,W,2]!=0: 
                    output_event = [time_stamp, input_layer_ind, [H,W], [2,inp[H,W,2]]]
                    queue['out_event_queue'].put(output_event)    
                    queue['out_queue_occupancy'].value += int(parameters.cnt_event_flit(output_event))
                    time_stamp += time_between_events
    else:
        raise Exception("not implemented")
    
    return time_stamp

def quantize(x, q):
    if q==0:
        return x
    else:
        return np.round(x/q)*q

def event2frame_mnist(queue, last_layer_ind, frame_buffer, parameters, capture_intervals, snapshot_file, time, capture_enable):
    '''
    Convert events into frame by simple accumulation
    '''
    if queue['in_event_queue']:
        while queue['in_event_queue'].isNotEmpty():
            ev = queue['in_event_queue'].get(False)
            if ev[1]==last_layer_ind: #filter events [time_stamp, input_layer_ind, [H,W], [0,inp[H,W]]]
                for spike in ev[3:]:
                    frame_buffer[spike[0]] += spike[1]
            queue['in_queue_occupancy'].value -= parameters.cnt_event_flit(ev)
            queue['in_event_queue'].get()
        if (time%capture_intervals == 0) and capture_enable:
            snapshot_file.write(str(time))
            for neuron in frame_buffer:
                snapshot_file.write(','+str(neuron))
            snapshot_file.write("\n")
        return 1
    else:
        return 0

def event2frame_pilotnet(queue, last_layer_ind, frame_buffer, parameters, capture_intervals, snapshot_file, time, capture_enable):
    '''
    Convert events into frame by simple accumulation
    '''
    if queue['in_event_queue']:
        while queue['in_event_queue'].isNotEmpty():
            ev = queue['in_event_queue'].get(False)
            if ev[1]==last_layer_ind: # filter events [time_stamp, input_layer_ind, [H,W], [0,inp[H,W]]]
                frame_buffer += ev[3][1]
            queue['in_queue_occupancy'].value -= parameters.cnt_event_flit(ev)
            queue['in_event_queue'].get()
        if (time%capture_intervals == 0) and capture_enable:
            snapshot_file.write(str(time))
            snapshot_file.write(','+str(frame_buffer))
            snapshot_file.write("\n")
    return frame_buffer

def event2frame_xray(queue, last_layer_ind, frame_buffer, parameters):
    '''
    Convert events into frame by simple accumulation
    '''
    if queue['in_event_queue']:
        while queue['in_event_queue'].isNotEmpty():
            ev = queue['in_event_queue'].get(False)
            if ev[1]==last_layer_ind: #filter events [time_stamp, input_layer_ind, [H,W], [0,inp[H,W]]]
                frame_buffer[ev[2][0],ev[2][1]] += ev[3][1]
            queue['in_queue_occupancy'].value -= parameters.cnt_event_flit(ev)
            queue['in_event_queue'].get()
        return 1
    else:
        return 0


def gui_setting_file(core_layer_map, mesh_size, file_name="setting.csv"):
    '''
    mesh_size = [row_size, col_size]
    '''
    setting_file = open(file_name, 'w')
    setting_file.write("#type,number_of_rows,number_of_cols\n")
    setting_file.write('mesh,'+str(mesh_size[0]) + ',' + str(mesh_size[1]) + ',1\n')

    setting_file.write("#core,core_x,core_y,core_name,num_mapped_layers,layer_name1,layer_name2,...\n")
    for core, layer_neurons_list in core_layer_map.items():
        if core.loc[0]<0 or core.loc[0]>=mesh_size[0]: continue
        if core.loc[1]<0 or core.loc[1]>=mesh_size[1]: continue
        setting_file.write('core,'+str(core.loc[0]) + ',' + str(core.loc[1]) + ',' + core.name + ',' + str(len(layer_neurons_list)))
        for layer_neurons in layer_neurons_list:
            layer = layer_neurons[0]
            setting_file.write(',' + layer.name)
        setting_file.write("\n")
    
    setting_file.close()
    return 

class snapshot:
    def __init__(self, core_list, bus_list, capture_intervals, max_num_captures, reset_states, energy_max=1e7, file_name="snapshot"):
        self.snapshot_id = 0
        self.core_list = core_list
        self.bus_list = bus_list
        self.file = open(file_name+"_cores.csv", 'w')

        #importants
        self.file.write('#snapshotID,core_name,time(cc),internal_time,peak_in_queue,peak_out_queue,packet_loss,energy_total,idle_time(cc),')
        #who cares
        self.file.write('processor_utilization(ratio),in_queue_occupancy,out_queue_occupancy,energy_controller,energy_npe,energy_dmem,energy_fifo,core_x,core_y,')
        self.file.write('energy_total(ratio),peak_in_queue_occupancy(ratio),peak_out_queue_occupancy(ratio),snapshot_time\n')

        self.file2 = open(file_name+"_interconnects.csv", 'w')
        self.file2.write('#snapshotID,time(cc),name,total_flit,energy_bus,internal_time\n')
        self.capture_intervals = capture_intervals
        self.last_capture_time = 0
        self.max_num_captures = max_num_captures
        self.reset_states=reset_states
        self.energy_max = energy_max
        return 

    def capture(self, time):
        time_diff = (time-self.last_capture_time)
        if time_diff>self.capture_intervals and (self.snapshot_id<=self.max_num_captures):
            for core in self.core_list:
                if core.queue['out_queue_occupancy_peak'].value < core.queue['out_queue_occupancy'].value:
                    # This may happend because interconnects execute before cores
                    core.queue['out_queue_occupancy_peak'].value = core.queue['out_queue_occupancy'].value

                #snapshotID,core_name,time(cc),internal_time,peak_in_queue_occupancy
                self.file.write(str(self.snapshot_id)+','+core.name+','+str(time)+','+str(int(core.time.value))+','+str(int(core.queue['in_queue_occupancy_peak'].value)))
                #peak_out_queue_occupancy,total_packet_loss,energy_total,idle_time(cc)
                self.file.write(','+str(int(core.queue['out_queue_occupancy_peak'].value))+','+str(int(core.queue['packet_loss'].value))+','+str(int(core.energy['total'].value))+','+str(int(core.idle_times.value)))
                #processor_utilization(ratio),in_queue_occupancy,out_queue_occupancy
                self.file.write(','+str(1.0-core.idle_times.value/time_diff)+','+str(int(core.queue['in_queue_occupancy'].value))+','+str(int(core.queue['out_queue_occupancy'].value)))
                #energy_controller,energy_npe,energy_dmem,energy_fifo
                self.file.write(','+str(int(core.energy['CON'].value))+','+str(int(core.energy['NPE'].value))+','+str(int(core.energy['DMEM'].value))+','+str(int(core.energy['FIFO'].value)))
                #core_x,core_y
                self.file.write(','+str(core.loc[0])+','+str(core.loc[1]))
                #energy_total(ratio),peak_in_queue_occupancy(ratio),peak_out_queue_occupancy(ratio),snapshot_time
                self.file.write(','+str(core.energy['total'].value/self.energy_max)+','+str(core.queue['in_queue_occupancy_peak'].value/core.input_queue_depth)+','+str(np.minimum(core.queue['out_queue_occupancy_peak'].value/core.output_queue_depth,1.0))+','+str(time_diff)+'\n')
                
                #print (core.queue['out_event_queue'])
                if self.reset_states:
                    core.energy['CON'].value = 0
                    core.energy['NPE'].value = 0
                    core.energy['DMEM'].value = 0
                    core.energy['FIFO'].value = 0
                    core.energy['total'].value = 0
                    core.idle_times.value = 0
                    core.queue['in_queue_occupancy_peak'].value=0
                    core.queue['out_queue_occupancy_peak'].value=0
                    core.queue['packet_loss'].value=0
            for bus in self.bus_list:
                #snapshotID,time(cc),name,energy_bus,internal_time, total_flit
                self.file2.write(str(self.snapshot_id)+','+str(int(time))+','+bus.name+','+str(int(bus.total_flits.value))+','+str(int(bus.energy.value))+','+str(int(bus.time.value))+'\n')
                # print("interconnect: ", bus.name, " num flits: ", bus.total_flits)
                if self.reset_states: bus.reset_bus()
            self.last_capture_time = time

        self.snapshot_id += 1

        return

    def file_close(self):
        self.file.close()
        self.file2.close()
        return 

