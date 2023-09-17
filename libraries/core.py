import numpy as np
import skimage.measure
from multiprocessing import Value
import libraries.utils as utils

class core:
    def __init__(self, name=None, loc=None, queue_depths=[2048,2048], N_NPE=4):
        '''
        name: optional name of the core
        loc: 3-D locations of the core [x,y,z]
        queue_depths: [input queue depth, output queue depth] --> capacity based number of flits not number of events
        N_NPE: number of NPE units in the core
        '''
        self.name = name
        self.loc = loc

        # Event-queue depth
        self.input_queue_depth = queue_depths[0]
        self.output_queue_depth = queue_depths[1]

        # number of NPEs
        self.N_NPE = N_NPE
        
        queue = utils.createEmptyQueueObj(utils.NrFilitsToMemorySize(self.input_queue_depth), utils.NrFilitsToMemorySize(self.output_queue_depth))
        queue['in_event_queue'].setParentName(name)
        queue['out_event_queue'].setParentName(name)
        queue['in_event_queue'].setLocation(loc)
        queue['out_event_queue'].setLocation(loc)

        self.queue = queue
        self.time = Value("d", 0, lock=False)
        self.idle_times = Value("d", 0, lock=False)
        self.energy = {
                'CON'   : Value("d", 0, lock=False),
                'NPE'   : Value("d", 0, lock=False),
                'DMEM'  : Value("d", 0, lock=False),
                'FIFO'  : Value("d", 0, lock=False),
                'total' : Value("d", 0, lock=False),
        }
               
        self.reset()
    
    def getQueue(self):
        return self.queue

    def reset(self):
        # event_queue contains a list of events [ev0, ev1,...]     
        # Event format defines in parameters.event_format       

        # Current time of the core can be ahead of the simulation current time.
        # When self.time > simulation_current_time it means core is busy for now
        self.time.value = 0

        #energy consuption details [ALU, local memory, event queue]
        self.energy['CON'].value = 0
        self.energy['NPE'].value = 0
        self.energy['DMEM'].value = 0
        self.energy['FIFO'].value = 0
        self.energy['total'].value = 0

        # last evaluation time for time-step inference
        self.last_eval_time = 0
        self.last_intruppt_trigger = 0
        self.timer_interrupt = 0
        #logging the start of processing time
        self.start_time=-1e3
        #logging total idle times
        self.idle_times.value = 0
        #internal output queue which can expand
        self.internal_output_queue = []
    
    def setMappedPackage(self, mappedPackage, layerList, parameters=None):
        '''
        Build the core by compiling the mapping dictionary (extract source layers, source cores, destination layers and destination cores)

        mappedPackage holds the neurons and layers mapped into a core
            example: (CMR is Cache Missed Rate)
                [(L1,range(0,600), CMR), (L2,range(600,1000), CMR)]
            It is possible to map a layer two times in one core like: [(L1,range(0,600), CMR1), (L1,range(800,1000), CMR2)]
            Range of neurons can be [range H, range W, range C] or [range C]
        '''

        self.mappedPackage=mappedPackage
        self.layerList = layerList

        # mapped_layers (a list of the mapped layers + neurons in this core)
        self.mappedLayers = []
        self.neurons= dict()
        self.CMR= dict()
        self.neurons_3D= dict()
        for item in mappedPackage: #[(layer, neuron ranges, CMR), (layer, neuron ranges, CMR)]
            self.mappedLayers.append(item[0])   
            if correct_mapping_in_range(layer=item[0], neuron_ranges=item[1])==False: 
                print("WARNINIG: Mapping of "+item[0].name+" into core "+self.name+" is modified to be in range, since you are neurons that does not exist (probably due to Valid padding)")
            self.neurons.setdefault(item[0], item[1]) #(L,neuron range)
            self.CMR.setdefault(item[0], item[2]) #(L, CMR)
            # making a 3D mapping mask for neurons
            if len(item[1])==1:   
                mapped_neurons = np.zeros([1,1,item[0].shape[0]])
                ranges = ([0],[0],item[1])
            elif len(item[1])==3: 
                mapped_neurons = np.zeros(item[0].shape)
                ranges = item[1]
            for x in ranges[0]:
                for y in ranges[1]:
                    mapped_neurons[x,y,ranges[2]]=True

            self.neurons_3D.setdefault(item[0], mapped_neurons.astype('bool'))

        # update_flag: 1 bit per each [H,W] position in each layer to indicate if the neurons in this position are updated during previous time step (only used in TimeStep_flag sync mode)
        # at the beginning all are 1, to propagate the bias value
        self.update_flag = dict()
        for layer in self.mappedLayers:
            if len(layer.shape)==1:   
                self.update_flag.setdefault(layer, True)
            elif len(layer.shape)==3: 
                self.update_flag.setdefault(layer, np.ones(layer.neuron_states.shape[0:2]).astype('bool'))
            else: 
                raise Exception("layer shape for ", layer.name, " is neither 1D (for dense) nor 3D (for conv)")
            if parameters.suppress_bias_wave:
                self.update_flag[layer] = self.update_flag[layer] * False
    
    
    #def process_callback(self, time=None, parameters=None):
    def process_callback(self, time=None, parameters=None):
        '''
        Call this process for normal cores (not input/output) once in each time-step
            Timer interup: (for TimeStep sync type)
            - loop over all neurons to evaluate them

            Event Intrupt:
            - Loop over input event (no event=no process)
                Event format -> in parameter file
            - Consume events from the input_event_queue (remove it from the list)
            - Continue until time_stamp = time + time_step

        time: current time of the simulator (in the unit of clock period) 
        '''
        end_time = time + parameters.time_step
        if self.queue['in_queue_occupancy_peak'].value < self.queue['in_queue_occupancy'].value :
            self.queue['in_queue_occupancy_peak'].value = self.queue['in_queue_occupancy'].value 

        core_idle(core=self, time=time)
        if len(self.internal_output_queue)>0: # If len(self.internal_output_queue)>0 the core cannot generate new events but still can process incomming events
            while self.internal_output_queue:
                evt = self.internal_output_queue[0]
                out_queue_full = ( self.queue['out_queue_occupancy'].value + parameters.cnt_event_flit(evt) ) >=  self.output_queue_depth
                if out_queue_full: 
                    break
                self.queue['out_event_queue'].put(evt.copy())
                self.energy['FIFO'].value += parameters.Efifo_wr*parameters.cnt_event_flit(evt)*parameters.flit_width #event write
                self.queue['out_queue_occupancy'].value  += parameters.cnt_event_flit(evt)
                self.energy['CON'].value += parameters.E_CON * parameters.cnt_event_flit(evt) # send flits
                self.time.value += parameters.cnt_event_flit(evt)
                self.internal_output_queue.pop(0)

        
        # Set timer interrupt
        if (parameters.sync_type=='TimeStep' or parameters.sync_type=='TimeStep_flag') and (time - self.last_intruppt_trigger)>=parameters.evaluation_time_step:
            # Print missed interrupt warnings:
            # if self.timer_interrupt and len(self.internal_output_queue)>0:
            #     print("\nWARNINIG: Core "+ self.name +" missed a timer interrupt service. Core output queue is full. Last eval time: "+'{:2E}'.format(self.last_eval_time)+". Internal queue occupancy:", str(len(self.internal_output_queue)))
            # elif self.timer_interrupt and parameters.consume_then_fire==True and self.queue['in_event_queue'].isNotEmpty():
            #     print("\nWARNINIG: Core "+ self.name +" missed a timer interrupt service. Core input queue is not empty. Last eval time: "+'{:2E}'.format(self.last_eval_time)+". Input queue occupancy:", str(self.queue['in_queue_occupancy'].value))
            # elif self.timer_interrupt:    
            #     print("\nWARNINIG: Core "+ self.name +" missed a timer interrupt service. Core is overloaded. Last eval time: "+str(self.last_eval_time)+ ". Internal time:"+str(self.time.value))
            self.last_intruppt_trigger = time
            self.timer_interrupt = 1

        # evalute all neurons for TimeStep syncronization type
        if self.timer_interrupt and self.time.value<end_time and len(self.internal_output_queue)==0 and (parameters.consume_then_fire==False or self.queue['in_event_queue'].isEmpty()): 
            self.timer_interrupt = 0 
            self.last_eval_time = time  
            # Loop over implemented layers to evaluate all the neurons in this core
            for L in self.mappedLayers:
                layer_process(core=self, layer=L, parameters=parameters, event='eval') 

        while self.queue['in_event_queue'].isNotEmpty() and self.time.value<end_time: #There is an event and Core is free
            ev = self.queue['in_event_queue'].get(False) # This is a complete Event (maybe multiple flits)
            # check if we should stop for this time-step 
            fifo_pass_time   = parameters.Tfifo #Tfifo is the minimum time spent in the fifo (pass-though fifo)
            src_layer  = self.layerList[int(ev[1])]
            # if either of neurons or weights are outside, add one time initial access latency
            for L in src_layer.output_layers:
                if L not in self.mappedLayers: 
                    continue
                if self.CMR[L][0]>0 or self.CMR[L][1]>0: 
                    fifo_pass_time += parameters.Text_mem
                    break

            if (ev[0]+fifo_pass_time)>end_time: 
                break # done for this time-step
            #advance time_stamp by the minimum time that event needs to spent in the fifo
            ev[0]   = ev[0]+fifo_pass_time 
            
            # check if the src layer targets any implemented layer in this core
            if not intersection(src_layer.output_layers, self.mappedLayers):
                print("WARNINIG: an event  with source layer of "+ src_layer.name +" reached to "+ self.name +"(you have feedback?)")
            
            #Add FIFO "write+read" energy 
            self.energy['FIFO'].value += parameters.Efifo_rd*parameters.cnt_event_flit(ev)*parameters.flit_width #event read
            self.queue['in_queue_occupancy'].value -= parameters.cnt_event_flit(ev)
            if (self.queue['in_queue_occupancy'].value < 0) :
                self.queue['in_event_queue'].printYourName()
                raise Exception("input_queue_occupancy is negative!!!") 

            #Considering the core idle time
            core_idle(core=self, time=ev[0])    

            for L in src_layer.output_layers:
                # Skip the L if it is not in this core
                if L not in self.mappedLayers: 
                    continue

                # Retrieve pointers to the weights and neuron states 
                self.energy['CON'].value += parameters.PAI*(parameters.E_CON)
                self.time.value += parameters.PAI

                # Perform the acctual neuron updates
                layer_process(core=self, layer=L, parameters=parameters, event=ev)

            # clear the event from the queue
            self.queue['in_event_queue'].get()

            #logging the start of processing time (first event processed)
            if self.start_time<0: 
                self.start_time = end_time 
        # update total energy consumption in this time step
        self.energy['total'].value = self.energy['CON'].value + self.energy['NPE'].value + self.energy['DMEM'].value + self.energy['FIFO'].value

    def dense_layer_process_delta(self, event, parameters, target_layer):
        '''
        Dense layer with sigma-delta neuron model 
        Process: 
            1- Read weights
            2- Read neuron states
            3- Calculate new neuron states (state_new)
            4- Evaluate outputs delta = quantize(f(new_state)) - quantize(f(old_state))
            5- Update states
            6- Send non-zero events out 
        '''
        
        spikes = event[3:]
        L = target_layer
        N = self.neurons[L][0]
        if event[2]!=[0,0]: 
            raise Exception("input to layer "+L.name+" is not properly flatten")
        
        for spike in spikes: 
            C = spike[0]   
            value= spike[1]
            weights = L.weights[C,N]
            L.neuron_states[N] += weights*value


            # steps 1,2,3 : W*V multiplication + Addition  TODO: adjust precision
            
        
        E_neuron_rd = (1-self.CMR[L][0])*parameters.EDmem_rd + self.CMR[L][0]*parameters.Eext_mem_rd
        E_neuron_wr = (1-self.CMR[L][0])*parameters.EDmem_wr + self.CMR[L][0]*parameters.Eext_mem_wr
        E_weight_rd = (1-self.CMR[L][1])*parameters.EDmem_rd + self.CMR[L][1]*parameters.Eext_mem_rd

        compute_time = len(spikes)*np.ceil(len(weights)/self.N_NPE)*parameters.T_NPE*2
        # If either of neurons or weights are outside we need external memory access
        # For external memory we assume enough BW is available to read/write 1 word (32b) per cycle
        ext_mem_time = (self.CMR[L][0] * len(N) * 2 * parameters.bitwidths['States'])/parameters.BWext_mem + (self.CMR[L][1] * len(weights) * len(spikes) * parameters.bitwidths['Weights'])/parameters.BWext_mem
        self.time.value += np.max([ext_mem_time, compute_time])  # memory access and compute runs in parallel  

        # energy for reading weights [read all the weights once for each spike]
        self.energy['DMEM'].value+= len(weights) * len(spikes) * E_weight_rd * parameters.bitwidths['Weights']
        # energy for read/write states [read/write all the states once for each few of spike --> depends on the number of spike registers in the controller]
        self.energy['DMEM'].value+= len(N) * np.ceil(len(spikes)/parameters.Spike_Register) * (E_neuron_rd + E_neuron_wr) * parameters.bitwidths['States']
        # energy for NPE and controller
        self.energy['NPE'].value += len(N) * len(spikes) * 2 * parameters.E_NPE
        self.energy['CON'].value += np.ceil(len(N)/self.N_NPE) * len(spikes) * 2 * parameters.E_CON

        self.update_flag[L]=True

        # step 4,5
        if parameters.sync_type=='Async': 
            self.evaluation_process_delta(parameters, target_layer)
        
    def conv_layer_process_delta(self, event, parameters, target_layer):
        '''
        Conv layer with sigma-delta neuron model 
        event format: (time-stamp, Source Layer (optional), [H,W](optional), [C,Value], [C,Value], ...)
        parameters: pooling type('stride', 'maxpool', 'avgpool', None), pooling_size=[H, W], padding ('same','valid'), is_flatten
            
        Effect of poolings on the neuron/output sizes:
            avg_pooling: neuron statens= layer_shape/pooling_size,  outputs = layer_shape/pooling_size 
            stride:      neuron statens= layer_shape,               outputs = layer_shape/pooling_size 
            max_pool:    neuron statens= layer_shape,               outputs = layer_shape/pooling_size 

        Process: 
            0- Calculate the projection filed (destination neuron addresses) once per each CSC event
            1- Read weights
            2- Read neuron states
            3- Calculate new neuron states (state_new)
            4- Evaluate outputs delta = quantize(f(new_state)) - quantize(f(old_state))
            5- Update states
            6- Send non-zero events out 
        '''
        spikes = event[3:]
        L = target_layer
        N = self.neurons[L]
        [H,W] = event[2]
        [Kx,Ky] = L.weights.shape[0:2] # kernel sizes
        [Rx,Ry] = [(Kx-1)//2, (Ky-1)//2]# radious

        # For ease of calculations, define a temporary state equal to the [X,Y] size of the input
        if   L.padding=='same' : 
            input_shape = np.array(L.shape[0:2])*L.pooling_size+[2*Rx, 2*Ry]
        elif L.padding=='valid': 
            input_shape = np.array(L.shape[0:2])*L.pooling_size+[4*Rx, 4*Ry]
        else: 
            raise Exception(L.padding+" is not supported")
        
        state_tmp = np.zeros([input_shape[0], input_shape[1], len(N[2])])
        N_tmp = [N[0]*L.pooling_size[0], N[1]*L.pooling_size[1]] # mapped neurons from state_tmp
        for spike in spikes: 
            C = spike[0]   
            value= spike[1]
            weights = L.weights[:,:,C,N[2]] # debug

            # Steps 1,2,3 : W*V multiplication + Addition TODO: adjust precision
            state_tmp[H:H+Kx, W:W+Ky, :] += weights[::-1,::-1]*value
        
        # Handle the borders
        if L.padding=='same':       
            projection_field = [(intersection(range(H-Rx,H+Rx+1),N_tmp[0])/L.pooling_size[0]).astype(int), (intersection(range(W-Ry,W+Ry+1),N_tmp[1])/L.pooling_size[1]).astype(int)]
            if Rx>0: 
                state_tmp = state_tmp[Rx:-Rx, Ry:-Ry,:]
        elif L.padding=='valid':    
            projection_field = [(intersection(range(H-Kx+1,H+1),N_tmp[0])/L.pooling_size[0]).astype(int), (intersection(range(W-Ky+1,W+1),N_tmp[1])/L.pooling_size[1]).astype(int)]
            if Rx>0: 
                state_tmp = state_tmp[2*Rx:-2*Rx, 2*Ry:-2*Ry,:]
        else: 
            raise Exception("padding type ", L.padding, " is not supported")
        # Handle the pooling typs
        
        if L.pooling=='avg_pooling':
            state_tmp = skimage.measure.block_reduce(state_tmp, (L.pooling_size[0], L.pooling_size[1], len(N[2])), np.mean)
        elif L.pooling=='stride':
            state_tmp = state_tmp[::L.pooling_size[0], ::L.pooling_size[1], :]
        # check the size of temporary state 
        if state_tmp.shape[0:2]!=L.neuron_states.shape[0:2]: 
            raise Exception("layer shape for "+L.name+" doesn't work for my calculations")

        # adding the results to the neuron states
        for i in projection_field[0]:
            for j in projection_field[1]:
                L.neuron_states[i,j,N[2]] += state_tmp[i, j, :]
                self.update_flag[L][i,j]=True

        E_neuron_rd = (1-self.CMR[L][0])*parameters.EDmem_rd + self.CMR[L][0]*parameters.Eext_mem_rd
        E_neuron_wr = (1-self.CMR[L][0])*parameters.EDmem_wr + self.CMR[L][0]*parameters.Eext_mem_wr
        E_weight_rd = (1-self.CMR[L][1])*parameters.EDmem_rd + self.CMR[L][1]*parameters.Eext_mem_rd

        # For overheads and address calculatif the output queue is filled or notions
        self.energy['CON'].value += parameters.E_CON * (parameters.PAI + len(weights[0])*len(weights[1]))
        self.time.value += parameters.PAI + len(weights[0])*len(weights[1]) 
        # number of RISC-V operations = N_channels/N_NPE
        self.energy['CON'].value  += len(projection_field[0]) * len(projection_field[1]) * np.ceil(len(N[2])/self.N_NPE) * len(spikes) * parameters.E_CON * 2 
        compute_time = len(projection_field[0]) * len(projection_field[1]) * np.ceil(len(N[2])/self.N_NPE) * len(spikes) * parameters.T_NPE * 2
        # If either of neurons or weights are outside we need external memory access
        # For external memory we assume enough BW is available to read/write 1 word (32b) per cycle
        ext_mem_time = len(projection_field[0]) * len(projection_field[1]) * len(N[2]) * (self.CMR[L][0]*2*parameters.bitwidths['States'] + self.CMR[L][1]*len(spikes)*parameters.bitwidths['Weights'])/parameters.BWext_mem            
        self.time.value += np.max([ext_mem_time, compute_time])  # memory access and compute runs in parallel
        
        # energy for reading weights [read relevant weights once for each spike]
        self.energy['DMEM'].value += len(projection_field[0]) * len(projection_field[1]) * len(N[2]) * len(spikes) * E_weight_rd * parameters.bitwidths['Weights']
        # energy for read/write states [read/write relevant states once for each few of spike --> depends on the number of spike registers in the controller]
        self.energy['DMEM'].value += len(projection_field[0]) * len(projection_field[1]) * len(N[2]) * np.ceil(len(spikes)/parameters.Spike_Register)  * (E_neuron_rd + E_neuron_wr) * parameters.bitwidths['States']
        # energy for NPE
        self.energy['NPE'].value  += len(projection_field[0]) * len(projection_field[1]) * len(N[2]) * parameters.E_NPE * 2 
        # step 4,5
        if parameters.sync_type=='Async': 
            self.evaluation_process_delta(parameters, target_layer)

        return None
       

    def evaluation_process_delta(self, parameters, target_layer): 
        '''
        Process: 
            4- Evaluate outputs delta = quantize(f(new_state)) - quantize(f(old_state))
            5- Update states
            6- Send non-zero events out 
        '''
        L = target_layer

        neuron_states = L.neuron_states
        old_output = L.old_output
        update_flag = self.update_flag[L]
        if len(L.shape)==1: # add dummy dimentiones to neuron states and 
            neuron_states = np.expand_dims(neuron_states, axis=[0,1])
            old_output = np.expand_dims(old_output, axis=[0,1])
            update_flag = np.expand_dims([update_flag],axis=[1])
        
        neurons_to_update = self.neurons_3D[L]*np.expand_dims(update_flag,axis=[2])

        new_output = L.quantize(L.act(neuron_states)) # TODO: adjust precision
        delta_out = (new_output - old_output)*neurons_to_update # TODO: adjust precision

        # read update flag for Timestep_flag sync type (for all the mapped neurons)
        time_for_reading_update_flag = 0
        if parameters.sync_type=='TimeStep_flag': 
            self.energy['DMEM'].value+= parameters.EDmem_rd * np.product(update_flag.shape)
            # Flag reading can be accelerated by event generator to detect non-zero values (onces for every 16 bit)
            self.energy['NPE'].value += parameters.E_NPE * 2 * np.prod(update_flag.shape)/16 #read + event generator
            self.energy['CON'].value += parameters.E_CON * (np.sum(update_flag)+ 2 * np.prod(update_flag.shape)/self.N_NPE/16)
            time_for_reading_update_flag = parameters.T_NPE * (np.sum(update_flag) + 2 * np.prod(update_flag.shape)/self.N_NPE/16) # <-- TIME
        if np.all(neurons_to_update==0): 
            return None #skip the evaluation process

        E_neuron_rd = (1-self.CMR[L][0])*parameters.EDmem_rd + self.CMR[L][0]*parameters.Eext_mem_rd
        E_neuron_wr = (1-self.CMR[L][0])*parameters.EDmem_wr + self.CMR[L][0]*parameters.Eext_mem_wr


        # read states (only if it is not immediate evaluation)
        if parameters.sync_type!='Async': 
            self.energy['DMEM'].value += E_neuron_rd * np.sum(neurons_to_update) * parameters.bitwidths['States']
        # read old outputs
        self.energy['DMEM'].value += E_neuron_rd * np.sum(neurons_to_update) * parameters.bitwidths['Outputs'] 
        # activation function evaluation + quantization 
        self.energy['NPE'].value += parameters.E_NPE * np.sum(neurons_to_update) * 2 
        # RISC-V needs to read state, old_output, 2xcommand NPEs --> whenever 2D neurons_to_update[X,Y] is 1 operate over all the channels divided by N_NPEs
        self.energy['CON'].value += parameters.E_CON * np.sum(neurons_to_update[:,:,0]) * np.ceil(np.sum(neurons_to_update[0,0,:])/self.N_NPE) * (3 + (parameters.sync_type!='Async'))
        time_for_read_state_oldoutput_act_quantize = parameters.T_NPE * np.sum(neurons_to_update[:,:,0]) * np.ceil(np.sum(neurons_to_update[0,0,:])/self.N_NPE) * (2 + L.act_time() + (parameters.sync_type!='Async'))
        # If neuron outputs are outside we need external memory access
        # For external memory we assume enough BW is available to read/write 1 word (32b) per cycle
        ext_mem_time_for_output_read = self.CMR[L][0] * np.sum(neurons_to_update) * parameters.bitwidths['Outputs'] /parameters.BWext_mem
        

        # step 6
        # find nonzero delta_outs
        if L.is_flatten: 
            delta_out = delta_out.reshape([1,1,delta_out.shape[0]*delta_out.shape[1]*delta_out.shape[2]])
        spike_ids = np.nonzero(delta_out)
        self.energy['NPE'].value += parameters.E_NPE * np.sum(neurons_to_update) 
        self.energy['CON'].value += parameters.E_CON * (np.sum(neurons_to_update[:,:,0]) * np.ceil(np.sum(neurons_to_update[0,0,:])/self.N_NPE)+3*L.is_flatten)
        time_for_extracting_non_zero_deltas = (parameters.sync_type!='Async') * parameters.T_NPE * (np.sum(neurons_to_update[:,:,0]) * np.ceil(np.sum(neurons_to_update[0,0,:])/self.N_NPE)+3*L.is_flatten)
        if len(spike_ids[0])==0: 
            return None # no need to continue
        # update old outputs
        if   len(L.shape)==1: 
            L.old_output[neurons_to_update[0,0,:]] = np.squeeze(new_output[neurons_to_update])
        elif len(L.shape)==3: 
            L.old_output[neurons_to_update] = new_output[neurons_to_update]
        else: 
            raise Exception("dimentions of the "+L.name+" is nither 1 (dense) or 3 (conv)")
        
        self.energy['DMEM'].value += E_neuron_wr * len(spike_ids[0]) * parameters.bitwidths['Outputs'] 
        self.energy['CON'].value += parameters.E_CON * len(spike_ids[0])
        time_for_updating_old_state = parameters.T_NPE * len(spike_ids[0])
        ext_mem_time_for_output_write = self.CMR[L][0] * len(spike_ids[0]) * parameters.bitwidths['Outputs']  /32
        
        time_for_reseting_update_flag=0
        if parameters.sync_type=='TimeStep_flag': # reset update flags
            self.update_flag[L]=self.update_flag[L]*False
            self.energy['DMEM'].value+= parameters.EDmem_wr * np.sum(neurons_to_update)
            # can reset one line of the kernel in one cycle at least
            self.energy['CON'].value += parameters.E_CON * np.ceil(np.sum(neurons_to_update)/np.min([L.weights.shape[0], 32])) 
            time_for_reseting_update_flag = parameters.T_NPE * np.ceil(np.sum(neurons_to_update)/np.min([L.weights.shape[0], 32])) 
        if parameters.sync_type=='Async': # reset update flags
            self.update_flag[L]=self.update_flag[L]*False


        total_time_for_evaluation = np.max([(time_for_reading_update_flag + time_for_read_state_oldoutput_act_quantize + time_for_updating_old_state + time_for_extracting_non_zero_deltas + time_for_reseting_update_flag), (ext_mem_time_for_output_read + ext_mem_time_for_output_write)])
        time_for_each_output_spike = np.ceil(total_time_for_evaluation/len(spike_ids[0]))
        H = spike_ids[0][0]
        W = spike_ids[1][0]
        out_event = [self.time.value, L.ID, [H,W]] #(time-stamp, Source Layer, [H,W]) --> then append the [C,Value(s)]
        for i in range(len(spike_ids[0])):
            # max event flit or moving to another [H,W]
            if parameters.cnt_event_flit(out_event)==parameters.max_event_flits or H!=spike_ids[0][i] or W!=spike_ids[1][i]:
                out_queue_full = self.queue['out_queue_occupancy'].value >=self.output_queue_depth #cannot process anything new because output fifo is full
                if out_queue_full:
                    self.internal_output_queue.append(out_event)
                else:
                    out_event[0] = self.time.value
                    self.queue['out_event_queue'].put(out_event)
                    self.energy['FIFO'].value += parameters.Efifo_wr*parameters.cnt_event_flit(out_event)*parameters.flit_width #event write
                    self.queue['out_queue_occupancy'].value  += parameters.cnt_event_flit(out_event)
                    self.energy['CON'].value += parameters.E_CON * parameters.cnt_event_flit(out_event) # send flits
                    self.time.value += parameters.cnt_event_flit(out_event)
                # reset the output event
                H = spike_ids[0][i]
                W = spike_ids[1][i]
                out_event = [self.time.value, L.ID, [H,W]] 
            C = spike_ids[2][i]
            out_event.append([C, delta_out[H,W,spike_ids[2][i]]])
            self.time.value += time_for_each_output_spike
        # for the last spikes  
        out_queue_full = self.queue['out_queue_occupancy'].value >=self.output_queue_depth #cannot process anything new because output fifo is full
        if out_queue_full:
            self.internal_output_queue.append(out_event)
        else:
            out_event[0] = self.time.value
            self.queue['out_event_queue'].put(out_event) # send the remaining of events         
            self.energy['FIFO'].value += parameters.Efifo_wr*parameters.cnt_event_flit(out_event)*parameters.flit_width #event write
            self.queue['out_queue_occupancy'].value  += parameters.cnt_event_flit(out_event)
            self.energy['CON'].value += parameters.E_CON * parameters.cnt_event_flit(out_event) # send flits
            self.time.value += parameters.cnt_event_flit(out_event) 


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return np.array(lst3)  

def layer_process(core, layer, parameters, event):
    if event!='eval': 
        if layer.layer_type=='dense':  
            if layer.neuron_type=='SigmaDelta':  
                core.dense_layer_process_delta(event=event, parameters=parameters, target_layer=layer) 
            else:  
                raise Exception("neuron type: "+layer.neuron_type+" is not implemented") 
        elif layer.layer_type=='conv':  
            if layer.neuron_type=='SigmaDelta':  
                core.conv_layer_process_delta(event=event, parameters=parameters, target_layer=layer) 
            else:  
                raise Exception("neuron type: "+layer.neuron_type+" is not implemented") 
        else: 
            raise Exception("layer type: "+layer.type+" is not implemented") 
    else:
        if layer.neuron_type=='SigmaDelta': 
            core.evaluation_process_delta(parameters=parameters, target_layer=layer)
        else:  
            raise Exception("neuron type: "+layer.neuron_type+" is not implemented")
        return

def core_idle(core, time):
    if core.time.value<time:
        core.idle_times.value += time- core.time.value
        core.time.value = time
    return


def correct_mapping_in_range(layer, neuron_ranges):
    output = True
    for i in range(len(neuron_ranges)):
        neuron_ranges[i]= np.array(neuron_ranges[i])
        if any(np.array(neuron_ranges[i])>=layer.neuron_states.shape[i]): 
            neuron_ranges[i] = neuron_ranges[i][neuron_ranges[i]<layer.neuron_states.shape[i]]
            output = False
    return output






