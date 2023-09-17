'''
Interconnect model
This is the interconnect class of which only a single instance should be instantiated in the main function.
An instance of this class can be:
1- bus segments (one object per segment)
'''

import libraries.utils as utils
from multiprocessing import Value
import numpy as np

class interconnect:
    
    def __init__(self, master_cores=None, slave_cores=None, name=None, verbose=False):
        """
        The function initializes an object with specified parameters and builds a bus using the provided
        master and slave cores.
        
        :param master_cores: The `master_cores` parameter is a list of objects representing the master
        cores in a system. Each master core object should have a method called `getQueue()` that returns
        a queue object
        :param slave_cores: The "slave_cores" parameter is a list of cores that are connected to the bus
        as slaves. Each core in the list represents a slave core that can send and receive data through
        the bus
        :param name: The name of the object being initialized. It is an optional parameter and can be
        set to any string value
        :param verbose: The `verbose` parameter is a boolean flag that determines whether or not to
        print additional information during the execution of the code. If `verbose` is set to `True`,
        then additional information will be printed. If it is set to `False`, then no additional
        information will be printed, defaults to False (optional)
        """
        self.name = name
        self.verbose = verbose
        self.master_cores = master_cores
        self.slave_cores= slave_cores
        master_queues = []
        slave_queues = []
        for core in master_cores:
            master_queues.append(core.getQueue())

        for core in slave_cores:
            slave_queues.append(core.getQueue())

        self.time = Value("d", 0, lock=False)
        self.energy = Value("d", 0, lock=False)
        self.total_flits = Value("d", 0, lock=False)
        self.reset_bus()
        
        self.build_bus(master_queues, slave_queues)

    def communication(self, time, parameters):
        """
        Perform communication for this timestep
            - time: current time
            - parameters: hardware parameters
        return number of event-flits sent in this time-step
        """
        count_flits = self.communication_bus(time, parameters)
        self.total_flits.value += count_flits
        return count_flits
 


#------------------  BUS topology ---------------------------------------------

    def build_bus(self, master_queues, slave_queues):
        '''
        Building segmented BUS interconnect. 
        Each bus segment should be define as a seperate object
            - master_queues: list of master cores in the bus segment, should be put here in order of tocken ring hopping like [C1,C2] 
            - slave_queues: list of slave cores in the bus segment like [C3,C4]
        '''
        self.master_queues = master_queues
        self.slave_queues = slave_queues
        self.reset_bus()
        
    def reset_bus(self):
        self.time.value = 0 
        self.energy.value = 0
        self.total_flits.value = 0
        self.tocken_ind = 0 # Core[0] will contain the tocken at the start time
        

    def communication_bus(self, time, parameters):       
        """
        Loop over all the output_event_queues for master cores and deliver their events to 
        the input_event_queues of slave cores
        Assumption: output event queue is sorted in time
        This process guarantees that Input_event_queues is sorted in time
        return number of event-flits transfered in this time-step
        - to access the name of the queue --> master_queues[queue_ind]['out_event_queue'].parentName
        - if one of the destinations if full, the event will not be transmitted to any other destinations (in strict flow control)
        """ 
        end_time = time + parameters.time_step
        count_flits = 0
        # consider the idle time for the bus
        if self.time.value<time: self.time.value = time

        for master_queues in self.master_queues:
            if master_queues['out_queue_occupancy_peak'].value < master_queues['out_queue_occupancy'].value: 
                master_queues['out_queue_occupancy_peak'].value = master_queues['out_queue_occupancy'].value
                
        if self.time.value>end_time: return count_flits # end the segment process for this timestep! 

        # Loop over master cores. Randomly shuffle the order of cores to have fair service
        order = np.arange(len(self.master_queues))
        np.random.seed(int(end_time%1e6))
        np.random.shuffle(order)
        back_pressure = 0
        
        for core_ind in order:
            masterQueue = self.master_queues[core_ind]
            # Does this core has something to send?
            if masterQueue['out_event_queue'].isEmpty(): continue
            if time_stamp(masterQueue['out_event_queue'].get(False))>end_time: continue

            
            while masterQueue['out_event_queue'].isNotEmpty() and time_stamp(masterQueue['out_event_queue'].get(False)) <= time:
                ev = masterQueue['out_event_queue'].get(False)
                if parameters.flow_control=='strict':
                    for slaveQueue in self.slave_queues: 
                        fifo_full = (slaveQueue['in_queue_occupancy'].value+parameters.cnt_event_flit(ev)) > utils.MemorySizeToNrFilits(slaveQueue['in_event_queue'].size.value)
                        if fifo_full: back_pressure=1
                if self.time.value>=end_time or back_pressure: break # end the segment process for this timestep! 


                if parameters.multicast==True:
                    # assume the event will arrive to the slaves with delay equal to average bus distance
                    self.time.value   += parameters.cnt_event_flit(ev)*parameters.Tbus*find_total_connection_length_multicast(masterQueue, self.slave_queues)/len(self.slave_queues)
                    # assume the event needs to hop equal to all the bus legs once
                    self.energy.value += parameters.cnt_event_flit(ev)*parameters.flit_width*parameters.Ebus*find_total_connection_length_multicast(masterQueue, self.slave_queues)           
                    
                if parameters.multicast==False:
                    # assume the event will arrive to the slaves with delay equal to average bus distance
                    self.time.value   += parameters.cnt_event_flit(ev)*parameters.Tbus*find_total_connection_length_unicast(masterQueue, self.slave_queues)/len(self.slave_queues)
                    # assume the event needs to hop equal to all the bus legs once
                    self.energy.value += parameters.cnt_event_flit(ev)*parameters.flit_width*parameters.Ebus*find_total_connection_length_unicast(masterQueue, self.slave_queues)           
                    
                ev[0] = self.time.value # Update the time stamp of the event
                for slaveQueue in self.slave_queues: 
                    fifo_full = (slaveQueue['in_queue_occupancy'].value+parameters.cnt_event_flit(ev)) > utils.MemorySizeToNrFilits(slaveQueue['in_event_queue'].size.value)
                    if not(fifo_full): 
                        slaveQueue['in_event_queue'].put(ev.copy())
                        slaveQueue['in_queue_occupancy'].value += parameters.cnt_event_flit(ev)
                    else:
                        slaveQueue['packet_loss'].value+=1

                
                masterQueue['out_queue_occupancy'].value -= parameters.cnt_event_flit(ev)
                if masterQueue['out_queue_occupancy'].value<0: 
                    raise Exception("output_queue_occupancy is negative!!!")
                
                if parameters.multicast==True : count_flits += parameters.cnt_event_flit(ev) * find_total_connection_length_multicast(masterQueue, self.slave_queues)
                if parameters.multicast==False: count_flits += parameters.cnt_event_flit(ev) * find_total_connection_length_unicast(masterQueue, self.slave_queues)
                if self.verbose: print("[time=", str(end_time), ", time_stamp="+str(ev[0])+", src_layer="+str(ev[1])+", value="+str([ev[2],ev[3]])+"]")
                masterQueue['out_event_queue'].get()
 
            if self.time.value>=end_time or back_pressure: break # end the segment process for this timestep! 
            # print("interconnect: ", self.name, "number of flits: ", count_flits)
        return count_flits

def time_stamp(ev=None):
    return ev[0]
        


def find_total_connection_length_multicast(masterQueue, slaveQueues):
    """
    The function calculates the total distance between a master queue and multiple slave queues by
    finding the maximum and minimum distances in each dimension and summing them.
    
    :param masterQueue: The masterQueue parameter is a dictionary that contains information about the
    master queue. It has a key 'out_event_queue' which corresponds to the output event queue of the
    master queue
    :param slaveQueues: slaveQueues is a list of dictionaries, where each dictionary represents a slave
    queue. Each dictionary contains two keys: 'in_event_queue' and 'out_event_queue'. The value of
    'in_event_queue' is an object representing the input event queue of the slave queue, and the value
    of '
    :return: the total distance of the connection length for multicast communication.
    """
    loc_master   = np.array(masterQueue['out_event_queue'].location)
    number_of_hops_positive = [0,0,0]
    number_of_hops_negative = [0,0,0]
    for slaveQueue in slaveQueues:
        loc_slave = np.array(slaveQueue['in_event_queue'].location)
        distance = loc_slave-loc_master
        number_of_hops_positive = np.max([distance, number_of_hops_positive], axis=0)
        number_of_hops_negative = np.min([distance, number_of_hops_negative], axis=0)
    total_distance = np.sum(number_of_hops_positive-number_of_hops_negative)
    return total_distance

def find_total_connection_length_unicast(masterQueue, slaveQueues):
    """
    The function calculates the total distance between the master queue and all the slave queues in a
    unicast network.
    
    :param masterQueue: The masterQueue parameter is a dictionary that contains information about the
    master queue. It has a key called 'out_event_queue' which represents the outgoing event queue of the
    master queue
    :param slaveQueues: The `slaveQueues` parameter is a list of dictionaries. Each dictionary
    represents a slave queue and contains the following key-value pairs:
    :return: the total distance between the master queue and all the slave queues.
    """
    loc_master   = np.array(masterQueue['out_event_queue'].location)
    total_distance = 0
    for slaveQueue in slaveQueues:
        loc_slave = np.array(slaveQueue['in_event_queue'].location)
        total_distance += np.sum(np.abs(loc_slave-loc_master))
    return total_distance