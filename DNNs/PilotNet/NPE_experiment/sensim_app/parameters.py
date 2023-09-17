import numpy as np

class parameters:
    def __init__(self):
        '''
        Energy estimations from technology node (all energies are in the unit of pJ)
        Timing of architecture (all times are in the unit of clock period)
        We do not consider the pipelining layency, therefore the time is only the overage throughput
        '''

        self.time_step= 100
        self.NrPhysicalCore4Cores = 2
        self.NrPhysicalCore4Buses = 1
        self.queue_depths = [1280,1280] # when time_step is higher, the queue size should also be higher (simulator effect)
        self.N_NPE = 16
        self.mesh_size = [12,4] # size of the HW [col, row]

        self.snapshot_log_time_step = 10e3 
        self.clk_freq = 200e6 # to calculate real-time for plotting

        # energy per each controller operation + instruction memory read
        self.E_CON = 3
        # energy per each NPE operation
        self.E_NPE = 1
        # energy per each Data memory read bit
        self.EDmem_rd = 3
        # energy per each Data memory write bit
        self.EDmem_wr = 3
        # energy per each shared memory read bit
        self.Eext_mem_rd = 300
        # energy per each shared memory write bit
        self.Eext_mem_wr = 300
        # energy per each event queue read bit
        self.Efifo_rd = 1.5
        # energy per each event queue write bit
        self.Efifo_wr = 1.5

         
        # time per each NPE operation (compare to one RISC-V cycle)
        self.T_NPE = 1
        # time per each event queue access
        self.Tfifo = 1
        # latency for accessing the shared memory
        self.Text_mem = 100
        # shared memeory BW (bits per cycle)
        self.BWext_mem = 32

        # energy for sending a bit of data for one bus leg
        self.Ebus = 6
        # time for sending a flit of data for one bus leg
        self.Tbus = 1

        # number of operations + memory access for control unit
        self.PAI = 10 # For Pointer Access Indirection


        

        # flow control mode in the interconnect
        #   - free : without flow control (No back pressure)
        #   - strict: with back-pressure (no packet loss allowed)
        self.flow_control='strict'
        # define if the NoC supports multicasting (NOTE: naive implementation)
        self.multicast=True

        # Format of event in the platform (time-stamp is only exists for simulation) 
        # Only in simulation for dense layers [H,W]=[0,0], Source layer is always present here but in HW it is optional, time step is only in simulation. 
        # in HW we may have extra fileds in the header (like number of flits)
        # Compressed Sparse Channel format (time-stamp, Source Layer (optional), [H,W](optional), [C,Value], [C,Value], ...) multiple flits / variable size
        self.flit_width = 32 #number of bits per flits        
        self.max_event_flits = 9 # limit the max number of flits per event
        self.spike_per_flits = 2 # number of [C,Value] per each flit 
        self.header_flits = 1 # number of flits for the header [Source Layer, H, W] --> can be 0 for single flit AER events

        
        # sync_type: type of synchronization
        #   - TimeStep: All neurons in the core will be evaluated once in a evaluation time-step
        #   - Async: Neurons can fire anytime
        #   - TimeStep_flag Async: Only updated neurons during the previous time step will be evaluated once in an evaluation time-step (costs 1 bit per [H,W] position)
        #      update flag is initially 1 for all neurons, to let the neurons communicate their biasses. 
        self.sync_type='TimeStep_flag'
        self.evaluation_time_step= 10e3 
        self.suppress_bias_wave = False # This option make the initial update flag to 0 and may result in inconsistant outcome from TF
        self.consume_then_fire = False # First priority is to consume input events, so no evaluation until input queue is empty

        # WNO_bits: bit widths of Weights, Neurons and Outputs [W, N, O]
        self.bitwidths = {
                'Weights'  : 4,
                'States'  : 16,
                'Outputs'  : 1 
                }

        # number of spike register [C,Value] inside the controller [results in reuse of neuron state]
        self.Spike_Register = 8

        # Data memory of each core (in bit)
        self.Core_Dmem = 2*1024*1024



    def cnt_event_flit(self, ev):
        # calculate the number of flits for events 
        return int(self.header_flits + np.ceil((len(ev)-3)/self.spike_per_flits))
