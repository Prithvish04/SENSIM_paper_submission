from multiprocessing import Value
import multiprocessing
import time as tm
from multiprocessing import Value

from libraries.core import core
from libraries.parameters import parameters
import libraries.utils as utils
from libraries.utils import DEBUG_SIM

class Simulator:
    def __init__(self, outputDir=None):
        multiprocessing.freeze_support()
        self.param = parameters()
        if outputDir == None:
            self.output_dir = "gui/"
        else:
            self.output_dir = outputDir + '/'
        self.terminate_on = 'time' # termination can be based on 'time' or 'data'

    def setHW(self):
        """
        The function sets up a simulation with a specified number of physical cores and a simulation
        timestep, creates barrier objects for synchronization, defines core objects with their locations and
        queue depths, and creates a list of cores and a dictionary mapping layers to cores.
        """
        if(DEBUG_SIM):
            print("Starting simulation with %d physcores, simulation timestep set to %.2E ..."%(self.param.NrPhysicalCore4Cores, self.param.time_step))

        self.barrier_obj1 = multiprocessing.Barrier(self.param.NrPhysicalCore4Buses+1)
        self.barrier_obj2 = multiprocessing.Barrier(self.param.NrPhysicalCore4Cores+self.param.NrPhysicalCore4Buses+1)
        self.barrier_obj3 = multiprocessing.Barrier(self.param.NrPhysicalCore4Cores+1)
        self.barrier_obj4 = multiprocessing.Barrier(self.param.NrPhysicalCore4Cores+self.param.NrPhysicalCore4Buses+1)

        #define core objects (loc=[x,y,z])
        # virtual cores (not assigned/mapped into real cores)
        self.CI = core(name='CI', loc=[-1,0,0], queue_depths=[self.param.queue_depths[0], 10**8]) # input Qdepth is used for online frame2event conv
        self.CO = core(name='CO', loc=[self.param.mesh_size[0],0,0], queue_depths=[self.param.queue_depths[0], 0])

        # make list of cores, inputQes and outputQes (based on topology dimentions)
        self.coresList = []
        for i in range(0,self.param.mesh_size[0]):
            cores_sublist = []
            for j in range(0, self.param.mesh_size[1]):
                cores_sublist.append(core(name='C'+str(i)+'_'+str(j), loc=[i,j,0], queue_depths=self.param.queue_depths, N_NPE=self.param.N_NPE))
            self.coresList.append(cores_sublist)
    
        self.layer_core_map = dict()
        self.bus_list = []

    def setIO(self, ioObj):
        self.io = ioObj

    def run(self):
        # Core_Layer mapping (to extract the mapped layers in each core) 
        # Example of one item --> C1:(L1, [range(0,6), range(0,6), range(0,4)])
        # {CoreObj: (layerObj, [multidimentional neuron def])}
        core_layer_map = dict()
        for layr, cores_neurons in self.layer_core_map.items():
            for core_neurons in cores_neurons: 
                core_layer_map.setdefault(core_neurons[0], list()).append([layr,core_neurons[1], core_neurons[2]])     

        self.usedCoresList = []
        for core in core_layer_map.keys():
            if((core != self.CO) and (core != self.CI)):
                self.usedCoresList.append(core)

        layerList = []             
        for layer in self.layer_core_map.keys():
            layer.ID = len(layerList)
            layerList.append(layer)
                
        #build core objects (layer_core_map=None, parameters=None) -- no need to build input/output cores
        for core in self.usedCoresList: 
            core.setMappedPackage(core_layer_map[core], layerList, parameters=self.param)

        # make GUI setting file
        utils.gui_setting_file(core_layer_map=core_layer_map, mesh_size=self.param.mesh_size, file_name=self.output_dir+"gui_setting.csv")
        self.gui_snapshot = utils.snapshot(core_list=self.usedCoresList, bus_list= self.bus_list, capture_intervals=0, max_num_captures=1e6, reset_states=True, file_name=self.output_dir+"snapshots")

        ####################### Event generation from whole dataset ########################
        if(self.fetchWholeDataSet):
            self.io.fetchWholeData()
            fetch_data_status = 1
        #######################


        NrBuses = len(self.bus_list)    
        time = Value("d", 0, lock=False)
        sim_terminate = Value("d", 0, lock=False)

         # Divide usedCoresList among physical processors in computer
        coresToPhyCoreList = []
        NrUsedCores = len(self.usedCoresList)

        for i in range(self.param.NrPhysicalCore4Cores):
            coresToPhyCoreList.append([])

        for i in range(NrUsedCores):
            indexOfPhysCore = i % self.param.NrPhysicalCore4Cores
            coresToPhyCoreList[indexOfPhysCore].append(self.usedCoresList[i])

        barrier_obj1 = multiprocessing.Barrier(self.param.NrPhysicalCore4Buses+1)
        barrier_obj2 = multiprocessing.Barrier(self.param.NrPhysicalCore4Cores+self.param.NrPhysicalCore4Buses+1)
        barrier_obj3 = multiprocessing.Barrier(self.param.NrPhysicalCore4Cores+1)
        barrier_obj4 = multiprocessing.Barrier(self.param.NrPhysicalCore4Cores+self.param.NrPhysicalCore4Buses+1)

    # Divide busesList among physical processors in computer
        busesToPhyCoreList = []

        for i in range(self.param.NrPhysicalCore4Buses):
            busesToPhyCoreList.append([])

        for i in range(NrBuses):
            indexOfPhysCore = i % self.param.NrPhysicalCore4Buses
            busesToPhyCoreList[indexOfPhysCore].append(self.bus_list[i])

        busRunnerProcsList = []
        for idx, mappedBuses in enumerate(busesToPhyCoreList):
            proc = busRunnerProcess(idx, mappedBuses, sim_terminate, self.param, barrier_obj1, barrier_obj2, barrier_obj4, time)
            busRunnerProcsList.append(proc)
            proc.start()

        coreRunnerProcsList = []
        for idx, mappedCores in enumerate(coresToPhyCoreList):
            proc = coreRunnerProcess(idx, mappedCores, sim_terminate, self.param, barrier_obj2, barrier_obj3, barrier_obj4, time)
            coreRunnerProcsList.append(proc)
            proc.start()
        
        start_time = tm.time()
        if(DEBUG_SIM):
            print("\nRunning Simulation...\n")  
        while sim_terminate.value==0:
            if(self.fetchWholeDataSet == False):
                fetch_data_status = self.io.fetchData()

            barrier_obj1.wait() 
            barrier_obj2.wait() # all bus processed finished
            barrier_obj3.wait() # all core processes finished

            self.io.dumpData(time.value)
            if (time.value%self.param.snapshot_log_time_step) == 0: #out_event_processed:
                if(DEBUG_SIM):
                    print('\rSimulation Time: %.3E'%(time.value), end='')
                self.gui_snapshot.capture(time.value)

            time.value += self.param.time_step
            barrier_obj4.wait() # all extra works (main process) finished

            #termination condition
            if self.terminate_on=='time':
                sim_terminate.value=(time.value>=self.simulation_end_time)
            elif self.terminate_on=='data':
                sim_terminate.value=(fetch_data_status==0)
            else:
                raise Exception("Unknown sim_terminate condition")

        self.gui_snapshot.file_close()
        if(DEBUG_SIM):
            print("\nTotal execution time=", (tm.time()-start_time)," sec. Terminated based on ", self.terminate_on)

        self.io.close()
        return time.value
    
    def postRun(self):
        if self.plot_outputs:
            self.io.doPlot()

class busRunnerProcess (multiprocessing.Process):
    def __init__(self, idx, mappedBuses, sim_terminate, param, barrier1, barrier2, barrier4, time):
        multiprocessing.Process.__init__(self)
        self.idx = idx
        self.mappedBuses = mappedBuses
        self.param = param
        self.barrier1 = barrier1
        self.barrier2 = barrier2
        self.barrier4 = barrier4
        self.time = time
        self.sim_terminate = sim_terminate
        if(DEBUG_SIM):
            print ("BusProcess %d is started..." % (self.idx))
    
    def run(self):
        while self.sim_terminate.value==0:
            self.barrier1.wait()
            for bus in self.mappedBuses:
                bus.communication(time=self.time.value, parameters=self.param)
            self.barrier2.wait()
            self.barrier4.wait()
        else: 
            if(DEBUG_SIM):
                print("BusProcess %d is terminated..." % (self.idx))  
    
class coreRunnerProcess (multiprocessing.Process):
    def __init__(self, idx, mappedCores, sim_terminate, param, barrier2, barrier3, barrier4, time):
        multiprocessing.Process.__init__(self)
        self.idx = idx
        self.mappedCores = mappedCores
        self.param = param
        self.barrier2 = barrier2
        self.barrier3 = barrier3
        self.barrier4 = barrier4
        self.time = time
        self.sim_terminate = sim_terminate
        if(DEBUG_SIM):
            print ("CoreProcess %d is started..." % (self.idx))
    
    def run(self):
        while self.sim_terminate.value==0:
            self.barrier2.wait()
            for core in self.mappedCores:
                core.process_callback(time=self.time.value, parameters=self.param)               
            self.barrier3.wait()
            self.barrier4.wait()    
        else:
            if(DEBUG_SIM): 
                print("CoreProcess %d is terminated..." % (self.idx))  
