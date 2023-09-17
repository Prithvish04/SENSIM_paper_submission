import os, sys
sys.path.append(os.getcwd())
from libraries.simulator import Simulator

from iodefs  import PilotNetIO as IO
from appdefs import PilotNet_app as App
import time as tm

 # define NrPhysicalCore4Cores
def experiment(NrPhysicalCore4Cores, time_step, file_name):
    print ('NrPhysicalCore4Cores= ', str(NrPhysicalCore4Cores))
    print("time_step is adjusted to:", str(time_step))

    sim = Simulator()
    myapp = App(sim, NrPhysicalCore4Cores=NrPhysicalCore4Cores, time_step=time_step)        
    sim.setIO(IO(sim))
    sim.setHW()

    myapp.composeLayers()
    sim.layer_core_map = myapp.composeLayersCoresMap()
    sim.bus_list = myapp.composeBusSegmentsList()
    
    start_time = tm.time()
    sim.run()
    total_time = (tm.time()-start_time)
    print("\nTotal execution time=", total_time," sec")

    
    sim.postRun() 
    print ('NrPhysicalCore4Cores= ', str(NrPhysicalCore4Cores))
    print("time_step is adjusted to:", str(time_step))
    return total_time


if __name__ == '__main__':
    file_name = "DNNs/PilotNet/parallelization_speed_experiment/outputs/time_measurement_results.csv"
    setting_file = open(file_name, 'w')    
    setting_file.write("#NrPhysicalCore4Cores,time_step,total_time(sec)\n")
    
    NrPhysicalCore4Cores=[1,2,3,4]
    time_step = [100,1000,10000,100000]

    for NrPC in NrPhysicalCore4Cores:
        for ts in time_step:
            total_time = experiment(NrPhysicalCore4Cores=NrPC, time_step=ts, file_name=file_name)
            setting_file.write(str(NrPC)+","+str(ts)+","+str(total_time)+"\n")

    setting_file.close()