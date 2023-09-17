import os, sys
sys.path.append(os.getcwd())
from libraries.simulator import Simulator

from iodefs  import PilotNetIO as IO
from appdefs import PilotNet_app as App


# sync_type: type of synchronization
#   - TimeStep: All neurons in the core will be evaluated once in a evaluation time-step
#   - Async: Neurons can fire anytime
#   - TimeStep_flag Async: Only updated neurons during the previous time step will be evaluated once in an evaluation time-step (costs 1 bit per [H,W] position)
#      update flag is initially 1 for all neurons, to let the neurons communicate their biasses.
sync_type= 'TimeStep_flag'
if len(sys.argv)>0:
    for args in sys.argv:
        print(args)
        if args[0:len('-sync_type:')]=='-sync_type:':
            sync_type= args[len('-sync_type:'):]
print("sync_type is adjusted to:", sync_type)

if __name__ == '__main__':  
    print ('sync_type= ', sync_type)
    if sync_type!='TimeStep' and sync_type!='Async' and sync_type!='TimeStep_flag':
        raise Exception("invalid sync type")

    sim = Simulator()
    myapp = App(sim, sync_type=sync_type)        
    sim.setIO(IO(sim))
    sim.setHW()

    myapp.composeLayers()
    sim.layer_core_map = myapp.composeLayersCoresMap()
    sim.bus_list = myapp.composeBusSegmentsList()
    
    sim.run()

    sim.postRun() 
    print ('sync_type= ', sync_type)



    