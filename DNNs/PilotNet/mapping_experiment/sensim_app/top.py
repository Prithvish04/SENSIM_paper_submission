import os, sys
sys.path.append(os.getcwd())
from libraries.simulator import Simulator

from iodefs  import PilotNetIO as IO
from appdefs import PilotNet_app_map3 as App



if __name__ == '__main__':  
    sim = Simulator()
    myapp = App(sim)        
    sim.setIO(IO(sim))
    sim.setHW()

    myapp.composeLayers()
    sim.layer_core_map = myapp.composeLayersCoresMap()
    sim.bus_list = myapp.composeBusSegmentsList()
    
    sim.run()

    sim.postRun() 



    