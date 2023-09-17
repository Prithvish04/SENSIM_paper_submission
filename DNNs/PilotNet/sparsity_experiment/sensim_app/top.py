import os, sys
sys.path.append(os.getcwd())
from libraries.simulator import Simulator

from iodefs  import PilotNetIO as IO
from appdefs import PilotNet_app as App

threshold_factor= 14.4
if len(sys.argv)>0:
    for args in sys.argv:
        print(args)
        if args[0:len('-10xthr:')]=='-10xthr:':
            threshold_factor= float(args[len('-10xthr:'):])/10
print("threshold_factor is adjusted to:", threshold_factor)
    

if __name__ == '__main__':  
    print ('threshold_factor= ', threshold_factor) 
    sim = Simulator()
    myapp = App(sim, threshold_factor)        
    sim.setIO(IO(sim, threshold_factor))
    sim.setHW()

    myapp.composeLayers()
    sim.layer_core_map = myapp.composeLayersCoresMap()
    sim.bus_list = myapp.composeBusSegmentsList()
    
    sim.run()

    sim.postRun()
    print ('threshold_factor= ', threshold_factor)  



    