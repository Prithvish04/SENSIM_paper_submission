import os, sys
sys.path.append(os.getcwd())
from libraries.simulator import Simulator

from iodefs  import PilotNetIO as IO
from appdefs import PilotNet_app as App

Spike_Register= 8
if len(sys.argv)>0:
    for args in sys.argv:
        print(args)
        if args[0:len('-Spike_Register:')]=='-Spike_Register:':
            Spike_Register= int(args[len('-Spike_Register:'):])
print("Spike_Register is adjusted to:", Spike_Register)
    

if __name__ == '__main__':  
    print ('Spike_Register= ', Spike_Register) 
    sim = Simulator()
    myapp = App(sim, Spike_Register)        
    sim.setIO(IO(sim))
    sim.setHW()

    myapp.composeLayers()
    sim.layer_core_map = myapp.composeLayersCoresMap()
    sim.bus_list = myapp.composeBusSegmentsList()
    
    sim.run()

    sim.postRun()
    print ('Spike_Register= ', Spike_Register)  



    