import os, sys
sys.path.append(os.getcwd())
from libraries.simulator import Simulator

from iodefs  import PilotNetIO as IO
from appdefs import PilotNet_app as App


 # define if the NoC supports multicasting (NOTE: naive implementation)
multicast=1
if len(sys.argv)>0:
    for args in sys.argv:
        print(args)
        if args[0:len('-multicast:')]=='-multicast:':
            multicast= int(args[len('-multicast:'):])
print("multicast is adjusted to:", str(multicast))

if __name__ == '__main__':  
    print ('multicast= ', str(multicast))

    sim = Simulator()
    myapp = App(sim, multicast=multicast)        
    sim.setIO(IO(sim))
    sim.setHW()

    myapp.composeLayers()
    sim.layer_core_map = myapp.composeLayersCoresMap()
    sim.bus_list = myapp.composeBusSegmentsList()
    
    sim.run()

    sim.postRun() 
    print ('multicast= ', str(multicast))



    