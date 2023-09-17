import os, sys
sys.path.append(os.getcwd())
from libraries.simulator import Simulator

from iodefs  import PilotNetIO as IO
from appdefs import PilotNet_app as App

N_NPE= 1.0
if len(sys.argv)>0:
    for args in sys.argv:
        print(args)
        if args[0:len('-NPE:')]=='-NPE:':
            N_NPE= int(float(args[len('-NPE:'):]))
print("N_NPE is adjusted to:", N_NPE)
    

if __name__ == '__main__':  
    print ('N_NPE= ', N_NPE) 
    sim = Simulator()
    myapp = App(sim, N_NPE)        
    sim.setIO(IO(sim))
    sim.setHW()

    myapp.composeLayers()
    sim.layer_core_map = myapp.composeLayersCoresMap()
    sim.bus_list = myapp.composeBusSegmentsList()
    
    sim.run()

    sim.postRun()
    print ('N_NPE= ', N_NPE)  



    