import os, sys
sys.path.append(os.getcwd())
from libraries.simulator import Simulator

from iodefs  import PilotNetIO as IO
from appdefs import PilotNet_app as App


# shared memeory BW (bits per cycle)
BWext_mem = 32
Weights_ext= 0
Neurons_ext=0

if len(sys.argv)>0:
    for args in sys.argv:
        print(args)
        if args[0:len('-BWext_mem:')]=='-BWext_mem:':
            BWext_mem= int(args[len('-BWext_mem:'):])
        if args[0:len('-Weights_ext:')]=='-Weights_ext:':
            Weights_ext= float(args[len('-Weights_ext:'):])
        if args[0:len('-Neurons_ext:')]=='-Neurons_ext:':
            Neurons_ext= float(args[len('-Neurons_ext:'):])
print("BWext_mem is adjusted to:", BWext_mem)
print("Weights_ext is adjusted to:", Weights_ext)
print("Neurons_ext is adjusted to:", Neurons_ext)
    

if __name__ == '__main__':  
    print ('BWext_mem= ', BWext_mem) 
    print ('Weights_ext= ', Weights_ext)
    print ('Neurons_ext= ', Neurons_ext)
    sim = Simulator()
    myapp = App(sim, BWext_mem, Weights_ext, Neurons_ext)        
    sim.setIO(IO(sim))
    sim.setHW()

    myapp.composeLayers()
    sim.layer_core_map = myapp.composeLayersCoresMap()
    sim.bus_list = myapp.composeBusSegmentsList()
    
    sim.run()

    sim.postRun()
    print ('BWext_mem= ', BWext_mem) 
    print ('Weights_ext= ', Weights_ext)
    print ('Neurons_ext= ', Neurons_ext) 



    