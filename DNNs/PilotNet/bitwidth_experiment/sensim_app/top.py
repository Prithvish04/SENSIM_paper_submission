import os, sys
sys.path.append(os.getcwd())
from libraries.simulator import Simulator

from iodefs  import PilotNetIO as IO
from appdefs import PilotNet_app as App

Weights=4
States=16
Outputs=1
if len(sys.argv)>0:
    for args in sys.argv:
        print(args)
        if args[0:len('-Weights:')]=='-Weights:':
            Weights= int(args[len('-Weights:'):])
        if args[0:len('-States:')]=='-States:':
            States= int(args[len('-States:'):])
        if args[0:len('-Outputs:')]=='-Outputs:':
            Outputs= int(args[len('-Outputs:'):])
print("Weights_State_Output is adjusted to: "+str(Weights)+"_"+str(States)+"_"+str(Outputs))
    

if __name__ == '__main__':  
    print("Weights_State_Output: "+str(Weights)+"_"+str(States)+"_"+str(Outputs))
    sim = Simulator()
    myapp = App(sim, Weights, States, Outputs)        
    sim.setIO(IO(sim))
    sim.setHW()

    myapp.composeLayers()
    sim.layer_core_map = myapp.composeLayersCoresMap()
    sim.bus_list = myapp.composeBusSegmentsList()
    
    sim.run()

    sim.postRun()
    print("Weights_State_Output: "+str(Weights)+"_"+str(States)+"_"+str(Outputs))



    