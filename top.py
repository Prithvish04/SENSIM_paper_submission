from libraries.simulator import Simulator
# from appdefs import mnistapp as App
# from iodefs import mnistIO as IO
from appdefs import PilotNet_app as App
from iodefs  import PilotNetIO as IO

if __name__ == '__main__':    
    sim = Simulator('./gui/') 
    app = App(sim)
    sim.setIO(IO(sim,'./gui/'))
    sim.setHW()

    app.composeLayers()
    sim.layer_core_map = app.composeLayersCoresMap()
    sim.bus_list = app.composeBusSegmentsList()
    time = sim.run()
    sim.postRun()
    print('\rSENeCA processing Time: %.3E'%(time), end='')





    