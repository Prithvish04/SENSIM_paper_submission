import numpy as np


script_file = open('DNNs/PilotNet/SharedMem_experiment/sensim_app/run.sh', 'w')

# No_ext
script_file.write("python DNNs/PilotNet/SharedMem_experiment/sensim_app/top.py -BWext_mem:1 -Weights_ext:0.0 -Neurons_ext:0.0 &\n")

# Weights_ext
for i in range(0,8):
    script_file.write("python DNNs/PilotNet/SharedMem_experiment/sensim_app/top.py -BWext_mem:"+str(2**i)+" -Weights_ext:1.0 -Neurons_ext:0.0 &\n")
script_file.write("python DNNs/PilotNet/SharedMem_experiment/sensim_app/top.py -BWext_mem:256 -Weights_ext:1.0 -Neurons_ext:0.0\n")

# Neurons_ext
for i in range(0,8):
    script_file.write("python DNNs/PilotNet/SharedMem_experiment/sensim_app/top.py -BWext_mem:"+str(2**i)+" -Weights_ext:0.0 -Neurons_ext:1.0 &\n")
script_file.write("python DNNs/PilotNet/SharedMem_experiment/sensim_app/top.py -BWext_mem:256 -Weights_ext:0.0 -Neurons_ext:1.0\n")






script_file.close()