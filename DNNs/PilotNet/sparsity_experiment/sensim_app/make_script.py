import numpy as np


script_file = open('DNNs/PilotNet/sparsity_experiment/sensim_app/run.sh', 'w')
for i in range(61,151):
    if i%5!=0:
        script_file.write("python DNNs/PilotNet/sparsity_experiment/sensim_app/top.py -10xthr:"+str(i)+" &\n")
    else:
        script_file.write("python DNNs/PilotNet/sparsity_experiment/sensim_app/top.py -10xthr:"+str(i)+" \n")

script_file.close()