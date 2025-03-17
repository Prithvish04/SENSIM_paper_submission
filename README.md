# SENSIM: 

Codebase for the SENSIM simulator for emulation of Spiking Neural Network on SENeCA. The simulator uses certain dependencies. The easiest way to recreate the environment and play with the tool is to use annaconda to create the environment using the tf2.yml file.

NOTE: This code is only experimental and therefore not comes with no liability (particularly in what relates to its quality or efficiency)

## Setting up the environment with annaconda 

- ``` conda env create -f tf2.yml```

## Application Dataset Setup

For the purpose of experimentation the we use the MNIST Dataset and the PilotNet DataSet
Default location of the dataset searched by the scripts in ```DNNs/<application>/dataset```  folder.
SENSIM is hardcoded to work with PilotNet but the application can be modified in the ```appdefs.py``` and ```iodefs.py``` 


## General

- Dir ```libraries/``` contains the SENSIM simulator code
- Dir ```gui/``` contains the code for gui and is also the default location for snapshots collected during simulation 
- Dir ```DNNs/``` contains the DNN applications (dataset, model in keras, SNN, weights and model parameters)
- File ```top.py``` is the primary script to execute the simulator
- File ```startGui.py``` is the script used to take data from the simulation and execute the GUI
- File ```appdefs.py``` is the file where application layers get defined and manually mapped to SENSIM
- File ```iodefs.py``` is the where the input dataset to the application mapped to SENSIM is defined
- File ```libraries/core.py``` is where the functionality of a single neurosynaptic core execution is defined.
- File ```libraries/simulator.py``` is where the application is mapped to different neurosynaptic cores and is executed as a whole
- File ```libraries/parameters.py``` is where hardware, simulation, energy, timing and commucation parameters are defined
- File ```libraries/interconnect.py``` is where the interconnect framwork which connects the neurosynaptic core is defined 
- File ```libraries/Queue.py``` is where the queueing framwork for every neurosynaptic core is defined. 
- File ```libraries/DNN_layers.py``` is where the layer framwork for the DNN is defined.

## Execution of scripts

The execution of the simulator is majorly defined in 2 phases. The simulation of a SNN or DNN application on SENSIM.

- ``` python top.py ```

Visualization of the execution on a GUI 

- ``` python startGui.py ```

## Experiments

A few experiments with PilotNet and SENSIM were conducted and placed under the ```DNNs/PilotNet/*_experiments``` folder


- Dir ```bitwidth_experiment``` varies the bitwidth / precision for Weights, Neuron Outputs and thresholds.
- Dir ```eval_experiment``` varies the neuron evaluation type during simulation.
- Dir ```NoC_experiment``` varies with the multicast and unicast of events from 1 layer to another.
- Dir ```parallel_speed_experiment``` varies the number of physical cores for neurosynaptic cores and NoC interconnect
- Dir ```SharedMem_experiment``` varies the Bandwidth from the external memory.
- Dir ```sparsity_experiment``` varies the sparsity by varying the thresholds for the layer.
- Dir ```NPE_experiment``` varies the number of Neural Processing Elements in the neurosynaptic core.
- Dir ```mapping_experiment``` varies different mapping for PilotNet.

### PilotNet 

#### Dataset download and preparation 

To prepare the dataset:
```
$ cd DNNs/PilotNet/dataset
$ wget -t https://drive.google.com/file/d/1PZWa6H0i1PCH9zuYcIh5Ouk_p-9Gh58B/
$ unzip driving_dataset.zip -d .
```

We have provided a small dataset of 250 images for testing 
```
$ unzip pilotnet_0_250.zip -d .
```

#### SNN Model and DAL layer

Training the PilotNet with DAL layer to extract sparsity is detailed under ``` DNNs/PilotNet/PilotNet_Delta.py```

NOTE: The SNN model is already trained and the weights and thresholds are already provided.

### MNIST 

#### Model and quantization

Training and quantization of MNIST is specified under ``` DNNs/4layer_CONV_MNIST/```

- ``` 4layer_CONV_MNIST.py```
- ``` weight_quantization.py ```

NOTE: The model is already trained and weights are already provided for testing.

`Individual Contributions:`
```
Amirreza Yousefzadeh (SENSIM mathematical modelling, analytics build, paper reviewing)
Prithvish Vijaykumar Nembhani (fixing bugs, making it compatible with SENMap (Mapper), slides, poster, comparing the tool against other baseline tools, paper writing, experimentation and analytics)
Mohammad Taghighi (SENSIM modelling)
Guanghi Tang (Paper reviewing and suggesting changes)
Gert-Jan Van Schaik (Paper reviewing and suggesting changes)
Manolis Sifalakis (Paper reviewing and suggesting changes)
Kanishkan Vadivel (Paper reviewing and suggesting changes)
Zaid Al-Ars (Paper reviewing and suggesting changes)
```

`If you like the project and use it for your research, you can use the link for citing us. `
```
@INPROCEEDINGS{10651400,
  author={Nembhani, Prithvish and Vadivel, Kanishkan and Tang, Guangzhi and Tahghighi, Mohammad and van Schaik, Gert-Jan and Sifalakis, Manolis and Al-Ars, Zaid and Yousefzadeh, Amirreza},
  booktitle={2024 International Joint Conference on Neural Networks (IJCNN)}, 
  title={SENSIM: An Event-driven Parallel Simulator for Multi-core Neuromorphic Systems}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  keywords={Neuromorphics;Nonvolatile memory;Multicore processing;Neurons;Spiking neural networks;Benchmark testing;Silicon;Hardware-aware Simulation;Neuromorphic;SENECA;SENSIM;SNN/DNN inference},
  doi={10.1109/IJCNN60899.2024.10651400}}

```
