import csv
from distutils.ccompiler import show_compilers
import matplotlib.pyplot as plt
import numpy as np

def plot_output():
    path = "DNNs/PilotNet/mapping_experiment/"
    fps = 25
    degree_factor = (180/np.pi)*2 # convert the output to degree
    
    #map1
    time_axis = []
    out_membrane = []
    end_sample = 500
    with open(path+"outputs/outputs_map1/snapshots_output.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            time_axis.append(float(row[0])/200e6)
            out_membrane.append(float(row[1])*degree_factor) 
    plt.plot(time_axis[0:end_sample], out_membrane[0:end_sample])

    #map2
    time_axis = []
    out_membrane = []
    end_sample = 350
    with open(path+"outputs/outputs_map2/snapshots_output.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            time_axis.append(float(row[0])/200e6)
            out_membrane.append(float(row[1])*degree_factor) 
    plt.plot(time_axis[0:end_sample], out_membrane[0:end_sample])

    #map3
    time_axis = []
    out_membrane = []
    end_sample = 310
    with open(path+"outputs/outputs_map3/snapshots_output.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            time_axis.append(float(row[0])/200e6)
            out_membrane.append(float(row[1])*degree_factor) 
    plt.plot(time_axis[0:end_sample], out_membrane[0:end_sample])

    
    end_sample = 600
    plt.ylabel('Steering Angle')
    plt.xlabel('time (s)')
    plt.legend(['MAP1', 'MAP2', 'MAP3'])
    plt.xticks(ticks=np.arange(0, time_axis[end_sample], time_axis[end_sample]/10))
    plt.grid(b=True, axis='x')
    plt.savefig(path+'plots/output_mapping.png', bbox_inches='tight', dpi=1200)
    plt.clf()

def plot_utilization_time_per_NCC():
    path = "DNNs/PilotNet/mapping_experiment/"
    num_snapshots = 1000
    number_of_layers = 10
    time_step = (10e6/200e6) * 1e3 #ms
    time_axis = np.arange(1, num_snapshots+1)*time_step


    # Reading snapshots_cores
    experiments = ['map1', 'map2', 'map3']
    cores_utilization = []
    cores_energy = []
    for experiment in experiments:
        energy = np.zeros([num_snapshots, number_of_layers]) 
        utilization = np.zeros([num_snapshots, number_of_layers])
        with open(path+"outputs/outputs_"+experiment+"/snapshots_cores.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0][0]!='#' and float(row[0])<=num_snapshots:
                    energy[int(row[0])-1,int(row[1][1])] += float(row[7])
                    if row[1][3]=='0': utilization[int(row[0])-1,int(row[1][1])] += float(row[9])
        cores_utilization.append(np.sum(utilization,0))
        cores_energy.append(np.sum(energy,0))
    cores_utilization = np.array(cores_utilization)
    cores_energy = np.array(cores_energy)

    # plot utilization
    show_layers = 6
    labels=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10']
    plt.bar(np.arange(0, show_layers)-0.1, cores_utilization[0,0:show_layers]/np.max(cores_utilization), width=0.1)
    plt.bar(np.arange(0, show_layers)    , cores_utilization[1,0:show_layers]/np.max(cores_utilization), width=0.1)
    plt.bar(np.arange(0, show_layers)+0.1, cores_utilization[2,0:show_layers]/np.max(cores_utilization), width=0.1)
    plt.ylabel('Normalized NCC utilization time')
    plt.xlabel('Mapped Layers')
    plt.xticks(ticks=np.arange(0, show_layers), labels=labels[0:show_layers])
    plt.legend(['MAP1', 'MAP2', 'MAP3'])
    plt.savefig(path+'plots/Utilization_per_Layer_mapping.png', bbox_inches='tight', dpi=1200)
    plt.clf()

    # plot Energy
    show_layers = 6
    labels=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10']
    labels[show_layers]='ALL'
    cores_energy[:,show_layers]=np.sum(cores_energy,1)
    plt.bar(np.arange(0, show_layers+1)-0.1, cores_energy[0,0:show_layers+1]/np.max(cores_energy), width=0.1)
    plt.bar(np.arange(0, show_layers+1)    , cores_energy[1,0:show_layers+1]/np.max(cores_energy), width=0.1)
    plt.bar(np.arange(0, show_layers+1)+0.1, cores_energy[2,0:show_layers+1]/np.max(cores_energy), width=0.1)
    plt.ylabel('Normalized Layer Energy consumption')
    plt.xlabel('Mapped Layers')
    plt.xticks(ticks=np.arange(0, show_layers+1), labels=labels[0:show_layers+1])
    plt.legend(['MAP1', 'MAP2', 'MAP3'])
    plt.savefig(path+'plots/Energy_per_Layer_mapping.png', bbox_inches='tight', dpi=1200)
    plt.clf()



if __name__ == '__main__': 
    plot_utilization_time_per_NCC()
    # plot_output()