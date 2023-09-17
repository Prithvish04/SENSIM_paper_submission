import csv
from distutils.ccompiler import show_compilers
from turtle import width
import matplotlib.pyplot as plt
import numpy as np

def plot_output():
    path = "DNNs/PilotNet/NoC_experiment/"
    fps = 25
    degree_factor = (180/np.pi)*2 # convert the output to degree
    
    #,multicast
    time_axis = []
    out_membrane = []
    end_sample = 1000
    with open(path+"outputs/outputs_multicast/snapshots_output.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            time_axis.append(float(row[0])/200e6)
            out_membrane.append(float(row[1])*degree_factor) 
    plt.plot(time_axis[0:end_sample], out_membrane[0:end_sample])

    #unicast
    time_axis = []
    out_membrane = []
    end_sample = 1000
    with open(path+"outputs/outputs_unicast/snapshots_output.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            time_axis.append(float(row[0])/200e6)
            out_membrane.append(float(row[1])*degree_factor) 
    plt.plot(time_axis[0:end_sample], out_membrane[0:end_sample])

    start_sample = 100
    end_sample = 500
    plt.ylabel('Steering Angle')
    plt.xlabel('time (s)')
    plt.legend(['Multicast', 'Unicast'])
    plt.xticks(ticks=np.arange(0, time_axis[end_sample], time_axis[end_sample]/10))
    plt.xlim([time_axis[start_sample],time_axis[end_sample]])
    plt.grid(b=True, axis='x')
    plt.savefig(path+'plots/output_NoC.png', bbox_inches='tight', dpi=1200)
    plt.clf()

def plot_flits():
    path = "DNNs/PilotNet/NoC_experiment/"
    num_snapshots = 4000
    number_of_layers = 10
    time_step = (10e6/200e6) * 1e3 #ms
    time_axis = np.arange(1, num_snapshots+1)*time_step

    # Reading interconnect flits in each time step
    experiments = ['multicast', 'unicast']
    total_flits = []
    total_energy = []
    for experiment in experiments:   
        flits = np.zeros([num_snapshots, 11]) 
        energy = np.zeros([num_snapshots, 11])
        with open(path+"outputs/outputs_"+str(experiment)+"/snapshots_interconnects.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0][0]!='#' and float(row[0])<=num_snapshots and row[2][3]!='I' and row[2][3]!='O':
                    flits[int(float(row[0]))-1, int(row[2][3])-1] += float(row[3])
                    energy[int(float(row[0]))-1, int(row[2][3])-1] += float(row[4])
        total_flits.append(np.sum(flits,0))
        total_energy.append(np.sum(energy,0))
    total_flits = np.array(total_flits)
    total_energy = np.array(total_energy)

    # plot flits
    show_layers = 4
    labels=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'ALL']
    labels[show_layers]='ALL'
    total_flits[:,show_layers]=np.sum(total_flits,1)
    saving = total_flits[1,:]/total_flits[0,:]
    plt.bar(np.arange(0, show_layers+1)-0.1, total_flits[0,0:show_layers+1]/np.max(total_flits),width=0.2)
    plt.bar(np.arange(0, show_layers+1)+0.1, total_flits[1,0:show_layers+1]/np.max(total_flits),width=0.2)
    for i in range(show_layers+1):
        plt.text(i,total_flits[1,i]/np.max(total_flits)+0.015,str(saving[i])[0:4]+'x')
    plt.ylabel('Normalized number of communicated bits')
    plt.xlabel('Layers')
    plt.xticks(ticks=np.arange(0, show_layers+1), labels=labels[0:show_layers+1])
    plt.legend(experiments)
    # plt.grid(b=True, axis='y')
    plt.savefig(path+'plots/flits_NoC.png', bbox_inches='tight', dpi=1200)
    plt.clf()




if __name__ == '__main__': 
    plot_flits()
    # plot_output()