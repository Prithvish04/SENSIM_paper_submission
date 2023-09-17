import csv
from distutils.ccompiler import show_compilers
import matplotlib.pyplot as plt
import numpy as np

def plot_output():
    path = "DNNs/PilotNet/evcompress_experiment/"
    fps = 25
    degree_factor = (180/np.pi)*2 # convert the output to degree
    
    #singleflit
    time_axis = []
    out_membrane = []
    end_sample = 500
    with open(path+"outputs/outputs_singleflit/snapshots_output.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            time_axis.append(float(row[0])/200e6)
            out_membrane.append(float(row[1])*degree_factor) 
    plt.plot(time_axis[0:end_sample], out_membrane[0:end_sample])

    #multiflit
    time_axis = []
    out_membrane = []
    end_sample = 350
    with open(path+"outputs/outputs_multiflit/snapshots_output.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            time_axis.append(float(row[0])/200e6)
            out_membrane.append(float(row[1])*degree_factor) 
    plt.plot(time_axis[0:end_sample], out_membrane[0:end_sample])

    
    end_sample = 600
    plt.ylabel('Steering Angle')
    plt.xlabel('time (s)')
    plt.legend(['Single Spike packet', 'Multi Spike Packet'])
    plt.xticks(ticks=np.arange(0, time_axis[end_sample], time_axis[end_sample]/10))
    plt.grid(b=True, axis='x')
    plt.savefig(path+'plots/output_evcompress.png', bbox_inches='tight', dpi=1200)
    plt.clf()

def plot_utilization_time_per_NCC():
    path = "DNNs/PilotNet/evcompress_experiment/"
    num_snapshots = 1000
    number_of_layers = 11
    time_step = (10e6/200e6) * 1e3 #ms
    time_axis = np.arange(1, num_snapshots+1)*time_step


    # Reading interconnect flits in each time step
    experiments = ['singleflit', 'multiflit']
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
    show_layers = 9
    labels=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'ALL']
    labels[show_layers]='ALL'
    total_flits[:,show_layers]=np.sum(total_flits,1)
    saving = total_flits[1,:]/total_flits[0,:]
    plt.bar(np.arange(0, show_layers+1), total_flits[0,0:show_layers+1]/np.max(total_flits))
    plt.bar(np.arange(0, show_layers+1), total_flits[1,0:show_layers+1]/np.max(total_flits))
    for i in range(show_layers+1):
        plt.text(i-0.3,total_flits[0,i]/np.max(total_flits)+0.015,str(saving[i])[0:4])
    plt.ylabel('Normalized number of communicated flits')
    plt.xlabel('Layers')
    plt.xticks(ticks=np.arange(0, show_layers+1), labels=labels[0:show_layers+1])
    plt.legend(['Single Spike Packet', 'Multi Spike Packet'])
    # plt.grid(b=True, axis='y')
    plt.savefig(path+'plots/Flits_evcompress.png', bbox_inches='tight', dpi=1200)
    plt.clf()


if __name__ == '__main__': 
    plot_utilization_time_per_NCC()
    plot_output()