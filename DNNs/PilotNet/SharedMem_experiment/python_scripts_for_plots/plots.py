import csv
from turtle import width
import matplotlib.pyplot as plt
import numpy as np

def plot_output():
    path = "DNNs/PilotNet/SharedMem_experiment/"
    fps = 25
    degree_factor = (180/np.pi)*2 # convert the output to degree
    
    start_sample = 100
    end_sample = 350
    experiments = ['BW1_W0.0_N0.0', 'BW32_W1.0_N1.0', 'BW64_W1.0_N1.0', 'BW128_W1.0_N1.0']
    # no shared memeory use
    for exp in experiments:
        time_axis = []
        out_membrane = []
        with open(path+"outputs/outputs_"+exp+"/snapshots_output.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                time_axis.append(float(row[0])*1e3/200e6)
                out_membrane.append(float(row[1])*degree_factor) 
        plt.plot(time_axis[start_sample:end_sample], out_membrane[start_sample:end_sample])

    plt.ylabel('Steering Angle')
    plt.xlabel('time (s)')
    plt.legend(['Local memory', '32b/cycle', '64b/cycle', '128b/cycle'])
    plt.xticks(ticks=np.arange(time_axis[start_sample], time_axis[end_sample], time_axis[end_sample]/10-time_axis[start_sample]/10))
    plt.grid(b=True, axis='x')
    plt.savefig(path+'plots/output_SharedMem.png', bbox_inches='tight', dpi=1200)
    plt.clf()

def plot_energy():
    path = "DNNs/PilotNet/SharedMem_experiment/"
    num_snapshots = 1000
    experiments = ['BW1_W0.0_N0.0', 'BW64_W1.0_N0.0', 'BW64_W0.0_N1.0', 'BW64_W1.0_N1.0']
    time_step = (10e6/200e6) * 1e3 #ms

    # Reading snapshots_cores
    total_energy = []
    total_energy_mem =[]
    total_utilization =[]
    for experiment in experiments:   
        energy = np.zeros(num_snapshots) 
        energy_mem = np.zeros(num_snapshots) 
        unitlization = np.zeros(num_snapshots) 
        with open(path+"outputs/outputs_"+experiment+"/snapshots_cores.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0][0]!='#' and float(row[0])<=num_snapshots:
                    energy[int(float(row[0]))-1] += float(row[7])
                    unitlization[int(float(row[0]))-1] += float(row[9])
                    energy_mem[int(float(row[0]))-1] += float(row[14])
        total_utilization.append(np.sum(unitlization))
        total_energy.append(np.sum(energy))
        total_energy_mem.append(np.sum(energy_mem))
    total_utilization = np.array(total_utilization)
    total_energy = np.array(total_energy)
    total_energy_mem = np.array(total_energy_mem)
    

    plt.bar(np.arange(len(experiments))+0.1, total_energy/np.min(total_energy), width=0.2)
    plt.bar(np.arange(len(experiments))-0.1, total_utilization/np.min(total_utilization), width=0.2)
    plt.ylabel('Normalized energy/time consumption')
    plt.legend(['Energy', 'Time'])
    plt.xticks(ticks=np.arange(len(experiments)), labels=['All local', 'Weights', 'Neurons', 'Both'])
    plt.xlabel('Mapping weights/neurons in the shared memeory')
    for i in range(len(experiments)):
        plt.text(i+0.0,total_energy[i]/np.min(total_energy)+0.3,str(total_energy[i]/np.min(total_energy))[0:3])
    for i in range(len(experiments)):
        plt.text(i-0.2,total_utilization[i]/np.min(total_utilization)+0.3,str(total_utilization[i]/np.min(total_utilization))[0:3])
    # plt.grid(b=True, axis='y')
    plt.savefig(path+'plots/Energy_SharedMem.png', bbox_inches='tight', dpi=1200)
    plt.clf()
    





if __name__ == '__main__': 
    # plot_output()
    plot_energy()