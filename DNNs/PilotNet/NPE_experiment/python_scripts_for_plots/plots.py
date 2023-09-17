import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_output():
    path = "DNNs/PilotNet/sparsity_experiment/outputs_5xthr/"
    fps = 25
    degree_factor = (180/np.pi)*2 # convert the output to degree
    time_axis = []
    out_membrane = []
    with open(path+"snapshots_output.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            time_axis.append(float(row[0])*1e3/200e6)
            out_membrane.append(float(row[1])*degree_factor) 


    plt.plot(time_axis, out_membrane)
    plt.ylabel('sensim output value')
    plt.xlabel('time (ms)')
    plt.legend(['SENSIM_output'])
    plt.xticks(ticks=np.arange(0, time_axis[-1], time_axis[-1]/10))
    plt.grid(b=True, axis='x')
    plt.savefig(path+'output_sensim.png', bbox_inches='tight', dpi=1200)
    plt.clf()

def plot_energy_latency():
    path = "DNNs/PilotNet/NPE_experiment/"
    num_snapshots = 3200
    NPE_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    time_step = (10e6/200e6) * 1e3 #ms
    time_axis = np.arange(1, num_snapshots+1)*time_step


    # Reading snapshots_cores
    number_of_cores = 10
    cores_energy = []
    cores_utilization = []
    for N_NPE in NPE_list:   
        energy = np.zeros([num_snapshots,number_of_cores]) 
        unitlization = np.zeros([num_snapshots,number_of_cores])
        with open(path+"outputs/outputs_"+str(N_NPE)+"xNPE/snapshots_cores.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0][0]!='#' and float(row[0])<=num_snapshots:
                    energy[int(row[0])-1,int(row[1][1:-2])] += float(row[7])
                    unitlization[int(row[0])-1,int(row[1][1:-2])] += float(row[9])
        cores_utilization.append(np.sum(unitlization,0))
        cores_energy.append(np.sum(energy,0))
    cores_utilization = np.array(cores_utilization)
    cores_energy = np.array(cores_energy)

    # Reading snapshots_interconnects
    number_of_interconnects = 10
    bus_energy = []
    bus_flits = []
    for N_NPE in NPE_list:   
        energy = np.zeros([num_snapshots,number_of_interconnects]) 
        flits = np.zeros([num_snapshots,number_of_interconnects])
        with open(path+"outputs/outputs_"+str(N_NPE)+"xNPE/snapshots_interconnects.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0][0]!='#' and float(row[0])<=num_snapshots:
                    if row[2][3]=='I':
                        energy[int(row[0])-1,0] += float(row[4])
                        flits[int(row[0])-1,0] += float(row[3])
                    elif row[2][3]!='O':
                        energy[int(row[0])-1,int(row[2][3])] += float(row[4])
                        flits[int(row[0])-1,int(row[2][3])] += float(row[3])
        bus_energy.append(np.sum(energy,0))
        bus_flits.append(np.sum(flits,0))
    bus_energy = np.array(bus_energy)
    bus_flits = np.array(bus_flits)

    # plot energy
    legend = []
    all_energy = np.sum(cores_energy,1)
    plt.plot(all_energy/np.max(all_energy), marker='*')
    legend.append('All NCCs')
    for i in range(8):
        plt.plot(cores_energy[:,i]/np.max(cores_energy[:,i]))
        legend.append('Layer'+str(i+1))
    plt.ylabel('Normalized consumed energy')
    plt.xlabel('Number of NPEs per NCC')
    plt.legend(legend)
    plt.xticks(ticks=np.arange(0, 9), labels=['1', '2', '4', '8', '16', '32', '64', '128', '256'])
    plt.grid(b=True, axis='x')
    plt.savefig(path+'plots/Energy_vs_NPE.png', bbox_inches='tight', dpi=1200)
    plt.clf()

    # plot utilization
    legend = []
    all_utilization = np.sum(cores_utilization,1)
    plt.plot(all_utilization/np.max(all_utilization), marker='*')
    legend.append('All NCCs')
    for i in range(8):
        plt.plot(cores_utilization[:,i]/np.max(cores_utilization[:,i]))
        legend.append('Layer'+str(i+1))
    plt.ylabel('Normalized utilization times')
    plt.xlabel('Number of NPEs per NCC')
    plt.legend(legend)
    plt.xticks(ticks=np.arange(0, 9), labels=['1', '2', '4', '8', '16', '32', '64', '128', '256'])
    plt.grid(b=True, axis='x')
    plt.savefig(path+'plots/Utilization_vs_NPE.png', bbox_inches='tight', dpi=1200)
    plt.clf()
    
    # plot flits
    legend = []
    all_flits = np.sum(bus_flits,1)
    plt.plot(all_flits/np.max(all_flits), marker='*')
    legend.append('All NCCs')
    for i in range(8):
        plt.plot(bus_flits[:,i]/np.max(bus_flits[:,i]))
        legend.append('Layer'+str(i+1))
    plt.ylabel('Normalized number of input events')
    plt.xlabel('Number of NPEs per NCC')
    plt.legend(legend)
    plt.xticks(ticks=np.arange(0, 9), labels=['1', '2', '4', '8', '16', '32', '64', '128', '256'])
    plt.grid(b=True, axis='x')
    plt.savefig(path+'plots/flits_vs_NPE.png', bbox_inches='tight', dpi=1200)
    plt.clf()

    # plot energy/flits
    legend = []
    energy_flit = cores_energy/bus_flits
    energy_flit_all = np.sum(energy_flit,1)
    for i in range(8):
        plt.plot(energy_flit[:,i]/np.max(energy_flit[:,i]))
        legend.append('Layer'+str(i+1))
    plt.plot(energy_flit_all/np.max(energy_flit_all), marker='*')
    legend.append('All')
    plt.ylabel('Normalized consumed energy per event')
    plt.xlabel('Number of NPEs per NCC')
    plt.legend(legend)
    plt.xticks(ticks=np.arange(0, 9), labels=['1', '2', '4', '8', '16', '32', '64', '128', '256'])
    plt.grid(b=True, axis='x')
    plt.savefig(path+'plots/EnergyPerFlit_vs_NPE.png', bbox_inches='tight', dpi=1200)
    plt.clf()

    # plot utilization/flits
    legend = []
    utilization_flit = cores_utilization/bus_flits
    utilization_flit_all = np.sum(utilization_flit,1)
    for i in range(8):
        plt.plot(utilization_flit[:,i]/np.max(utilization_flit[:,i]))
        legend.append('Layer'+str(i+1))
    plt.plot(utilization_flit_all/np.max(utilization_flit_all), marker='*')
    legend.append('All')
    plt.ylabel('Normalized NCC utilization time per each input event')
    plt.xlabel('Number of NPEs per NCC')
    plt.legend(legend)
    plt.xticks(ticks=np.arange(0, 9), labels=['1', '2', '4', '8', '16', '32', '64', '128', '256'])
    plt.grid(b=True, axis='x')
    plt.savefig(path+'plots/UtilizationPerFlit_vs_NPE.png', bbox_inches='tight', dpi=1200)
    plt.clf()



if __name__ == '__main__': 
    plot_energy_latency()