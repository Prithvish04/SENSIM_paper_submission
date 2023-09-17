import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_output():
    path = "DNNs/PilotNet/RegFile_experiment/"
    fps = 25
    degree_factor = (180/np.pi)*2 # convert the output to degree
    time_axis = []
    out_membrane = []
    end_sample = 1000
    for exp in range(7):
        n_reg = int(2**exp)
        with open(path+"outputs/outputs_"+str(n_reg)+"/snapshots_output.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                time_axis.append(float(row[0])*1e3/200e6)
                out_membrane.append(float(row[1])*degree_factor) 
        plt.plot(time_axis[0:end_sample], out_membrane[0:end_sample])

    plt.ylabel('Steering Angle')
    plt.xlabel('time (s)')
    plt.legend(['1', '2', '4', '8', '16', '32', '64'])
    plt.xticks(ticks=np.arange(0, time_axis[end_sample], time_axis[end_sample]/10))
    plt.grid(b=True, axis='x')
    plt.savefig(path+'plots/output_RegFile.png', bbox_inches='tight', dpi=1200)
    plt.clf()

def plot_energy():
    path = "DNNs/PilotNet/RegFile_experiment/"
    num_snapshots = 1000
    num_reg_files = [1,2,4,8,16,32,64]
    time_step = (10e6/200e6) * 1e3 #ms
    time_axis = np.arange(1, num_snapshots+1)*time_step

    # Reading snapshots_cores
    total_energy = []
    total_energy_mem =[]
    total_utilization =[]
    for num_reg_file in num_reg_files:   
        energy = np.zeros(num_snapshots) 
        energy_mem = np.zeros(num_snapshots) 
        unitlization = np.zeros(num_snapshots) 
        with open(path+"outputs/outputs_"+str(num_reg_file)+"/snapshots_cores.csv") as csv_file:
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
    

    plt.bar(np.arange(len(num_reg_files)), total_energy/np.max(total_energy))
    # plt.bar(np.arange(len(num_reg_files)), total_energy_mem/np.max(total_energy))
    plt.ylabel('Normalized energy consumption')
    plt.xlabel('Size of the Register File for spike storage')
    # plt.legend(['Total Energy', 'Data Memory Energy'])
    plt.xticks(ticks=np.arange(len(num_reg_files)), labels=['1', '2', '4', '8', '16', '32', '64'])
    plt.grid(b=True, axis='y')
    plt.savefig(path+'plots/Energy_RegFile.png', bbox_inches='tight', dpi=1200)
    plt.clf()
    





if __name__ == '__main__': 
    # plot_output()
    plot_energy()