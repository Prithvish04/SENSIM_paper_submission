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

def plot_energy():
    path = "DNNs/PilotNet/sparsity_experiment/"
    num_snapshots = 1000
    thr_list = np.arange(1,151)/10
    time_step = (10e6/200e6) * 1e3 #ms
    time_axis = np.arange(1, num_snapshots+1)*time_step

    # Reading sparsity numbers
    sparsity_numbers = []
    with open(path+"python_scripts_for_plots/threshold_factor_vs_Operation_Density.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0][0]!='#':
                [threshold_factor,Operation_Density, loss] = [float(row[0]), float(row[1]), float(row[2])]
                sparsity_numbers.append([threshold_factor,Operation_Density, loss])   
    sparsity_numbers = np.array(sparsity_numbers)

    # Reading interconnect flits in each time step
    total_flits_energy = []
    for threshold_factor in thr_list:   
        flits = np.zeros(num_snapshots) 
        energy = np.zeros(num_snapshots)
        with open(path+"outputs/outputs_"+str(threshold_factor)+"xthr/snapshots_interconnects.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0][0]!='#' and float(row[0])<=num_snapshots:
                    flits[int(float(row[0]))-1] += float(row[3])
                    energy[int(float(row[0]))-1] += float(row[4])
        total_flits_energy.append([threshold_factor, np.sum(flits), np.sum(energy)])
    total_flits_energy = np.array(total_flits_energy)

    # Reading snapshots_cores
    end_proc_time=[]
    total_energy = []
    for threshold_factor in thr_list:   
        energy = np.zeros(num_snapshots) 
        unitlization = np.zeros(num_snapshots) 
        with open(path+"outputs/outputs_"+str(threshold_factor)+"xthr/snapshots_cores.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0][0]!='#' and float(row[0])<=num_snapshots:
                    energy[int(float(row[0]))-1] += float(row[7])
                    unitlization[int(float(row[0]))-1] += float(row[9])
        end_proc_time.append(np.sum(unitlization))
        total_energy.append(np.sum(energy))
    end_proc_time = np.array(end_proc_time)
    total_energy = np.array(total_energy)
    

    start_sample = 0
    end_sample=len(thr_list)
    total_core_bus_energy = (total_energy[start_sample:end_sample]+total_flits_energy[start_sample:end_sample,2])
    end_proc_time = end_proc_time[start_sample:end_sample]
    plt.plot(thr_list, total_core_bus_energy/np.max(total_core_bus_energy))
    plt.plot(thr_list, total_flits_energy[start_sample:end_sample,1]/np.max(total_flits_energy[start_sample:end_sample,1]))
    plt.plot(thr_list, end_proc_time/np.max(end_proc_time))
    plt.ylabel('Normalized Values')
    plt.xlabel('Threshold scale factor')
    plt.legend(['Energy', 'Number of events (spikes)', 'Utilization time'])
    plt.grid(b=True, axis='x')
    plt.savefig(path+'plots/EnergySpikes_vs_thrfactor.png', bbox_inches='tight', dpi=1200)
    plt.clf()
    

    plt.plot(sparsity_numbers[0:150,0], (1-sparsity_numbers[0:150,1]))
    plt.ylabel('Operation Sparsity')
    plt.xlabel('Thresholds scaling factor')
    plt.xticks(ticks=np.arange(0, 15))
    plt.grid(b=True, axis='both')
    plt.savefig(path+'plots/sparsity_vs_thrFactor.png', bbox_inches='tight', dpi=1200)
    plt.clf()




if __name__ == '__main__': 
    plot_energy()