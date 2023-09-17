import csv
from email.policy import default
import matplotlib.pyplot as plt
import numpy as np


def plot_energy():
    path = "DNNs/PilotNet/bitwidth_experiment/"
    num_snapshots = 1000
    thr_list = np.arange(1,151)/10
    time_step = (10e6/200e6) * 1e3 #ms
    time_axis = np.arange(1, num_snapshots+1)*time_step

    
    # weights
    weights_energy = []
    experiments = ['4_16_1', '8_16_1', '16_16_1']
    for experiment in experiments:   
        energy = np.zeros(num_snapshots) 
        with open(path+"outputs/outputs_"+str(experiment)+"/snapshots_cores.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0][0]!='#' and float(row[0])<=num_snapshots:
                    energy[int(float(row[0]))-1] += float(row[7])
        weights_energy.append(np.sum(energy))
    weights_energy = np.array(weights_energy)

    # states
    states_energy = []
    experiments = ['4_4_1', '4_8_1', '4_16_1']
    for experiment in experiments:   
        energy = np.zeros(num_snapshots) 
        with open(path+"outputs/outputs_"+str(experiment)+"/snapshots_cores.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0][0]!='#' and float(row[0])<=num_snapshots:
                    energy[int(float(row[0]))-1] += float(row[7])
        states_energy.append(np.sum(energy))
    states_energy = np.array(states_energy)

    # outputs
    outputs_energy = []
    experiments = ['4_16_1', '4_16_4', '4_16_8', '4_16_16']
    for experiment in experiments:   
        energy = np.zeros(num_snapshots) 
        with open(path+"outputs/outputs_"+str(experiment)+"/snapshots_cores.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0][0]!='#' and float(row[0])<=num_snapshots:
                    energy[int(float(row[0]))-1] += float(row[7])
        outputs_energy.append(np.sum(energy))
    outputs_energy = np.array(outputs_energy)
    

    default = outputs_energy[0]
    plt.bar(np.arange(0,3), weights_energy/default)
    plt.bar(np.arange(4,7), states_energy/default)
    plt.bar(np.arange(8,12), outputs_energy/default)
    plt.ylabel('Normalized energy')
    plt.xticks(ticks=[0, 1, 2, 4, 5, 6, 8, 9, 10, 11], labels=['4b', '8b', '16b', '4b', '8b', '16b', '1b', '4b', '8b', '16b'])
    plt.text(0.3,0.05,'Weights')
    plt.text(4.3,0.05,'Neurons')
    plt.text(9,0.05,'Outputs')
    plt.text(3,1.23,'Default bitwidths: W=4b, N=16b, O=1b')
    # plt.xlabel('Bitwidth resolution (defaults: W=4b, N=16b, O=1b)')
    # plt.legend(['Weights', 'Neurons', 'Output'])
    plt.grid(b=True, axis='y')
    plt.savefig(path+'plots/Energy_bitwidth.png', bbox_inches='tight', dpi=1200)
    plt.clf()




if __name__ == '__main__': 
    plot_energy()