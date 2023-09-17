import csv
from distutils.ccompiler import show_compilers
from turtle import width
import matplotlib.pyplot as plt
import numpy as np


# Experiment conditions:
# 500 frames, 1e9 simulation cycles, 1e6 evaluation cycles
# Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz
def plot_output_timestep():
    path = "DNNs/PilotNet/parallelization_speed_experiment/outputs/time_measurement_results.csv"

    

    #unicast
    num_threads = []
    time_step = []
    exe_time = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            num_threads.append(float(row[0]))
            time_step.append(float(row[1])/1000.0)
            exe_time.append(float(row[2])/60.0)
        
    plt.plot([1,2,3,4],    exe_time[0:4])
    plt.plot([1,2,3,4],    exe_time[4:8])
    plt.plot([1,2,3,4],   exe_time[8:12])
    plt.plot([1,2,3,4],  exe_time[12:16])

    
    plt.ylabel('Execution time (min)')
    plt.xlabel('time step (KCycles)')
    plt.legend(['Single Thread', 'Double Threads', 'Triple Threads', 'Quad Threads'])
    plt.xticks(ticks=[1,2,3,4], labels=['0.1', '1', '10', '100'],)
    # plt.xlim([time_axis[start_sample],time_axis[end_sample]])
    plt.grid(b=True, axis='x')
    plt.savefig('DNNs/PilotNet/parallelization_speed_experiment/plots/parallelism.png', bbox_inches='tight', dpi=1200)
    plt.clf()

def plot_output_threads():
    path = "DNNs/PilotNet/parallelization_speed_experiment/outputs/time_measurement_results.csv"

    

    #unicast
    num_threads = []
    time_step = []
    exe_time = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            num_threads.append(float(row[0]))
            time_step.append(float(row[1])/1000.0)
            exe_time.append(float(row[2])/60.0)
        
    plt.plot([1,2,3,4],    exe_time[0::4])
    plt.plot([1,2,3,4],    exe_time[1::4])
    plt.plot([1,2,3,4],   exe_time[2::4])
    plt.plot([1,2,3,4],  exe_time[3::4])

    
    plt.ylabel('number of threads')
    plt.xlabel('time step (KCycles)')
    plt.legend(['time_step=0.1kCycle', 'time_step=1kCycle', 'time_step=10kCycle', 'time_step=100kCycle'])
    # plt.xticks(ticks=[1,2,3,4], labels=['0.1', '1', '10', '100'],)
    # plt.xlim([time_axis[start_sample],time_axis[end_sample]])
    plt.grid(b=True, axis='x')
    plt.savefig('DNNs/PilotNet/parallelization_speed_experiment/plots/parallelism.png', bbox_inches='tight', dpi=1200)
    plt.clf()

if __name__ == '__main__': 
    plot_output_timestep()