import csv
from distutils.ccompiler import show_compilers
import matplotlib.pyplot as plt
import numpy as np

def plot_output():
    path = "DNNs/PilotNet/eval_experiment/"
    fps = 25
    degree_factor = (180/np.pi)*2 # convert the output to degree
    
    #Async
    time_axis = []
    out_membrane = []
    end_sample = 500
    with open(path+"outputs/outputs_Async1/snapshots_output.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            time_axis.append(float(row[0])/200e6)
            out_membrane.append(float(row[1])*degree_factor) 
    plt.plot(time_axis[0:end_sample], out_membrane[0:end_sample])

    #TimeStep
    time_axis = []
    out_membrane = []
    end_sample = 350
    with open(path+"outputs/outputs_TimeStep/snapshots_output.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            time_axis.append(float(row[0])/200e6)
            out_membrane.append(float(row[1])*degree_factor) 
    plt.plot(time_axis[0:end_sample], out_membrane[0:end_sample])

    #TimeStep_flag
    time_axis = []
    out_membrane = []
    end_sample = 310
    with open(path+"outputs/outputs_TimeStep_flag/snapshots_output.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            time_axis.append(float(row[0])/200e6)
            out_membrane.append(float(row[1])*degree_factor) 
    plt.plot(time_axis[0:end_sample], out_membrane[0:end_sample])

    
    end_sample = 600
    plt.ylabel('Steering Angle')
    plt.xlabel('time (s)')
    plt.legend(['Event-Based', 'Periodic', 'Selective Periodic'])
    plt.xticks(ticks=np.arange(0, time_axis[end_sample], time_axis[end_sample]/10))
    plt.grid(b=True, axis='x')
    plt.savefig(path+'plots/output_eval.png', bbox_inches='tight', dpi=1200)
    plt.clf()

def plot_utilization_time_per_NCC():
    path = "DNNs/PilotNet/eval_experiment/"
    num_snapshots = 4000
    number_of_layers = 10
    time_step = (10e6/200e6) * 1e3 #ms
    time_axis = np.arange(1, num_snapshots+1)*time_step


    # Reading snapshots_cores
    experiments = ['Async1', 'TimeStep_flag', 'TimeStep']
    cores_energy_in_time = []
    cores_utilization_in_time = []
    for experiment in experiments:
        energy = np.zeros([num_snapshots, number_of_layers]) 
        utilization = np.zeros([num_snapshots, number_of_layers])
        with open(path+"outputs/outputs_"+experiment+"/snapshots_cores.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0][0]!='#' and float(row[0])<=num_snapshots:
                    energy[int(row[0])-1,int(row[1][1])] += float(row[7])
                    if row[1][3]=='0': utilization[int(row[0])-1,int(row[1][1])] += float(row[9])
        cores_energy_in_time.append(np.sum(energy,1))
        cores_utilization_in_time.append(np.sum(utilization,1))
    cores_energy_in_time = np.array(cores_energy_in_time)
    cores_utilization_in_time = np.array(cores_utilization_in_time)


    end_sample = 1800
    # plot enegy
    cores_energy_in_time = cores_energy_in_time/np.max(cores_energy_in_time)
    plt.plot(np.arange(0, end_sample)/200, cores_energy_in_time[0,0:end_sample])
    plt.plot(np.arange(0, end_sample)/200, cores_energy_in_time[1,0:end_sample])
    plt.plot(np.arange(0, end_sample)/200, cores_energy_in_time[2,0:end_sample])
    plt.text(4, 0.03, '-> Event-Based ='+str("{:.0e}".format(cores_energy_in_time[0,end_sample])))
    plt.text(4, 0.1, '-> Selective ='+str("{:.0e}".format(cores_energy_in_time[1,end_sample])))
    plt.text(4, 0.17, '-> Periodic ='+str("{:.0e}".format(cores_energy_in_time[2,end_sample])))
    plt.text(4, 0.24, 'Idle dynamic power consumtion')

    plt.ylabel('Normalized power consumption')
    plt.xlabel('Time(s)')
    plt.legend(['Event-Based', 'Selective', 'Periodic'])
    plt.savefig(path+'plots/Power_time_eval.png', bbox_inches='tight', dpi=1200)
    plt.clf()

    # plot Utilization
    # plt.plot(np.arange(0, end_sample)/200, cores_utilization_in_time[0,0:end_sample]/np.max(cores_utilization_in_time))
    # plt.plot(np.arange(0, end_sample)/200, cores_utilization_in_time[1,0:end_sample]/np.max(cores_utilization_in_time))
    # plt.plot(np.arange(0, end_sample)/200, cores_utilization_in_time[2,0:end_sample]/np.max(cores_utilization_in_time))
    # plt.ylabel('Normalized NCCs utilization in time')
    # plt.xlabel('Time(s)')
    # plt.legend(['Event-Based', 'Periodic', 'Selective Periodic'])
    # plt.savefig(path+'plots/Utilization_time_eval.png', bbox_inches='tight', dpi=1200)
    # plt.clf()



    # Reading interconnect flits in each time step
    experiments = ['Async1', 'TimeStep', 'TimeStep_flag']
    bus_energy_in_time = []
    bus_flits_in_time = []
    for experiment in experiments:
        energy = np.zeros([num_snapshots]) 
        flits = np.zeros([num_snapshots])
        with open(path+"outputs/outputs_"+experiment+"/snapshots_interconnects.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0][0]!='#' and float(row[0])<=num_snapshots:
                    energy[int(row[0])-1] += float(row[4])
                    flits[int(row[0])-1] += float(row[3])
        bus_energy_in_time.append(energy)
        bus_flits_in_time.append(flits)
    bus_energy_in_time = np.array(bus_energy_in_time)
    bus_flits_in_time = np.array(bus_flits_in_time)

    # plot Utilization
    # plt.plot(np.arange(0, end_sample)/200, bus_flits_in_time[0,0:end_sample]/np.max(bus_flits_in_time))
    # plt.plot(np.arange(0, end_sample)/200, bus_flits_in_time[1,0:end_sample]/np.max(bus_flits_in_time))
    # plt.plot(np.arange(0, end_sample)/200, bus_flits_in_time[2,0:end_sample]/np.max(bus_flits_in_time))
    # plt.ylabel('Normalized event communication in time')
    # plt.xlabel('Time(s)')
    # plt.legend(['Event-Based', 'Periodic', 'Selective Periodic'])
    # plt.savefig(path+'plots/Flit_time_eval.png', bbox_inches='tight', dpi=1200)
    # plt.clf()




if __name__ == '__main__': 
    plot_utilization_time_per_NCC()
    # plot_output()