import sys
from gui.SENSIMgui import SENSIMgui
from PyQt5 import QtWidgets
import csv
import tkinter as tk
import platform
import os

root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
#print ("Monitor resolutuon: %d X %d "%(screen_width, screen_height))

base_dir = './gui'
if platform.system() == "Linux":
    settingFileAddr = os.path.join(base_dir,'gui_setting.csv')
    snapshotsFileAddr = os.path.join(base_dir,'snapshots_cores.csv')
else:
    settingFileAddr = "gui\gui_setting.csv"
    snapshotsFileAddr = "gui\snapshots_cores.csv"

# if(len(sys.argv) > 1):
#     settingFileAddr = sys.argv[1]

# if(len(sys.argv) > 2):
#     snapshotsFileAddr = sys.argv[2]

print("Starting Sensim ...\n")

app = QtWidgets.QApplication(sys.argv)
print("Reading setting file :wq:from \"%s\" file...\n"%settingFileAddr)
with open(settingFileAddr) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    core_cnt = 0
    for row in csv_reader:
        if(row[0][0] != '#'):
            if(row[0] == "mesh"):
                core_X = int(row[1])
                core_Y = int(row[2])
                bus_no = int(row[3])            
                ui = SENSIMgui(bus_no, core_X, core_Y, snapshotsFileAddr)
            
            if(row[0] == "core"):
                core_cnt += 1
                ui.NCCinfoUpdate(int(row[1]), int(row[2]), [row[5]], 50, 0, 100, 0, 0) # needs to be fixed to form a list from layers names.    

    ui.viewSnapshot(1)
    

sys.exit(app.exec())
