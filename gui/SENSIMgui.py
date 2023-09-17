import sys
import math
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QLineEdit, QLabel
from gui.senecaTile import senecaTile
import csv
from time import sleep
import threading

class SENSIMgui(QtWidgets.QMainWindow):
    buswidth = 20
    NccWidth = 80
    offset = 20
    NrSnapshots = 1800

    def __init__(self, nrSegBusses, nrCoreX, nrCoreY, snapshotsFileAddr):
        self.NCCgrid = []*nrCoreX
        self.NCCline = []*nrCoreY
        super().__init__()
        self.snapshotsFileAddr = snapshotsFileAddr
        canvasWidth  = (self.NccWidth+((nrSegBusses+1)*self.buswidth))*nrCoreX + ((nrSegBusses+1)*self.buswidth)
        canvasHeight = (self.NccWidth+((nrSegBusses+1)*self.buswidth))*nrCoreY + ((nrSegBusses+1)*self.buswidth)+self.offset
        self.senecaLabel = QtWidgets.QLabel()
        self.senecaLabel.setGeometry(QtCore.QRect(0, 0, canvasWidth, canvasHeight))
        canvas = QtGui.QPixmap(canvasWidth, canvasHeight)
        canvas.fill(QtGui.QColor("gray").lighter(150))
        self.senecaLabel.setPixmap(canvas)
        self.setCentralWidget(self.senecaLabel)
        self.setWindowTitle(" SENeCA Simulator ")
        for x in range(nrCoreX):
            for y in range(nrCoreY):
                self.NCCline.append( senecaTile(self.senecaLabel, nrSegBusses, self.buswidth, self.NccWidth, x, y, self.offset) )
            self.NCCgrid.append(self.NCCline)
            self.NCCline = []*nrCoreY
        count = 0
        for x in range(nrCoreX):
            for y in range(nrCoreY):
                nameList = ["C" + str(count)]
                self.setNameNCC(x, y, nameList)
                count += 1

        self.loadAllSnapshots()
        self.NrSnapshots = len(self.allSnapshots)

        self.snapshotLB = QLabel(self.senecaLabel)
        self.snapshotLB.setText(str("Snapshot ID:"))
        self.snapshotLB.move(5, 6);        

        self.snapshotLE = QLineEdit(self.senecaLabel)
        self.snapshotLE.setText(str("1"))
        self.snapshotLE.move(75+10, 2);
        self.snapshotLE.resize(50, 25);
        
        self.snapshotLB = QLabel(self.senecaLabel)
        self.snapshotLB.setText(str(("out of %d snapshots")%self.NrSnapshots))
        self.snapshotLB.move(140, 6);
                        
        self.prevSnapshotPB = QPushButton(self.senecaLabel)
        self.prevSnapshotPB.setText(str("prev."))
        self.prevSnapshotPB.setStyleSheet("QPushButton { background-color: grey; }\n"
                      "QPushButton:enabled { background-color: rgb(153,208,230); }\n")
        self.prevSnapshotPB.move(140+130, 0);
        self.prevSnapshotPB.resize(40, 25);
        self.prevSnapshotPB.clicked.connect(self.handlePrevSnapshotClicked);
        
        self.nextSnapshotPB = QPushButton(self.senecaLabel)
        self.nextSnapshotPB.setStyleSheet("QPushButton { background-color: grey; }\n"
                      "QPushButton:enabled { background-color: rgb(153,208,230); }\n")
                      
        self.nextSnapshotPB.setText(str("next"))
        self.nextSnapshotPB.move(140+130+50, 0);
        self.nextSnapshotPB.resize(40, 25);
        self.nextSnapshotPB.clicked.connect(self.handleNextSnapshotClicked);

        self.intervalLB = QLabel(self.senecaLabel)
        self.intervalLB.setText(str("interval (ms):"))
        self.intervalLB.move(140+130+50+70, 6);

        self.intervalLE = QLineEdit(self.senecaLabel)
        self.intervalLE.setText(str("100"))
        self.intervalLE.move(140+130+50+70+83, 0);
        self.intervalLE.resize(50, 25);
        
        self.playSnapshotPB = QPushButton(self.senecaLabel)
        self.playSnapshotPB.setText(str("play"))
        self.playSnapshotPB.setStyleSheet("QPushButton { background-color: grey; }\n"
                      "QPushButton:enabled { background-color: rgb(153,208,230); }\n")
        self.playSnapshotPB.move(140+130+50+70+83+60, 0);
        self.playSnapshotPB.resize(40, 25);
        self.playSnapshotPB.clicked.connect(self.handlePlaySnapshotClicked);

        self.playSnapshotPB = QPushButton(self.senecaLabel)
        self.playSnapshotPB.setText(str("stop"))
        self.playSnapshotPB.setStyleSheet("QPushButton { background-color: grey; }\n"
                      "QPushButton:enabled { background-color: rgb(153,208,230); }\n")
        self.playSnapshotPB.move(140+130+50+70+83+60+50, 0);
        self.playSnapshotPB.resize(40, 25);
        self.playSnapshotPB.clicked.connect(self.handleStopPlayingSnapshotClicked);
        
        self.SnapshotPlayerThreadAlive = False       
        self.show()
    
    def loadAllSnapshots(self):
        print("Reading snapshots file from \"%s\" file..."%self.snapshotsFileAddr)
        with open(self.snapshotsFileAddr) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            snapshotID = 0
            self.allSnapshots = []
            snapshot = dict()
            for row in csv_reader:
                if(row[0][0] == '#'):
                    #print("comment: ", row)
                    header = row
                    headerSize = len(row)
                else:
                    snapshotIDStr = row[0]                    
                    if (int(snapshotIDStr) != snapshotID):
                            if(snapshotID != 0):
                                self.allSnapshots.append(snapshot) # TODO: question: is it always added to the end of list?
                            snapshotID += 1
                            snapshot = dict()
                            snapshot['coreInfoList'] = []
                            #snapshot['maxTotalEnergy'] = 0
                            #snapshot['minTotalEnergy'] = 9999999
                    coreInfo = dict()
                    for i in range(1,headerSize):
                        if(header[i] != 'core_name'):
                            coreInfo[header[i]] = float(row[i])
                        else:
                            coreInfo[header[i]] = row[i]
                        
                    #snapshot['maxTotalEnergy'] = max(coreInfo['energy_total'], snapshot['maxTotalEnergy'])
                    #snapshot['minTotalEnergy'] = min(coreInfo['energy_total'], snapshot['minTotalEnergy'])
                    snapshot['coreInfoList'].append(coreInfo)            
            self.allSnapshots.append(snapshot) # TODO: question: is it always added to the end of list?
            
        print("Loaded %d snapshots successfully.\n"%len(self.allSnapshots))
    
    def viewSnapshot(self, snapshotID):
        # print("snapshot id %d:"%snapshotID)
        snapshot = self.allSnapshots[snapshotID-1]
        coreInfoList = snapshot['coreInfoList']
        # print("core_x, core_y, core_name, energy_total, energy_total(ratio), processor_utilization(ratio), peak_in_queue_occupancy(ratio), peak_out_queue_occupancy(ratio), total_packet_loss")
        for coreInfo in coreInfoList:
            core_x = int(coreInfo['core_x'])
            core_y = int(coreInfo['core_y'])
            totalEnergy = coreInfo['energy_total']
            totalEnergyPercent = int(100*coreInfo['energy_total(ratio)'])
            cpuBusyPercent = int(100*coreInfo['processor_utilization(ratio)'])
            inputQueueFill = int(100*coreInfo['peak_in_queue_occupancy(ratio)'])
            outputQueueFill = int(100*coreInfo['peak_out_queue_occupancy(ratio)'])
            pktLossNr = int(coreInfo['packet_loss'])
            # print("%d %d %s %d %d %d %d %d %d"%(core_x, core_y, coreInfo['core_name'], totalEnergy, totalEnergyPercent, cpuBusyPercent, inputQueueFill, outputQueueFill, pktLossNr))
            self.NCCinfoUpdate(core_x, core_y, [], totalEnergyPercent, cpuBusyPercent, inputQueueFill, outputQueueFill, pktLossNr) # empty list of names means no change in namesList
        self.Renew()
    
    def handlePrevSnapshotClicked(self):
        snapshotStr = self.snapshotLE.text()
        snapshotID = int(snapshotStr)
        if(snapshotID != 1):
            snapshotID -= 1
        
        self.snapshotLE.setText(str(snapshotID))
        self.viewSnapshot(snapshotID)
        
    def handleNextSnapshotClicked(self):
        snapshotStr = self.snapshotLE.text()
        snapshotID = int(snapshotStr)
        if(snapshotID != self.NrSnapshots):
            snapshotID += 1
        
        self.snapshotLE.setText(str(snapshotID))
        self.viewSnapshot(snapshotID)

    def snapshotPlayer(self):
        interval = int(self.intervalLE.text())
        if(int(self.snapshotLE.text()) == self.NrSnapshots):
            self.snapshotLE.setText(str("1"))
            
        while (int(self.snapshotLE.text()) < self.NrSnapshots):
            if(self.killSnapshotPlayerThread == True):
                break
            sleep(float(interval/1000))
            self.handleNextSnapshotClicked()
        
        self.SnapshotPlayerThreadAlive = False
    
    def handleStopPlayingSnapshotClicked(self):
        if(self.SnapshotPlayerThreadAlive == True):
            self.killSnapshotPlayerThread = True
            self.snapshotPlayerThread.join()
            self.SnapshotPlayerThreadAlive = False
                
    def handlePlaySnapshotClicked(self):
        if(self.SnapshotPlayerThreadAlive == False):
            self.killSnapshotPlayerThread = False
            self.SnapshotPlayerThreadAlive = True
            self.snapshotPlayerThread = threading.Thread(target=self.snapshotPlayer)
            self.snapshotPlayerThread.start()
        
    def setNameNCC(self, x, y, nameList):
        self.NCCgrid[x][y].NCCname(nameList)

    def Renew(self):
        self.update()
    
    def NCCinfoUpdate(self, x, y, layerList, layerColor, cpuBusyPercent, inputQFill, outputQFill, pktLossNr):
        self.NCCgrid[x][y].setLayerColor(layerColor)
        self.NCCgrid[x][y].drawNCC()
        nameList = layerList
        self.NCCgrid[x][y].NCCname(nameList)
        self.NCCgrid[x][y].NCCcpuBusyPercent(cpuBusyPercent)
        self.NCCgrid[x][y].NCCinputQFill(inputQFill)
        self.NCCgrid[x][y].NCCoutputQFill(outputQFill)
        self.NCCgrid[x][y].NCCPktLossNr(pktLossNr)
    
    def BusInfoUpdate(self, busList, busLayer, busColor):
        for NCCXY in busList:
            self.NCCgrid[NCCXY[0]][NCCXY[1]].setBusColor(busColor)
            self.NCCgrid[NCCXY[0]][NCCXY[1]].drawVerticalNet(busLayer)
 
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = SENSIMgui(3, 6, 4)
    sys.exit(app.exec_())
