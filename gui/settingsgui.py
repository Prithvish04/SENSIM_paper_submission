from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QLineEdit, QLabel, QWidget, QRadioButton, QScrollArea
from PyQt5.QtGui import QFont, QWindow
from time import sleep

class SettingsGUI(QtWidgets.QMainWindow):
    windowWidth = 1200
    windowHeight = 1200
    xSmallMargin = 5
    xLargeMargin = 15

    initXMargin = 10
    beginningLine = True

    def __init__(self, SystemSettings):
        super().__init__()
        self.SystemSettings = SystemSettings
        self.curser = {"x": self.initXMargin, "y": 0}
        self.resize(self.windowWidth,self.windowHeight)

        self.infoLB1 = self.createLabel("Select a System to show settings: ")
        self.moveCurserNextLine()
        
        self.ShownSetting = ""
        
        self.SystemsRBs = dict()
        for systemName in self.SystemSettings.keys() :
            RB = self.createRB(systemName)
            RB.toggled.connect(self.handleRBToggled)
            self.SystemsRBs[systemName] = RB

        self.moveCurserNextLine()
        self.show()
        self.baseCurserPoistion = self.curser.copy()

        self.itemsToShow = dict()
      
        for systemName in self.SystemSettings.keys() :
            self.moveCurserToBasePoistion()
            if(self.SystemSettings[systemName]['gui']['type'] == 'label-value'):
                for settingName in self.SystemSettings[systemName]['values'].keys():
                    [labelObj, dataObj] = self.createObjs(settingName, self.SystemSettings[systemName]['values'][settingName])
                    self.SystemSettings[systemName]['values'][settingName]['label_obj'] = labelObj
                    self.SystemSettings[systemName]['values'][settingName]['data_obj'] = dataObj                
            
            if(self.SystemSettings[systemName]['gui']['type'] == 'header-value'):
                headerObjList = []
                for headerItem in self.SystemSettings[systemName]['gui']['header']:
                    itemSize = int(self.SystemSettings[systemName]['gui']['size'])
                    headerObj = self.createLabel(headerItem, size= itemSize,withMargin=True)
                    headerObjList.append(headerObj)
                self.SystemSettings[systemName]['gui']['header_obj_list'] = headerObjList

                self.curser["x"] = self.initXMargin
                self.curser["y"] = 10
                self.beginningLine = True

                for settingName in self.SystemSettings[systemName]['values'].keys():
                    settingDetails = self.SystemSettings[systemName]['values'][settingName]                    
                    dataObjList = []                    
                    itemSize = int(settingDetails['gui']['size'])
                    itemcolour = int(settingDetails['gui']['color'])
                    for value in settingDetails['value']:                        
                        dataObjList.append(self.createLineEdit(value, size=itemSize, withMargin=True, color=itemcolour))
                    self.SystemSettings[systemName]['values'][settingName]['data_obj_list'] = dataObjList
                    self.moveCurserNextLine()


        aSystemName = list(self.SystemSettings.keys())[0]
        self.SystemsRBs[aSystemName].setChecked(True)
        self.ShownSetting = aSystemName

    def createObjs(self, settingName, settingDetails):
        settingValue = settingDetails['value']
        settingGUIType = settingDetails['gui']['type'] # TODO CHECK IT !!!!!! IT MUST SUPPORT OTHER TYPES
        settingGUISize = int(settingDetails['gui']['size']) # TODO CHECK IT !!!!!! IT MUST SUPPORT OTHER TYPES
        
        # print("creating objs for %s with value %s"%(settingName, settingValue))
        dataObj = self.createLineEdit(settingValue, size=settingGUISize, placeNow=False)
        labelObj = self.createLabel(settingName, withMargin=True, placeTogetherSize=dataObj.width()+self.xSmallMargin)
        self.placeObject(dataObj)

        if(settingDetails['gui']['move_curser_next_line'] == "Yes"):
            self.moveCurserNextLine()

        return [labelObj, dataObj]

    def moveCurserNextLine(self):
        self.beginningLine = True
        self.curser["y"] += 35
        self.curser["x"] = self.initXMargin
    
    def moveCurserToBasePoistion(self):
        # print("move from %d to %d" % (self.curser['x'], self.baseCurserPoistion['x']))
        self.curser["x"] = self.baseCurserPoistion['x']
        self.curser["y"] = self.baseCurserPoistion['y']
        self.beginningLine = True

    def placeObject(self, obj, withMargin = False, placeTogetherSize=0):
        width = obj.width()
        
        if(self.beginningLine):
            margin = 0
        elif(withMargin):
            margin = self.xLargeMargin
        else:
            margin = self.xSmallMargin

        if((width + self.curser["x"]+margin+placeTogetherSize) > self.windowWidth):
            self.moveCurserNextLine()
            margin = 0
            
        objX = self.curser["x"] + margin
        objY = self.curser["y"]

        if(str(type(obj)) == "<class 'PyQt5.QtWidgets.QLabel'>"):
            objY += 3
        
        obj.move(objX, objY)

        self.curser["x"] += width + margin
        self.beginningLine = False
           
    def createLabel(self, txt, size=0, withMargin=False, placeNow = True, placeTogetherSize = 0):
        label = QLabel(self)
        label.setFont(QFont('Arial', 10))
        label.setText(txt)
        if(size == 0):
            label.adjustSize()
        else:
            label.setFixedWidth(size*14)
        label.setAlignment(Qt.AlignCenter)
        if placeNow:
            self.placeObject(label, withMargin, placeTogetherSize)
        return label

    def createLineEdit(self, txt, withMargin=False, size=8, placeNow = True, color = 0):
        le = QLineEdit(self)
        le.setFont(QFont('Arial', 10))
        le.setText(txt)
        le.setFixedWidth(size*14)
        le.setFixedHeight(25)
        if placeNow:
            self.placeObject(le, withMargin)

        if(color == 1):
            le.setStyleSheet("color: black;  background-color: rgb(153,208,230)")
        return le

    def createPB(self, txt, withMargin=False):
        PB =  QPushButton(self)
        PB.setFont(QFont('Arial', 10))
        PB.setText(txt)
        PB.setStyleSheet("QPushButton { background-color: grey; }\n"
                      "QPushButton:enabled { background-color: rgb(153,208,230); }\n")
        PB.adjustSize()
        self.placeObject(PB, withMargin)
        return PB

    def createRB(self, txt, withMargin=False):
        RB =  QRadioButton(self)
        RB.setFont(QFont('Arial', 10))
        RB.setText(txt)
        # PB.setCheckable(True)
        # PB.setStyleSheet("QPushButton { background-color: grey; }\n"
                    #   "QPushButton:enabled { background-color: rgb(153,208,230); }\n")
        RB.adjustSize()
        self.placeObject(RB, withMargin)
        return RB
    
    def handleRBToggled(self, checked):
        if(not checked):
            return

        for systemName in self.SystemsRBs:
            if (self.SystemsRBs[systemName].isChecked()):
                print(systemName)
                self.ShowSettings(systemName)
    
    def ShowSettings(self, SystemName):        
        if (not self.ShownSetting == ""):
            if(self.SystemSettings[self.ShownSetting]['gui']['type'] == 'label-value'):
                for settingName in self.SystemSettings[self.ShownSetting]['values'].keys():
                    settingDetails = self.SystemSettings[self.ShownSetting]['values'][settingName]
                    settingDetails['label_obj'].hide()
                    settingDetails['data_obj'].hide()

            if(self.SystemSettings[self.ShownSetting]['gui']['type'] == 'header-value'):
                for headerObj in self.SystemSettings[self.ShownSetting]['gui']['header_obj_list']:
                    headerObj.hide()

                for settingName in self.SystemSettings[self.ShownSetting]['values'].keys():
                    settingDetails = self.SystemSettings[self.ShownSetting]['values'][settingName]
                    for data_obj in settingDetails['data_obj_list']:
                        data_obj.hide()
                
                self.newWindow.hide()
                self.scrollArea.hide()
                        
        if(self.SystemSettings[SystemName]['gui']['type'] == 'label-value'):
            for settingName in self.SystemSettings[SystemName]['values'].keys():
                settingDetails = self.SystemSettings[SystemName]['values'][settingName]
                settingDetails['label_obj'].show()
                settingDetails['data_obj'].show()

        if(self.SystemSettings[SystemName]['gui']['type'] == 'header-value'):               
            for headerObj in self.SystemSettings[SystemName]['gui']['header_obj_list']:                
                headerObj.show()
            
            self.scrollArea =  QScrollArea(self)
            self.scrollArea.move(0,100)
            self.scrollArea.resize(self.windowWidth,self.windowHeight-100)

            self.newWindow = QWidget(self)            

            for settingName in self.SystemSettings[SystemName]['values'].keys():
                settingDetails = self.SystemSettings[SystemName]['values'][settingName]
                for data_obj in settingDetails['data_obj_list']:
                    data_obj.setParent(self.newWindow)
                    data_obj.show()
                    lastObj = data_obj

            self.newWindow.resize(lastObj.geometry().bottomRight().x()+10,lastObj.geometry().bottomRight().y()+10)
            self.newWindow.show()
            self.scrollArea.setWidget(self.newWindow)
            self.scrollArea.show()
            
        self.ShownSetting = SystemName
        
# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     ui = SettingsGUI()
#     sys.exit(app.exec_())
