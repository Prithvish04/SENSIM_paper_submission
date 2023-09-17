from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygon, QFont
from PyQt5.QtCore import Qt, QPoint, QRect


class senecaTile:

    def __init__(self, canvas, nrSegBusses, buswidth, NccWidth, orginX, orginY, offset):
        self.layerColorObj  = QtGui.QColor('black')
        self.busColor    = 'gray'
        self.nrSegBusses = nrSegBusses
        self.canvas      = canvas
        self.horizontalNetOrginX = []
        self.horizontalNetOrginY = []
        self.verticalNetOrginX   = []
        self.verticalNetOrginY   = []
        self.circleOrginX        = []
        self.circleOrginY        = []
        rasterWidth              = (NccWidth+((nrSegBusses+1)*buswidth))
        self.netRectLength       = rasterWidth
        self.netRectWidth        = 0.2 * buswidth
        self.circleDiameter = buswidth
        for BusLevel in range(nrSegBusses):
            self.horizontalNetOrginX.insert(BusLevel, (orginX * rasterWidth) + (BusLevel + 1) * buswidth)
            self.horizontalNetOrginY.insert(BusLevel, (orginY * rasterWidth) + (nrSegBusses - BusLevel) * buswidth  +  offset)
            self.verticalNetOrginX.insert(BusLevel, self.horizontalNetOrginX[BusLevel] + rasterWidth)
            self.verticalNetOrginY.insert(BusLevel, self.horizontalNetOrginY[BusLevel])
            self.circleOrginX.insert(BusLevel, self.horizontalNetOrginX[BusLevel] + rasterWidth + (self.netRectWidth / 2) - (self.circleDiameter / 2))
            self.circleOrginY.insert(BusLevel, self.horizontalNetOrginY[BusLevel] + (self.netRectWidth / 2) - (self.circleDiameter / 2))
            self.drawNet(BusLevel)

        self.nccRectWidth   = rasterWidth - ((nrSegBusses +1) * buswidth)
        self.nccRectHight   = rasterWidth - ((nrSegBusses +1) * buswidth)
        self.nccOrginX = (orginX * rasterWidth) + ((nrSegBusses +1) * buswidth)
        self.nccOrginY = (orginY * rasterWidth) + ((nrSegBusses +1) * buswidth) + offset
        self.drawNCC()

    def drawNet(self, busLevel):
        self.drawHorizontalNet(busLevel)
        self.drawVerticalNet(busLevel)
        self.drawNetConnection(busLevel)

    def drawHorizontalNet(self, busLevel):
        if busLevel < self.nrSegBusses:
            painter = QtGui.QPainter(self.canvas.pixmap())
            pen     = QtGui.QPen()
            pen.setWidth(0.1)
            painter.setPen(pen)
            brush   = QBrush(QtGui.QColor(self.busColor))
            painter.setBrush(brush)
    
            painter.drawRect(self.horizontalNetOrginX[busLevel], self.horizontalNetOrginY[busLevel], self.netRectLength, self.netRectWidth)
            painter.end()

    def drawVerticalNet(self, busLevel):
        if busLevel < self.nrSegBusses:
            painter = QtGui.QPainter(self.canvas.pixmap())
            pen     = QtGui.QPen()
            pen.setWidth(0.1)
            painter.setPen(pen)
            brush   = QBrush(QtGui.QColor(self.busColor))
            painter.setBrush(brush)
    
            painter.drawRect(self.verticalNetOrginX[busLevel], self.verticalNetOrginY[busLevel], self.netRectWidth, self.netRectLength)
            painter.end()

    def drawNetConnection(self, busLevel):
        if busLevel < self.nrSegBusses:
            painter = QtGui.QPainter(self.canvas.pixmap())
            pen     = QtGui.QPen()
            pen.setWidth(0.1)
            painter.setPen(pen)
            brush   = QBrush(QtGui.QColor(self.busColor))
            painter.setBrush(brush)
    
            painter.drawEllipse(self.circleOrginX[busLevel], self.circleOrginY[busLevel], self.circleDiameter, self.circleDiameter)
            painter.end()

    def drawNCC(self):
        painter = QtGui.QPainter(self.canvas.pixmap())
        pen     = QtGui.QPen(self.layerColorObj)
        pen.setWidth(0.1)
        painter.setPen(pen)
        brush   = QBrush(self.layerColorObj)
        painter.setBrush(brush)

        painter.drawRect(self.nccOrginX, self.nccOrginY, self.nccRectWidth, self.nccRectHight)
        painter.end()

    def NCCname(self, nameList):
        if(nameList != []):
            self.nameList = nameList
        painter = QtGui.QPainter(self.canvas.pixmap())
        textBox = QRect(self.nccOrginX, self.nccOrginY, self.nccRectWidth, 25)
        painter.setFont(QFont('Times', 11))
        painter.setPen(self.layerColorObj)
        painter.setBrush(self.layerColorObj)
        painter.drawRect(textBox)
        painter.setPen(QtGui.QColor('green'))
        for idx,name in enumerate(self.nameList):
            painter.drawText(textBox.translated(20, 2+(11+5)*idx), 0, str(name))
        painter.end()
        
    def NCCcpuBusyPercent(self, cpuBusyPercent):
        painter = QtGui.QPainter(self.canvas.pixmap())
        textBox = QRect(self.nccOrginX, self.nccOrginY+self.nccRectWidth-25, self.nccRectWidth, 25)
        painter.setFont(QFont('Times', 9))
        painter.setPen(self.layerColorObj)
        painter.setBrush(self.layerColorObj)
        painter.drawRect(textBox)
        painter.setPen(QtGui.QColor('green'))
        painter.drawText(textBox.translated(22, 2), 0, str(round(cpuBusyPercent)) + "%")
        painter.end()        
  
    def NCCinputQFill(self, QFill):
        painter = QtGui.QPainter(self.canvas.pixmap())
        pen     = QtGui.QPen(self.layerColorObj)
        pen.setWidth(0.1)
        painter.setPen(pen)
        
        brush   = QBrush(QtGui.QColor('white'))
        painter.setBrush(brush)
        painter.drawRect(self.nccOrginX, self.nccOrginY+self.nccRectHight, 13, -(self.nccRectHight))

        
        brush   = QBrush(QtGui.QColor('blue').lighter(150))
        painter.setBrush(brush)
        painter.drawRect(self.nccOrginX, self.nccOrginY+self.nccRectHight, 13, -(self.nccRectHight)*QFill/100)
        painter.end()

    def NCCoutputQFill(self, QFill):
        painter = QtGui.QPainter(self.canvas.pixmap())
        pen     = QtGui.QPen(self.layerColorObj)
        pen.setWidth(0.1)
        painter.setPen(pen)
        
        brush   = QBrush(QtGui.QColor('white'))
        painter.setBrush(brush)
        painter.drawRect(self.nccOrginX+self.nccRectWidth-13, self.nccOrginY+self.nccRectHight, 13, -(self.nccRectHight))
        
        brush   = QBrush(QtGui.QColor('blue').lighter(150))
        painter.setBrush(brush)
        painter.drawRect(self.nccOrginX+self.nccRectWidth-13, self.nccOrginY+self.nccRectHight, 13, -(self.nccRectHight)*QFill/100)
        painter.end()
        
    def NCCPktLossNr(self, pktLossNr):
        painter = QtGui.QPainter(self.canvas.pixmap())
        painter.setFont(QFont('Times', 9))
        painter.setPen(QtGui.QColor('gray').lighter(150))
        painter.setBrush(QtGui.QColor('gray').lighter(150))
        painter.drawRect(self.nccOrginX-12, self.nccOrginY+self.nccRectHight, 12, -self.nccRectHight)
        painter.rotate(90)
        painter.setPen(QtGui.QColor('green').darker(150))
        painter.drawText(self.nccOrginY+3, 12-1*self.nccOrginX, "loss: " + str(pktLossNr))
        painter.end()        
        
        
    def setLayerColor(self, layerColor):
        factor = 100 + (2300 * layerColor/100)
        baseColor = QtGui.QColor(255, 0, 0).darker(1200)
        self.layerColorObj = baseColor.lighter(factor)

    def setBusColor(self, BusColor):
        self.busColor = BusColor