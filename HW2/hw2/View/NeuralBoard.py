

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QVBoxLayout

from PyQt5.QtCore import Qt, QSize, QLine
from PyQt5.QtGui import QPainter

class NeuralBoard(QWidget):


    def __init__(self, size, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)
        
        self.__InitData() #先初始化數據，再初始化界面
        self.__InitView(size)

    def __InitData(self):

        self.__size = QSize(360,560)


    def __InitView(self, size):
        #設置界面的尺寸爲__size
        self.setFixedSize(self.__size)
        self.board = QHBoxLayout(self)
        
        for layer_size in size:
            layer = self.addLayer()
            for neural_size in range(layer_size):
                self.addNeural(layer)


    def addNeural(self, layer):
        neural = QPushButton()
        #neural.setGeometry(-2, -2, 1, 1)
        neural.setMaximumSize(QSize(24,24))
        #neural.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        neural.setStyleSheet('background-color: rgb(0,0,255);border-radius:12; border:2px solid black')
        layer.addWidget(neural)
        
    def addLayer(self):
        layer = QVBoxLayout(self)
        self.board.addLayout(layer)
        return layer
    
    def setActivate(self, layer, index, value):
        #self.clear()
        neural = self.getNeural((layer, index))
        color = str(int(value*255))
        neural.setStyleSheet('background-color: rgb('+color+','+color+',255); border-radius:12; border:2px solid black')
        
 
    def clear(self):
        layers = self.board.findChildren(QVBoxLayout)
        for layer in layers:
            for index in range(layer.count()):
                neural = layer.itemAt(index).widget()
                neural.setStyleSheet('background-color: rgb(0,0,255);border-radius:12; border:2px solid black')
                
    def paintEvent(self, event):
        painter = QPainter(self)
        layers = self.board.findChildren(QVBoxLayout)
        for i in range(len(layers)-1):
            layer1 = layers[i]
            layer2 = layers[i+1]
            for j in range(layer1.count()):
                n1 = layer1.itemAt(j).widget()
                for k in range(layer2.count()):
                    n2 = layer2.itemAt(k).widget()
                    painter.setPen(Qt.gray)
                    painter.drawLine(QLine(n1.pos().x()+12,n1.pos().y()+12,n2.pos().x()+12,n2.pos().y()+12))
            
    def skeleton(self):
        layers = self.board.findChildren(QVBoxLayout)
        for layer in layers:
            for index in range(layer.count()):
                layer.itemAt(index).widget()
        
    def getNeural(self, n_index):
        [l_index, n_index] = n_index
        layer = self.board.itemAt(l_index)
        neural = layer.itemAt(n_index).widget()
        return neural
                
        



        