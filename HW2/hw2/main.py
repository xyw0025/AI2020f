import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtGui import qRed, qGreen, qBlue
from View.hw_ui import Ui_MainWindow
import numpy as np
import neuralnet as nl
import cv2

class AppWidget(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.prepareView()
        self.show()
        
    def prepareView(self):
        
        self.tmp = self.ui.paintBoard.mouseReleaseEvent
        self.ui.paintBoard.mouseReleaseEvent = self.pbMouseRelease
        self.ui.clearButton.clicked.connect(lambda: self.onClick(0))
        
    def pbMouseRelease(self, mouseEvent):
        self.tmp(mouseEvent)
        self.prediction()
        
    def prediction(self):
        try:
            w_list = [np.loadtxt('w_list0'), np.loadtxt('w_list1'), np.loadtxt('w_list2')]
            b_list = [np.loadtxt('b_list0'), np.loadtxt('b_list1'), np.loadtxt('b_list2')]
        except OSError:
            self.showDialog("Weight file not found! Please train the model first.", QMessageBox.Critical)
            self.ui.paintBoard.Clear()
            return
        
            
        img = self.ui.paintBoard.GetContentAsQImage()
        img = img.scaled(28, 28)
        img = self.QImage2CV(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        img /= 255.0
        img = img.reshape(-1, 28 * 28)
        
#        nl.test(img, w_list, b_list)
        
        val_dict = nl.calculate(img, w_list, b_list)
        
        
        value1 = val_dict['y_1'][0]
        value2 = val_dict['y_2'][0]
        value3 = val_dict['y_3'][0]
        
        valueGroup = [value1, value2, value3]
        self.clearActiveState(1)
        self.clearActiveState(2)
        self.clearActiveState(3)
        for i in range(3):
            value = valueGroup[i]
            for j in range(len(value)):
                self.ui.neuralBoard.setActivate(i, j, value[j])
            self.setOutputValue(value, i+1)
            self.setOutputActivate(value, i+1)
        
    def onClick(self, select):
        #reset state
        if select == 0:
            self.clearAll()
            self.ui.paintBoard.Clear()
            
    def clearAll(self):
        self.clearActiveState(1)
        self.clearActiveState(2)
        self.clearActiveState(3)
        
        self.ui.neuralBoard.clear()
        for i in range(self.ui.LayerLayout1.rowCount()):
            self.ui.LayerLayout1.itemAt(i, 1).widget().setText('')
        for i in range(self.ui.LayerLayout2.rowCount()):
            self.ui.LayerLayout2.itemAt(i, 1).widget().setText('')
        for i in range(self.ui.LayerLayout3.rowCount()):
            self.ui.LayerLayout3.itemAt(i, 1).widget().setText('')
        
    def setOutputValue(self, values, layer):
        if layer == 1:
            layerLayout = self.ui.LayerLayout1
        elif layer == 2:
            layerLayout = self.ui.LayerLayout2
        elif layer == 3:
            layerLayout = self.ui.LayerLayout3
            
        for i in range(layerLayout.rowCount()):
            value = values[i]
            value_r = (value*100).round(2)
            layerLayout.itemAt(i, 1).widget().setText(str(value_r)+'%')
                    
    def clearActiveState(self, layer):
        if layer == 1:
            layerLayout = self.ui.LayerLayout1
        elif layer == 2:
            layerLayout = self.ui.LayerLayout2
        elif layer == 3:
            layerLayout = self.ui.LayerLayout3
            
        for i in range(layerLayout.rowCount()):
            label = layerLayout.itemAt(i, 1).widget()
            label.setStyleSheet('background-color: white')
            
    def setOutputActivate(self, values, layer):
        if layer == 1:
            layerLayout = self.ui.LayerLayout1
        elif layer == 2:
            layerLayout = self.ui.LayerLayout2
        elif layer == 3:
            layerLayout = self.ui.LayerLayout3
            
        # max value set as blue yes
        # RGB(x, x, 255)
        # the smaller x is, the bluer color we will get
        max = 0
        max_index = 0
        for i in range(layerLayout.rowCount()):
            value = values[i]
            if value > max:
                max_index = i
                max = value
        color = int((1-max)*255)
        if color <= 150: # prevent it too blue to see the label text underneath
            color += 50
        
        color = str(color)
        print(r"neuron index:{} ,max val:{}, color code: {}".format(max_index, max, color))
        label = layerLayout.itemAt(max_index, 1).widget()
        label.setStyleSheet('background-color: rgb('+color+','+color+',255)')

    def QImage2CV(self, qimg):        
        tmp = qimg        
        
        cv_image = np.zeros((tmp.height(), tmp.width(), 3), dtype=np.uint8)
        
        for row in range(0, tmp.height()):
            for col in range(0,tmp.width()):
                r = qRed(tmp.pixel(col, row))
                g = qGreen(tmp.pixel(col, row))
                b = qBlue(tmp.pixel(col, row))
                cv_image[row,col,0] = r
                cv_image[row,col,1] = g
                cv_image[row,col,2] = b
        
        return cv_image
    
    def showDialog(self, msg, level):
        msgBox = QMessageBox()
        msgBox.setIcon(level)
        msgBox.setText(msg)
        msgBox.setWindowTitle("Message")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec_()
        
app = QApplication(sys.argv)
w = AppWidget()
w.show()
sys.exit(app.exec())
