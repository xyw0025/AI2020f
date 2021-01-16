# -*- coding: utf-8 -*-

from lib.autoencoder import Operation as op
from View.hwui import Ui_MainWindow
from View.PaintBoard import PaintBoard
import lib.imageConvertor as ic
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import numpy as np
            
class Thread(QThread):
    #signal format & argument type
    _signal = pyqtSignal(float)
    _signal2 = pyqtSignal()
    
    def __init__(self, value):
        super(Thread, self).__init__()
        self.epoch = value
        
    def __del__(self):
        self.wait()
    
    def run(self):
        def output(progress):
            self._signal.emit(progress)
            
        ae = op()
        [x_train, y_train, x_test, y_test] = ae.loadData()
        train_data = [x_train, x_train]
        test_data = [x_test, x_test]
        ae.construct()
        ae.train(train_data, test_data, output, self.epoch)
        ae.save()
 

        self._signal2.emit()
        
class AppWidget(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.pb = PaintBoard(self.ui.EncoderImg)
        self.pb.mouseReleaseEvent = self._mouseReleaseEvent(1)
        
        self.ui.TrainBtn.clicked.connect(lambda:self.onClick(0))
        self.ui.ClearBtn.clicked.connect(lambda:self.onClick(1))
        self.ui.TouchPanel.mouseMoveEvent = self._mouseHoverEvent
        self.ui.TouchPanel.mouseReleaseEvent = self._mouseReleaseEvent(0)
        self.ui.TouchPanel.mousePressEvent = self._mousePressEvent
        self.show()
        
        self.pointer_lb = QLabel(self)
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.pointer_lb.setFont(font)
        self.pointer_lb.setText('ttt')
        self.pointer_lb.setGeometry(0,-50,150,50)
        self.pointer_lb.setAlignment(Qt.AlignCenter)
        self.pointer_lb.setStyleSheet('background-color:red;color:white;')
        #self.ui.TouchPanel.addWidgets(self.pointer_lb)
        self.pointer_lb.show()
        self.pointer_lb.setVisible(False)
        
        self.ae = op()
        if self.ae.load() == False:
            self.showDialog('No model found. Please trainig first', QMessageBox.Information)
            self.enableFunction(False)
        else:
            img = QImage('autoencoder_2dim/digits_over_latent.png')
            self.setTouchBackground(img)

        
    def onClick(self, option):           
        def setProgress(progress):
            p = progress*100
            if p > 5:
                self.ui.progressBar.setValue(p)
                
        def onFinished():
            self.ae.load()
            
            [x_train, y_train, x_test, y_test] = self.ae.loadData()
            self.ae.load()
            self.ae.plot(x_test, y_test)
            img = QImage('autoencoder_2dim/digits_over_latent.png')
            self.setTouchBackground(img)
            self.showDialog('Finished!', QMessageBox.Information)
            self.enableFunction(True)
            self.ui.progressBar.setValue(0)
            
        if option == 0:
            self.ui.progressBar.setValue(5)
            self.enableFunction(False)
            epoch_size = self.ui.EpochLabel.text()
            
            self.thread = Thread(int(epoch_size))
            self.thread._signal.connect(setProgress)
            self.thread._signal2.connect(onFinished)
            self.thread.start()

        elif option == 1:
            self.pb.Clear()
            self.ui.lineEdit.setText('')
            self.ui.lineEdit_2.setText('')
            self.ui.DecoderImg.setPixmap(QPixmap(''))
    
    def enableFunction(self, state):
        self.ui.TouchPanel.setEnabled(state)
        self.pb.setEnabled(state)
        self.ui.DecoderImg.setEnabled(state)
    
    def _mousePressEvent(self, mouseEvent):
        if self.ae.prepared() == False:
            return
        
        self.__currentPos =  mouseEvent.pos()
        mWidth = self.ui.TouchPanel.minimumWidth()
        mHeight = self.ui.TouchPanel.minimumHeight()
        width = self.__currentPos.x()
        height = self.__currentPos.y()
        
        x = width*8/mWidth-4
        y = 4-height*8/mHeight
        x = np.round_(x, decimals=2)
        y = np.round_(y, decimals=2)
        
        if x<-4 or x>4 or y<-4 or y>4:
            return
        
        img_decoded = self.ae.decode(x, y)
        qimg = ic.numpy2qimg(img_decoded)
        self.setOutputBackground(qimg)
        
        rect = self.ui.TouchPanel.geometry()
        self.pointer_lb.setText(str(x)+' , '+str(y))
        self.pointer_lb.setGeometry(self.__currentPos.x()+rect.x(),
                                    self.__currentPos.y()+rect.y()-50,
                                    150, 50)
        self.pointer_lb.setVisible(True)
    
    def _mouseHoverEvent(self, mouseEvent):
        if self.ae.prepared() == False:
            return
        
        self.__currentPos =  mouseEvent.pos()
        mWidth = self.ui.TouchPanel.minimumWidth()
        mHeight = self.ui.TouchPanel.minimumHeight()
        width = self.__currentPos.x()
        height = self.__currentPos.y()
        
        x = width*8/mWidth-4
        y = 4-height*8/mHeight
        x = np.round_(x, decimals=2)
        y = np.round_(y, decimals=2)
        
        if x<-4 or x>4 or y<-4 or y>4:
            return
        
        img_decoded = self.ae.decode(x, y)
        qimg = ic.numpy2qimg(img_decoded)
        self.setOutputBackground(qimg)
        
        rect = self.ui.TouchPanel.geometry()
        self.pointer_lb.setText(str(x)+' , '+str(y))
        self.pointer_lb.setGeometry(self.__currentPos.x()+rect.x(),
                                    self.__currentPos.y()+rect.y()-50,
                                    150, 50)
    
    def _mouseReleaseEvent(self, option):
        def s0(mouseEvent):
            self.pointer_lb.setVisible(False)
            self.ui.OutputImg.setPixmap(QPixmap(''))
            
        def s1(mouseEvent):
            if self.ae.prepared() == False:
                return
            
            img = self.pb.GetContentAsQImage()
            arr_img = ic.qimg2numpy(img)
            [x, y] = self.ae.encode(arr_img)[0]
            x = np.round(x, decimals=2)
            y = np.round(y, decimals=2)
            self.ui.lineEdit.setText(str(x))
            self.ui.lineEdit_2.setText(str(y))
            img_decoded = self.ae.predict(arr_img)
            qimg = ic.numpy2qimg(img_decoded)
            self.setDecoderBackground(qimg)
        
        if option==0:
            return s0
        elif option==1:
            return s1
        
        
    def setDecoderBackground(self, qimg):
        mWidth = self.ui.DecoderImg.minimumWidth()
        mHeight = self.ui.DecoderImg.minimumHeight()
        qpix = QPixmap.fromImage(qimg)
        qpix = qpix.scaled(mWidth, mHeight, transformMode=Qt.SmoothTransformation)
        self.ui.DecoderImg.setPixmap(qpix)
        return
      
    def setTouchBackground(self, qimg):
        mWidth = self.ui.TouchPanel.minimumWidth()
        mHeight = self.ui.TouchPanel.minimumHeight()
        qpix = QPixmap.fromImage(qimg)
        qpix = qpix.scaled(mWidth, mHeight, transformMode=Qt.SmoothTransformation)
        self.ui.TouchPanel.setPixmap(qpix)
        return
    
    def setOutputBackground(self, qimg):
        mWidth = self.ui.OutputImg.minimumWidth()
        mHeight = self.ui.OutputImg.minimumHeight()
        qpix = QPixmap.fromImage(qimg)
        qpix = qpix.scaled(mWidth, mHeight, transformMode=Qt.SmoothTransformation)
        self.ui.OutputImg.setPixmap(qpix)
        return
    
    def showDialog(self, msg, level):
        msgBox = QMessageBox()
        msgBox.setIcon(level)
        msgBox.setText(msg)
        msgBox.setWindowTitle("Message")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec_()
    
        

