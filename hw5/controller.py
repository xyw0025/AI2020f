# -*- coding: utf-8 -*-

from lib.vaencoder import Operation as op
from View.hwui import Ui_MainWindow
from View.PaintBoard import PaintBoard
import lib.imageConvertor as ic
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QLabel, QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import numpy as np
            
class Thread(QThread):
    #signal format & argument type
    _signal = pyqtSignal(float, float, dict)
    _signal2 = pyqtSignal(op)
    
    def __init__(self, encoder, value):
        super(Thread, self).__init__()
        self.epoch = value
        self.encoder = encoder
        
    def __del__(self):
        self.wait()
    
    def run(self):
        def output(epoch=-1, batch=-1, logs=''):
            self._signal.emit(epoch, batch, logs)
            
        [x_train, y_train, x_test, y_test] = self.encoder.loadData()
        train_data = [x_train, y_train]
        test_data = [x_test, y_test]
        self.encoder.train(train_data, test_data, output, self.epoch)
 
        self._signal2.emit(self.encoder)
        
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
        self.ui.actionload.triggered.connect(lambda:self.onTriggered(0))
        self.ui.actionsave.triggered.connect(lambda:self.onTriggered(1))
        self.ui.radioButton.toggled.connect(lambda:self.setLossType(0))
        self.ui.radioButton_2.toggled.connect(lambda:self.setLossType(1))
        self.ui.radioButton.setVisible(False)
        self.ui.radioButton_2.setVisible(False)

        self.show()
        
        self.enableFunction(False)
        
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
        self.ae.construct()
        self.onTriggered(0)
        
    def setLossType(self, option):
        if option == 0:
            self.ae.setArguments(loss_type='bce')
        elif option == 1:
            self.ae.setArguments(loss_type='mse')
        
    def onClick(self, option):           
        def setProgress(epoch_progress = -1, batch_progress = -1, logs=''):
            
            if epoch_progress != -1:
                maxlen = self.ui.progressBar.maximum()
                p = epoch_progress*maxlen
                self.ui.progressBar.setValue(p)
                self.ui.valLossValue.setText(str(np.round(logs['loss'], decimals=2)))
                
            if batch_progress != -1:
                maxlen = self.ui.subProgressBar.maximum()
                p = batch_progress*maxlen
                self.ui.subProgressBar.setValue(p)
                self.ui.lossValue.setText(str(np.round(logs['loss'], decimals=2)))
                
        def onFinished(ae):
            self.ae = ae
            [x_train, y_train, x_test, y_test] = self.ae.loadData()
            fig_en, fig_de = self.ae.plot(x_test, y_test)
            qimg = ic.plot2qimg(fig_de)
            self.setTouchBackground(qimg)
            self.showDialog('Finished!', QMessageBox.Information)
            self.enableFunction(True)
            self.ui.progressBar.setValue(0)
            self.ui.subProgressBar.setValue(0)
            
        if option == 0:
            self.enableFunction(False)
            epoch_size = self.ui.spinBox.value()
            #self.ae.construct()
            self.thread = Thread(self.ae, epoch_size)
            self.thread._signal.connect(setProgress)
            self.thread._signal2.connect(onFinished)
            self.thread.start()

        elif option == 1:
            self.pb.Clear()
            self.ui.lineEdit.setText('')
            self.ui.lineEdit_2.setText('')
            self.ui.DecoderImg.setPixmap(QPixmap(''))
    
    def clearAll(self):
        self.pb.Clear()
        self.ui.lineEdit.setText('')
        self.ui.lineEdit_2.setText('')
        self.ui.DecoderImg.setPixmap(QPixmap(''))
        self.ui.TouchPanel.setPixmap(QPixmap(''))
            
    #action bar option trigger
    def onTriggered(self, option):   
        if option == 0:
            directory = QFileDialog.getOpenFileName(self,'load model','models','model (*.index)')[0]
            try:
                if directory != '':
                    #self.ae.construct()
                    self.ae.load(directory)
                    [x_train, y_train, x_test, y_test] = self.ae.loadData()
                    fig_en, fig_de = self.ae.plot(x_test, y_test)
                    qimg = ic.plot2qimg(fig_de)
                    self.setTouchBackground(qimg)
                    self.showDialog('Load finished', QMessageBox.Information)
                    self.enableFunction(True)
            except:
                self.showDialog('No model found. Please trainig first', QMessageBox.Information)
                self.clearAll()
                self.enableFunction(False)
                
        elif option == 1:
            directory = QFileDialog.getSaveFileName(self,'save model','models' ,'model (*.index)')[0]
            try:
                if directory != '':
                    self.ae.save(directory)
                    self.showDialog('Save finished', QMessageBox.Information)
            except:
                self.showDialog('Save failed', QMessageBox.Information)
    
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
        img_decoded = np.reshape(img_decoded,(28,28))
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
        img_decoded = np.reshape(img_decoded,(28,28))
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
            x, y = self.ae.encode(arr_img)
            x = np.round(x, decimals=2)
            y = np.round(y, decimals=2)
            self.ui.lineEdit.setText(str(x))
            self.ui.lineEdit_2.setText(str(y))
            img_decoded = self.ae.predict(arr_img)
            img_decoded = np.reshape(img_decoded,(28,28))
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
    
        

