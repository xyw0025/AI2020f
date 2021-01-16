import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import qRed, qGreen, qBlue
from View.hw_ui import Ui_MainWindow
import numpy as np
import neuralnet as nl
import load_mnist as lm
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
        self.ui.trainButton.clicked.connect(lambda: self.onClick(1))
        return
        
    def pbMouseRelease(self, mouseEvent):
        self.tmp(mouseEvent)
        self.identify()
        
    def identify(self):
        try:
            w_list = [np.loadtxt('w_list0'), np.loadtxt('w_list1')]
            b_list = [np.loadtxt('b_list0'), np.loadtxt('b_list1')]
        except OSError:
            self.showDialog("Weight file not found! Please train the model first.")
            return

            
        img = self.ui.paintBoard.GetContentAsQImage() 
        img = img.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img = self.QImage2CV(img)
        img = img.astype(np.float32)
        img /= 255.0
        img = img.reshape(-1,28*28)
#            print(img.shape)
#        print(img.reshape(-1,28*28).shape)

         
        index_max = -1
        val_dict = nl.calculate(img, w_list, b_list)
        values = val_dict['y_2'][0]
        self.setOutputValue(values)
            
        for i in range(10):
            if index_max == -1:
               index_max = i
            if values[index_max] < values[i]:
               index_max = i
                    
            self.setOutputActivate(index_max)
            
    def onClick(self, select):
        #reset state
        if select == 0:
            self.clearValues()
            self.ui.paintBoard.Clear()
        
        #training models
        elif select == 1:
            dataset = lm.load_mnist()
            x_train = dataset['x_train'].round(0)
            y_train = dataset['y_train'].round(0)
            x_test = dataset['x_test'].round(0)
            y_test = dataset['y_test'].round(0)           

            w_list, b_list = nl.make_params([784, 100, 10])
            
            epoches = int(self.ui.epochEdit.text())
            for epoch in range(epoches):
                ra = np.random.randint(60000,size=60000)
                
                for i in range(60):
#                    if i % 10 == 0:
#                        print(r"batch from {} to batch {}".format(i*1000, (i+1)*1000))
                    x_batch = x_train[ra[i*1000:(i+1)*1000], :]
                    y_batch = y_train[ra[i*1000:(i+1)*1000], :]
                    w_list, b_list = nl.update(x_batch, w_list, b_list, y_batch, eta=2.0)
                    
                (train_acc, train_loss) = self.showAccuracy(x_train, y_train, w_list, b_list, epoch)
                self.ui.progress_label.setText("Train Acc: %f, Loss: %f"%(train_acc, train_loss))
                QApplication.processEvents()

                self.ui.progressBar.setValue(((epoch+1)/ epoches)*100)
                self.ui.epoch_label.setText("Epoch: %d/%d" %(epoch + 1, epoches))
                (test_acc, test_loss) = self.showAccuracy(x_test, y_test, w_list, b_list, epoch)
                self.ui.test_label.setText("| Test Acc: %f, Loss: %f"%(test_acc, test_loss))

#            self.show
#            self.ui.test_label.setText("Test Acc: %f, Loss: %f"%(--, --))
#            test_label
                
                
            #save models
            np.savetxt('w_list0', w_list[0])
            np.savetxt('w_list1', w_list[1])
            np.savetxt('b_list0', b_list[0])
            np.savetxt('b_list1', b_list[1])
       
    def showAccuracy(self, x_test, y_test,  w_list, b_list, epoch):
        acc_list = []
        loss_list = []
        total_acc_list = []
        total_loss_list = []
        batch_size = 1000
                
        for k in range(10000//1000):
               
            x_batch, y_batch = x_test[k*batch_size:(k+1)*batch_size, :], y_test[k*batch_size:(k+1)*batch_size, :]
              
            acc_val = nl.accuracy(x_batch, w_list, b_list, y_batch)
            loss_val = nl.loss(x_batch, w_list, b_list, y_batch)
                
            acc_list.append(acc_val)
            loss_list.append(loss_val)
        
        acc = np.mean(acc_list)
        loss = np.mean(loss_list)
      
        total_acc_list.append(acc)
        total_loss_list.append(loss)
#        print("epoch:%d, Accuracy: %f, Loss: %f"%(epoch, acc, loss))
        return (acc, loss)
        
    def clearValues(self):
        self.setOutputActivate(-1)
#        self.ui.progressBar.setValue(0)
#        self.ui.progress_label.setText("Train:")
        for i in range(self.ui.outputLayout.rowCount()):
            self.ui.outputLayout.itemAt(i, 1).widget().setText('')
        
    def setOutputValue(self, values):
        for i in range(self.ui.outputLayout.rowCount()):
            value = values[i]
            self.ui.outputLayout.itemAt(i, 1).widget().setText(str(value.round(2)))
                    
    def setOutputActivate(self, index):
        for i in range(self.ui.outputLayout.rowCount()):
            if i == index:
                label = self.ui.outputLayout.itemAt(i, 1).widget()
                label.setStyleSheet('background-color: lightgreen')
            else:
                label = self.ui.outputLayout.itemAt(i, 1).widget()
                label.setStyleSheet('background-color: white')

    #convert img from QImage to opencv array
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
    
    def msgBoxbtn(self, index):
        self.ui.paintBoard.Clear()
    
    def showDialog(self, msg):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText(msg)
        msgBox.setWindowTitle("Message")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.buttonClicked.connect(self.msgBoxbtn)
        msgBox.exec_()
        
app = QApplication(sys.argv)
w = AppWidget()
w.show()
sys.exit(app.exec())
