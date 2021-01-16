# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hwui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1275, 993)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_9.setFont(font)
        self.label_9.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_9.setObjectName("label_9")
        self.verticalLayout.addWidget(self.label_9)
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.verticalLayout.addWidget(self.line_5)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.FuncLayout = QtWidgets.QHBoxLayout()
        self.FuncLayout.setObjectName("FuncLayout")
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setReadOnly(False)
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(20)
        self.spinBox.setStyleSheet('font: 12pt Arial')
        self.spinBox.setObjectName("spinBox")
        self.FuncLayout.addWidget(self.spinBox)
        self.verticalLayout.addLayout(self.FuncLayout)
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.verticalLayout.addWidget(self.line_4)
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.radioButton.setFont(font)
        self.radioButton.setChecked(True)
        self.radioButton.setObjectName("radioButton")
        self.verticalLayout.addWidget(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setObjectName("radioButton_2")
        self.verticalLayout.addWidget(self.radioButton_2)
        self.TrainBtn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.TrainBtn.setFont(font)
        self.TrainBtn.setObjectName("TrainBtn")
        self.verticalLayout.addWidget(self.TrainBtn)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(False)
        self.progressBar.setObjectName("progressBar")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.progressBar)
        self.epochLog = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.epochLog.setFont(font)
        self.epochLog.setObjectName("epochLog")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.epochLog)
        self.subProgressBar = QtWidgets.QProgressBar(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.subProgressBar.sizePolicy().hasHeightForWidth())
        self.subProgressBar.setSizePolicy(sizePolicy)
        self.subProgressBar.setMaximumSize(QtCore.QSize(16777215, 10))
        self.subProgressBar.setProperty("value", 0)
        self.subProgressBar.setTextVisible(False)
        self.subProgressBar.setObjectName("subProgressBar")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.SpanningRole, self.subProgressBar)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.valLossValue = QtWidgets.QLabel(self.centralwidget)
        self.valLossValue.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.valLossValue.setFont(font)
        self.valLossValue.setText("")
        self.valLossValue.setObjectName("valLossValue")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.valLossValue)
        self.lossValue = QtWidgets.QLabel(self.centralwidget)
        self.lossValue.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.lossValue.setFont(font)
        self.lossValue.setText("")
        self.lossValue.setObjectName("lossValue")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lossValue)
        self.verticalLayout.addLayout(self.formLayout_2)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.horizontalLayout_3.addWidget(self.line_3)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_4.addWidget(self.label_8)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.EncoderImg = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.EncoderImg.sizePolicy().hasHeightForWidth())
        self.EncoderImg.setSizePolicy(sizePolicy)
        self.EncoderImg.setMinimumSize(QtCore.QSize(300, 300))
        self.EncoderImg.setMaximumSize(QtCore.QSize(300, 300))
        self.EncoderImg.setAutoFillBackground(False)
        self.EncoderImg.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"")
        self.EncoderImg.setText("")
        self.EncoderImg.setAlignment(QtCore.Qt.AlignCenter)
        self.EncoderImg.setObjectName("EncoderImg")
        self.horizontalLayout.addWidget(self.EncoderImg)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem4)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setReadOnly(True)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_2)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.lineEdit.setFont(font)
        self.lineEdit.setReadOnly(True)
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.verticalLayout_3.addLayout(self.formLayout)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem5)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem6)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout.addWidget(self.label_5)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem7)
        self.DecoderImg = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.DecoderImg.sizePolicy().hasHeightForWidth())
        self.DecoderImg.setSizePolicy(sizePolicy)
        self.DecoderImg.setMinimumSize(QtCore.QSize(300, 300))
        self.DecoderImg.setMaximumSize(QtCore.QSize(300, 300))
        self.DecoderImg.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"")
        self.DecoderImg.setText("")
        self.DecoderImg.setAlignment(QtCore.Qt.AlignCenter)
        self.DecoderImg.setObjectName("DecoderImg")
        self.horizontalLayout.addWidget(self.DecoderImg)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.ClearBtn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.ClearBtn.setFont(font)
        self.ClearBtn.setObjectName("ClearBtn")
        self.verticalLayout_4.addWidget(self.ClearBtn)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_4.addWidget(self.line)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_4.addWidget(self.label_2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.TouchPanel = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.TouchPanel.sizePolicy().hasHeightForWidth())
        self.TouchPanel.setSizePolicy(sizePolicy)
        self.TouchPanel.setMinimumSize(QtCore.QSize(512, 512))
        self.TouchPanel.setMaximumSize(QtCore.QSize(512, 512))
        self.TouchPanel.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"")
        self.TouchPanel.setText("")
        self.TouchPanel.setAlignment(QtCore.Qt.AlignCenter)
        self.TouchPanel.setObjectName("TouchPanel")
        self.horizontalLayout_2.addWidget(self.TouchPanel)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem8)
        self.OutputImg = QtWidgets.QLabel(self.centralwidget)
        self.OutputImg.setMinimumSize(QtCore.QSize(512, 512))
        self.OutputImg.setMaximumSize(QtCore.QSize(512, 512))
        self.OutputImg.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.OutputImg.setText("")
        self.OutputImg.setObjectName("OutputImg")
        self.horizontalLayout_2.addWidget(self.OutputImg)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3.addLayout(self.verticalLayout_4)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem9)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1275, 21))
        self.menuBar.setObjectName("menuBar")
        self.menuModels = QtWidgets.QMenu(self.menuBar)
        self.menuModels.setObjectName("menuModels")
        MainWindow.setMenuBar(self.menuBar)
        self.actionmse = QtWidgets.QAction(MainWindow)
        self.actionmse.setObjectName("actionmse")
        self.actionbce = QtWidgets.QAction(MainWindow)
        self.actionbce.setObjectName("actionbce")
        self.actionload = QtWidgets.QAction(MainWindow)
        self.actionload.setObjectName("actionload")
        self.actionsave = QtWidgets.QAction(MainWindow)
        self.actionsave.setObjectName("actionsave")
        self.menuModels.addAction(self.actionload)
        self.menuModels.addAction(self.actionsave)
        self.menuBar.addAction(self.menuModels.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Variational Autoencoder(VAE)"))
        self.label_9.setText(_translate("MainWindow", "Model Training"))
        self.label_3.setText(_translate("MainWindow", "Epoch"))
        self.label.setText(_translate("MainWindow", "Loss"))
        self.radioButton.setText(_translate("MainWindow", "mse"))
        self.radioButton_2.setText(_translate("MainWindow", "bce"))
        self.TrainBtn.setText(_translate("MainWindow", "Train"))
        self.epochLog.setText(_translate("MainWindow", "Val_Loss:"))
        self.label_10.setText(_translate("MainWindow", "Loss:"))
        self.label_8.setText(_translate("MainWindow", "Paint Board"))
        self.label_4.setText(_translate("MainWindow", ">>"))
        self.label_6.setText(_translate("MainWindow", "x:"))
        self.label_7.setText(_translate("MainWindow", "y:"))
        self.label_5.setText(_translate("MainWindow", ">>"))
        self.ClearBtn.setText(_translate("MainWindow", "Clear"))
        self.label_2.setText(_translate("MainWindow", "2-dim hidden space"))
        self.menuModels.setTitle(_translate("MainWindow", "Models"))
        self.actionmse.setText(_translate("MainWindow", "mse"))
        self.actionbce.setText(_translate("MainWindow", "bce"))
        self.actionload.setText(_translate("MainWindow", "load ..."))
        self.actionsave.setText(_translate("MainWindow", "save ..."))
