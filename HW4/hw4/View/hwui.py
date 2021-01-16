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
        MainWindow.resize(1056, 929)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.FuncLayout = QtWidgets.QHBoxLayout()
        self.FuncLayout.setObjectName("FuncLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.FuncLayout.addWidget(self.label)
        self.EpochLabel = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.EpochLabel.setFont(font)
        self.EpochLabel.setText("")
        self.EpochLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.EpochLabel.setObjectName("EpochLabel")
        self.FuncLayout.addWidget(self.EpochLabel)
        self.TrainBtn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.TrainBtn.setFont(font)
        self.TrainBtn.setObjectName("TrainBtn")
        self.FuncLayout.addWidget(self.TrainBtn)
        self.ClearBtn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.ClearBtn.setFont(font)
        self.ClearBtn.setObjectName("ClearBtn")
        self.FuncLayout.addWidget(self.ClearBtn)
        self.verticalLayout.addLayout(self.FuncLayout)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy)
        self.progressBar.setMaximumSize(QtCore.QSize(16777215, 5))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(False)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
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
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem3)
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
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem4)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem5)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout.addWidget(self.label_5)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem6)
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
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem7)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
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
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "Auto Encoder"))
        self.label.setText(_translate("MainWindow", "epoch"))
        self.TrainBtn.setText(_translate("MainWindow", "Train"))
        self.ClearBtn.setText(_translate("MainWindow", "Clear"))
        self.label_4.setText(_translate("MainWindow", ">>"))
        self.label_6.setText(_translate("MainWindow", "x:"))
        self.label_7.setText(_translate("MainWindow", "y:"))
        self.label_5.setText(_translate("MainWindow", ">>"))
        self.label_2.setText(_translate("MainWindow", "2-dim hidden space"))
