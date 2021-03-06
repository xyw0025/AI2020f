# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hwui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from View.PaintBoard import PaintBoard
from View.NeuralBoard import NeuralBoard

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1461, 648)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.paintBoard = PaintBoard()
        self.paintBoard.setObjectName("paintBoard")
        self.horizontalLayout.addWidget(self.paintBoard)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.horizontalLayout.addWidget(self.line_2)
        self.neuralBoard = NeuralBoard((16,16,10))
        self.neuralBoard.setObjectName("neuralBoard")
        self.horizontalLayout.addWidget(self.neuralBoard)
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.horizontalLayout.addWidget(self.line_3)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_23.setFont(font)
        self.label_23.setAlignment(QtCore.Qt.AlignCenter)
        self.label_23.setObjectName("label_23")
        self.verticalLayout_2.addWidget(self.label_23)
        self.LayerLayout1 = QtWidgets.QFormLayout()
        self.LayerLayout1.setObjectName("LayerLayout1")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.LayerLayout1.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.LayerLayout1.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.LayerLayout1.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.LayerLayout1.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_2)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.LayerLayout1.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.LayerLayout1.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_3)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.LayerLayout1.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_4.setFont(font)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.LayerLayout1.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_4)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.LayerLayout1.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_6.setFont(font)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.LayerLayout1.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_6)
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.LayerLayout1.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_5.setFont(font)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.LayerLayout1.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.lineEdit_5)
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.LayerLayout1.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_12)
        self.lineEdit_14 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_14.setFont(font)
        self.lineEdit_14.setObjectName("lineEdit_14")
        self.LayerLayout1.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.lineEdit_14)
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.LayerLayout1.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.lineEdit_7 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_7.setFont(font)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.LayerLayout1.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.lineEdit_7)
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.LayerLayout1.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.label_14)
        self.lineEdit_13 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_13.setFont(font)
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.LayerLayout1.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.lineEdit_13)
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.LayerLayout1.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.label_15)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_8.setFont(font)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.LayerLayout1.setWidget(9, QtWidgets.QFormLayout.FieldRole, self.lineEdit_8)
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.LayerLayout1.setWidget(10, QtWidgets.QFormLayout.LabelRole, self.label_16)
        self.lineEdit_12 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_12.setFont(font)
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.LayerLayout1.setWidget(10, QtWidgets.QFormLayout.FieldRole, self.lineEdit_12)
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.LayerLayout1.setWidget(11, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.lineEdit_11 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_11.setFont(font)
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.LayerLayout1.setWidget(11, QtWidgets.QFormLayout.FieldRole, self.lineEdit_11)
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.LayerLayout1.setWidget(12, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.lineEdit_15 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_15.setFont(font)
        self.lineEdit_15.setObjectName("lineEdit_15")
        self.LayerLayout1.setWidget(12, QtWidgets.QFormLayout.FieldRole, self.lineEdit_15)
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.LayerLayout1.setWidget(13, QtWidgets.QFormLayout.LabelRole, self.label_19)
        self.lineEdit_10 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_10.setFont(font)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.LayerLayout1.setWidget(13, QtWidgets.QFormLayout.FieldRole, self.lineEdit_10)
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.LayerLayout1.setWidget(14, QtWidgets.QFormLayout.LabelRole, self.label_20)
        self.lineEdit_16 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_16.setFont(font)
        self.lineEdit_16.setObjectName("lineEdit_16")
        self.LayerLayout1.setWidget(14, QtWidgets.QFormLayout.FieldRole, self.lineEdit_16)
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.LayerLayout1.setWidget(15, QtWidgets.QFormLayout.LabelRole, self.label_22)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_9.setFont(font)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.LayerLayout1.setWidget(15, QtWidgets.QFormLayout.FieldRole, self.lineEdit_9)
        self.verticalLayout_2.addLayout(self.LayerLayout1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem1)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_24.setFont(font)
        self.label_24.setAlignment(QtCore.Qt.AlignCenter)
        self.label_24.setObjectName("label_24")
        self.verticalLayout_4.addWidget(self.label_24)
        self.LayerLayout2 = QtWidgets.QFormLayout()
        self.LayerLayout2.setObjectName("LayerLayout2")
        self.label_25 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_25.setFont(font)
        self.label_25.setObjectName("label_25")
        self.LayerLayout2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_25)
        self.lineEdit_17 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_17.setFont(font)
        self.lineEdit_17.setObjectName("lineEdit_17")
        self.LayerLayout2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_17)
        self.label_26 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_26.setFont(font)
        self.label_26.setObjectName("label_26")
        self.LayerLayout2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_26)
        self.lineEdit_18 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_18.setFont(font)
        self.lineEdit_18.setObjectName("lineEdit_18")
        self.LayerLayout2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_18)
        self.label_27 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_27.setFont(font)
        self.label_27.setObjectName("label_27")
        self.LayerLayout2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_27)
        self.lineEdit_19 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_19.setFont(font)
        self.lineEdit_19.setObjectName("lineEdit_19")
        self.LayerLayout2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_19)
        self.label_28 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_28.setFont(font)
        self.label_28.setObjectName("label_28")
        self.LayerLayout2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_28)
        self.lineEdit_20 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_20.setFont(font)
        self.lineEdit_20.setObjectName("lineEdit_20")
        self.LayerLayout2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_20)
        self.label_29 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_29.setFont(font)
        self.label_29.setObjectName("label_29")
        self.LayerLayout2.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_29)
        self.lineEdit_21 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_21.setFont(font)
        self.lineEdit_21.setObjectName("lineEdit_21")
        self.LayerLayout2.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_21)
        self.label_30 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_30.setFont(font)
        self.label_30.setObjectName("label_30")
        self.LayerLayout2.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_30)
        self.lineEdit_22 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_22.setFont(font)
        self.lineEdit_22.setObjectName("lineEdit_22")
        self.LayerLayout2.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.lineEdit_22)
        self.label_31 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_31.setFont(font)
        self.label_31.setObjectName("label_31")
        self.LayerLayout2.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_31)
        self.lineEdit_23 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_23.setFont(font)
        self.lineEdit_23.setObjectName("lineEdit_23")
        self.LayerLayout2.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.lineEdit_23)
        self.label_32 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_32.setFont(font)
        self.label_32.setObjectName("label_32")
        self.LayerLayout2.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_32)
        self.lineEdit_24 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_24.setFont(font)
        self.lineEdit_24.setObjectName("lineEdit_24")
        self.LayerLayout2.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.lineEdit_24)
        self.label_33 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_33.setFont(font)
        self.label_33.setObjectName("label_33")
        self.LayerLayout2.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.label_33)
        self.lineEdit_25 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_25.setFont(font)
        self.lineEdit_25.setObjectName("lineEdit_25")
        self.LayerLayout2.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.lineEdit_25)
        self.label_34 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_34.setFont(font)
        self.label_34.setObjectName("label_34")
        self.LayerLayout2.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.label_34)
        self.lineEdit_26 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_26.setFont(font)
        self.lineEdit_26.setObjectName("lineEdit_26")
        self.LayerLayout2.setWidget(9, QtWidgets.QFormLayout.FieldRole, self.lineEdit_26)
        self.label_35 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_35.setFont(font)
        self.label_35.setObjectName("label_35")
        self.LayerLayout2.setWidget(10, QtWidgets.QFormLayout.LabelRole, self.label_35)
        self.lineEdit_27 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_27.setFont(font)
        self.lineEdit_27.setObjectName("lineEdit_27")
        self.LayerLayout2.setWidget(10, QtWidgets.QFormLayout.FieldRole, self.lineEdit_27)
        self.label_36 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_36.setFont(font)
        self.label_36.setObjectName("label_36")
        self.LayerLayout2.setWidget(11, QtWidgets.QFormLayout.LabelRole, self.label_36)
        self.lineEdit_28 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_28.setFont(font)
        self.lineEdit_28.setObjectName("lineEdit_28")
        self.LayerLayout2.setWidget(11, QtWidgets.QFormLayout.FieldRole, self.lineEdit_28)
        self.label_37 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_37.setFont(font)
        self.label_37.setObjectName("label_37")
        self.LayerLayout2.setWidget(12, QtWidgets.QFormLayout.LabelRole, self.label_37)
        self.lineEdit_29 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_29.setFont(font)
        self.lineEdit_29.setObjectName("lineEdit_29")
        self.LayerLayout2.setWidget(12, QtWidgets.QFormLayout.FieldRole, self.lineEdit_29)
        self.label_38 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_38.setFont(font)
        self.label_38.setObjectName("label_38")
        self.LayerLayout2.setWidget(13, QtWidgets.QFormLayout.LabelRole, self.label_38)
        self.lineEdit_30 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_30.setFont(font)
        self.lineEdit_30.setObjectName("lineEdit_30")
        self.LayerLayout2.setWidget(13, QtWidgets.QFormLayout.FieldRole, self.lineEdit_30)
        self.label_39 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_39.setFont(font)
        self.label_39.setObjectName("label_39")
        self.LayerLayout2.setWidget(14, QtWidgets.QFormLayout.LabelRole, self.label_39)
        self.lineEdit_31 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_31.setFont(font)
        self.lineEdit_31.setObjectName("lineEdit_31")
        self.LayerLayout2.setWidget(14, QtWidgets.QFormLayout.FieldRole, self.lineEdit_31)
        self.label_40 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_40.setFont(font)
        self.label_40.setObjectName("label_40")
        self.LayerLayout2.setWidget(15, QtWidgets.QFormLayout.LabelRole, self.label_40)
        self.lineEdit_32 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.lineEdit_32.setFont(font)
        self.lineEdit_32.setObjectName("lineEdit_32")
        self.LayerLayout2.setWidget(15, QtWidgets.QFormLayout.FieldRole, self.lineEdit_32)
        self.verticalLayout_4.addLayout(self.LayerLayout2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem2)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_21.setFont(font)
        self.label_21.setAlignment(QtCore.Qt.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.verticalLayout_3.addWidget(self.label_21)
        self.LayerLayout3 = QtWidgets.QFormLayout()
        self.LayerLayout3.setObjectName("LayerLayout3")
        self.Label0 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.Label0.setFont(font)
        self.Label0.setObjectName("Label0")
        self.LayerLayout3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.Label0)
        self.LineEdit0 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.LineEdit0.setFont(font)
        self.LineEdit0.setAlignment(QtCore.Qt.AlignCenter)
        self.LineEdit0.setReadOnly(True)
        self.LineEdit0.setObjectName("LineEdit0")
        self.LayerLayout3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.LineEdit0)
        self.Label1 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.Label1.setFont(font)
        self.Label1.setObjectName("Label1")
        self.LayerLayout3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.Label1)
        self.LineEdit1 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.LineEdit1.setFont(font)
        self.LineEdit1.setAlignment(QtCore.Qt.AlignCenter)
        self.LineEdit1.setReadOnly(True)
        self.LineEdit1.setObjectName("LineEdit1")
        self.LayerLayout3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.LineEdit1)
        self.Label2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.Label2.setFont(font)
        self.Label2.setObjectName("Label2")
        self.LayerLayout3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.Label2)
        self.LineEdit2 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.LineEdit2.setFont(font)
        self.LineEdit2.setAlignment(QtCore.Qt.AlignCenter)
        self.LineEdit2.setReadOnly(True)
        self.LineEdit2.setObjectName("LineEdit2")
        self.LayerLayout3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.LineEdit2)
        self.Label3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.Label3.setFont(font)
        self.Label3.setObjectName("Label3")
        self.LayerLayout3.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.Label3)
        self.LineEdit3 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.LineEdit3.setFont(font)
        self.LineEdit3.setAlignment(QtCore.Qt.AlignCenter)
        self.LineEdit3.setReadOnly(True)
        self.LineEdit3.setObjectName("LineEdit3")
        self.LayerLayout3.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.LineEdit3)
        self.Label4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.Label4.setFont(font)
        self.Label4.setObjectName("Label4")
        self.LayerLayout3.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.Label4)
        self.LineEdit4 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.LineEdit4.setFont(font)
        self.LineEdit4.setAlignment(QtCore.Qt.AlignCenter)
        self.LineEdit4.setReadOnly(True)
        self.LineEdit4.setObjectName("LineEdit4")
        self.LayerLayout3.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.LineEdit4)
        self.Label5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.Label5.setFont(font)
        self.Label5.setObjectName("Label5")
        self.LayerLayout3.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.Label5)
        self.LineEdit5 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.LineEdit5.setFont(font)
        self.LineEdit5.setAlignment(QtCore.Qt.AlignCenter)
        self.LineEdit5.setReadOnly(True)
        self.LineEdit5.setObjectName("LineEdit5")
        self.LayerLayout3.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.LineEdit5)
        self.Label6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.Label6.setFont(font)
        self.Label6.setObjectName("Label6")
        self.LayerLayout3.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.Label6)
        self.LineEdit6 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.LineEdit6.setFont(font)
        self.LineEdit6.setAlignment(QtCore.Qt.AlignCenter)
        self.LineEdit6.setReadOnly(True)
        self.LineEdit6.setObjectName("LineEdit6")
        self.LayerLayout3.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.LineEdit6)
        self.Label7 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.Label7.setFont(font)
        self.Label7.setObjectName("Label7")
        self.LayerLayout3.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.Label7)
        self.LineEdit7 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.LineEdit7.setFont(font)
        self.LineEdit7.setAlignment(QtCore.Qt.AlignCenter)
        self.LineEdit7.setReadOnly(True)
        self.LineEdit7.setObjectName("LineEdit7")
        self.LayerLayout3.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.LineEdit7)
        self.Label8 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.Label8.setFont(font)
        self.Label8.setObjectName("Label8")
        self.LayerLayout3.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.Label8)
        self.LineEdit8 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.LineEdit8.setFont(font)
        self.LineEdit8.setAlignment(QtCore.Qt.AlignCenter)
        self.LineEdit8.setReadOnly(True)
        self.LineEdit8.setObjectName("LineEdit8")
        self.LayerLayout3.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.LineEdit8)
        self.Label9 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.Label9.setFont(font)
        self.Label9.setObjectName("Label9")
        self.LayerLayout3.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.Label9)
        self.LineEdit9 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.LineEdit9.setFont(font)
        self.LineEdit9.setAlignment(QtCore.Qt.AlignCenter)
        self.LineEdit9.setReadOnly(True)
        self.LineEdit9.setObjectName("LineEdit9")
        self.LayerLayout3.setWidget(9, QtWidgets.QFormLayout.FieldRole, self.LineEdit9)
        self.verticalLayout_3.addLayout(self.LayerLayout3)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem3)
        self.clearButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.clearButton.setFont(font)
        self.clearButton.setObjectName("clearButton")
        self.verticalLayout_3.addWidget(self.clearButton)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_23.setText(_translate("MainWindow", "Layer1"))
        self.label_3.setText(_translate("MainWindow", "0"))
        self.label_7.setText(_translate("MainWindow", "1"))
        self.label_8.setText(_translate("MainWindow", "2"))
        self.label_9.setText(_translate("MainWindow", "3"))
        self.label_10.setText(_translate("MainWindow", "4"))
        self.label_11.setText(_translate("MainWindow", "5"))
        self.label_12.setText(_translate("MainWindow", "6"))
        self.label_13.setText(_translate("MainWindow", "7"))
        self.label_14.setText(_translate("MainWindow", "8"))
        self.label_15.setText(_translate("MainWindow", "9"))
        self.label_16.setText(_translate("MainWindow", "10"))
        self.label_17.setText(_translate("MainWindow", "11"))
        self.label_18.setText(_translate("MainWindow", "12"))
        self.label_19.setText(_translate("MainWindow", "13"))
        self.label_20.setText(_translate("MainWindow", "14"))
        self.label_22.setText(_translate("MainWindow", "15"))
        self.label_24.setText(_translate("MainWindow", "Layer2"))
        self.label_25.setText(_translate("MainWindow", "0"))
        self.label_26.setText(_translate("MainWindow", "1"))
        self.label_27.setText(_translate("MainWindow", "2"))
        self.label_28.setText(_translate("MainWindow", "3"))
        self.label_29.setText(_translate("MainWindow", "4"))
        self.label_30.setText(_translate("MainWindow", "5"))
        self.label_31.setText(_translate("MainWindow", "6"))
        self.label_32.setText(_translate("MainWindow", "7"))
        self.label_33.setText(_translate("MainWindow", "8"))
        self.label_34.setText(_translate("MainWindow", "9"))
        self.label_35.setText(_translate("MainWindow", "10"))
        self.label_36.setText(_translate("MainWindow", "11"))
        self.label_37.setText(_translate("MainWindow", "12"))
        self.label_38.setText(_translate("MainWindow", "13"))
        self.label_39.setText(_translate("MainWindow", "14"))
        self.label_40.setText(_translate("MainWindow", "15"))
        self.label_21.setText(_translate("MainWindow", "Layer3"))
        self.Label0.setText(_translate("MainWindow", "0"))
        self.Label1.setText(_translate("MainWindow", "1"))
        self.Label2.setText(_translate("MainWindow", "2"))
        self.Label3.setText(_translate("MainWindow", "3"))
        self.Label4.setText(_translate("MainWindow", "4"))
        self.Label5.setText(_translate("MainWindow", "5"))
        self.Label6.setText(_translate("MainWindow", "6"))
        self.Label7.setText(_translate("MainWindow", "7"))
        self.Label8.setText(_translate("MainWindow", "8"))
        self.Label9.setText(_translate("MainWindow", "9"))
        self.clearButton.setText(_translate("MainWindow", "Clear"))

