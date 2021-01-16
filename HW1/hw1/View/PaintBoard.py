'''
Created on 2018年8月9日

@author: Freedom
'''
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap, QPainter, QPaintEvent, QMouseEvent, QPen,\
    QColor, QBrush
from PyQt5.QtCore import Qt, QPoint, QSize

class PaintBoard(QWidget):



    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)

        self.__InitData() #先初始化數據，再初始化界面
        self.__InitView()

    def __InitData(self):

        self.__size = QSize(560,560)

        #新建QPixmap作爲畫板，尺寸爲__size
        self.__board = QPixmap(self.__size)
        self.__board.fill(Qt.black) #用白色填充畫板

        self.__IsEmpty = True #默認爲空畫板 
        self.EraserMode = False #默認爲禁用橡皮擦模式

        self.__lastPos = QPoint(0,0)#上一次鼠標位置
        self.__currentPos = QPoint(0,0)#當前的鼠標位置

        self.__painter = QPainter()#新建繪圖工具

        self.__thickness = 20       #默認畫筆粗細爲10px
        self.__penColor = QColor("white")#設置默認畫筆顏色爲白色
        self.__brushColor = QColor("white")#設置默認筆刷顏色爲白色
        self.__colorList = QColor.colorNames() #獲取顏色列表

    def __InitView(self):
        #設置界面的尺寸爲__size
        self.setFixedSize(self.__size)

    def Clear(self):
        #清空畫板
        self.__board.fill(Qt.black)
        self.update()
        self.__IsEmpty = True

    def setMode(self, mode):
        self.EraserMode = mode

    def ChangePenColor(self, color="black"):
        #改變畫筆顏色
        self.__penColor = QColor(color)

    def ChangePenThickness(self, thickness=10):
        #改變畫筆粗細
        self.__thickness = thickness

    def IsEmpty(self):
        #返回畫板是否爲空
        return self.__IsEmpty

    def GetContentAsQImage(self):
        #獲取畫板內容（返回QImage）
        image = self.__board.toImage()
        return image

    def paintEvent(self, paintEvent):
        #繪圖事件
        #繪圖時必須使用QPainter的實例，此處爲__painter
        #繪圖在begin()函數與end()函數間進行
        #begin(param)的參數要指定繪圖設備，即把圖畫在哪裏
        #drawPixmap用於繪製QPixmap類型的對象
        self.__painter.begin(self)
        # 0,0爲繪圖的左上角起點的座標，__board即要繪製的圖
        self.__painter.drawPixmap(0,0,self.__board)
        self.__painter.end()

    def mousePressEvent(self, mouseEvent):
        #鼠標按下時，獲取鼠標的當前位置保存爲上一次位置
        self.__currentPos =  mouseEvent.pos()
        self.__lastPos = self.__currentPos


    def mouseMoveEvent(self, mouseEvent):
        #鼠標移動時，更新當前位置，並在上一個位置和當前位置間畫線
        self.__currentPos =  mouseEvent.pos()
        self.__painter.begin(self.__board)

        if self.EraserMode == False:
            #非橡皮擦模式
            self.__painter.setPen(QPen(self.__penColor,1)) #設置畫筆顏色，粗細
            self.__painter.setBrush(QBrush(self.__brushColor))
        else:
            #橡皮擦模式下畫筆爲純白色，粗細爲10
            self.__painter.setPen(QPen(Qt.black,10))
            

        #畫線    
        self.__painter.drawEllipse(self.__currentPos, self.__thickness, self.__thickness)
        self.__painter.end()
        self.__lastPos = self.__currentPos

        self.update() #更新顯示

    def mouseReleaseEvent(self, mouseEvent):
        self.__IsEmpty = False #畫板不再爲空
        