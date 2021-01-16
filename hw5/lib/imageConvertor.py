# -*- coding: utf-8 -*-

import cv2
from PyQt5.QtGui import qRed, qGreen, qBlue, QPixmap, QImage
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def numpy2qimg(arr):
    img = arr
    img *= 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
    
def numpy2qimg_rgba(arr):
    img = arr
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)

def qimg2numpy(img):
    img = img.scaled(28, 28)
    img = QImage2CV(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = img.astype(np.float32)
    img /= 255.0
    return img.reshape(-1, 28, 28, 1)
    
def QImage2CV(qimg):        
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

def plot2qimg(figure):
    canvas = FigureCanvas(figure)
    canvas.draw()
    buf = canvas.buffer_rgba()
    x = np.asarray(buf)
    img = numpy2qimg_rgba(x)
    return img