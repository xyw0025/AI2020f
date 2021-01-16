# -*- coding: utf-8 -*-


import sys
from PyQt5.QtWidgets import QApplication
from controller import AppWidget
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = AppWidget()
    w.show()
    sys.exit(app.exec_())
