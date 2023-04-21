#coding=utf-8
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

import SimpleITK as sitk
import numpy as np
import os
import vtkmodules.all as vtk
from modules import ui_test
import qdarktheme
envpath = 'D:\\Python\\Anaconda3\\Lib\\site-packages\\PySide6\\plugins\\platforms'
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = envpath

class MyWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        # SET AS GLOBAL WIDGETS
        # ///////////////////////////////////////////////////////////////
        self.ui = ui_test.Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

if __name__ == '__main__':
    app = QApplication([])
    qdarktheme.setup_theme("light")

    window = MyWindow()
    window.show()
    app.exec()