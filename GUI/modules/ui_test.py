# -*- coding: utf-8 -*-

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import pyqtgraph as pg
import sys
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(925, 469)
        MainWindow.setStyleSheet(u"")
        MainWindow.setIconSize(QSize(24, 24))
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")

        self.CTLayout = QHBoxLayout(self.centralwidget)
        self.CTLayout.setObjectName(u"CTLayout")

        self.LeftInfo = QVBoxLayout()
        self.LeftInfo.setObjectName(u"LeftInfo")
        self.btnOpenCT = QPushButton(self.centralwidget)
        self.btnOpenCT.setObjectName(u"btnOpenCT")
        self.LeftInfo.addWidget(self.btnOpenCT)
        self.btnShowInfo = QPushButton(self.centralwidget)
        self.btnShowInfo.setObjectName(u"btnShowInfo")
        self.LeftInfo.addWidget(self.btnShowInfo)

        self.CTLayout.addLayout(self.LeftInfo)

        self.QListWidget = QListWidget(self.centralwidget)
        self.QListWidget.setObjectName(u"QListWidget")
        self.LeftInfo.addWidget(self.QListWidget)

        self.pg_3d = QVTKRenderWindowInteractor()
        self.pg_tra = pg.ImageView()
        self.pg_sag = pg.ImageView()
        self.pg_cor = pg.ImageView()

        self.middlect = QVBoxLayout()
        self.middlect.setObjectName(u"middlect")
        self.middlect.addWidget(self.pg_tra)
        self.middlect.addWidget(self.pg_sag)

        self.rightct = QVBoxLayout()
        self.rightct.setObjectName(u"rightct")
        self.rightct.addWidget(self.pg_3d)
        self.rightct.addWidget(self.pg_cor)
        self.rightct.setStretch(0, 1)
        self.rightct.setStretch(1, 1)

        self.CTLayout.addLayout(self.middlect)
        self.CTLayout.addLayout(self.rightct)
        self.CTLayout.setStretch(0, 2)
        self.CTLayout.setStretch(1, 7)
        self.CTLayout.setStretch(2, 7)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 925, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.btnOpenCT.setText(QCoreApplication.translate("MainWindow", u"Open CT", None))
        self.btnShowInfo.setText(QCoreApplication.translate("MainWindow", u"Show Info", None))
