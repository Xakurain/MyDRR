# -*- coding: utf-8 -*-

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import pyqtgraph as pg
import sys
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from modules import funcs

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(925, 469)
        MainWindow.setStyleSheet(u"")
        MainWindow.setIconSize(QSize(24, 24))
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setStyleSheet(u"")
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.sideBar = QWidget(self.centralwidget)
        self.sideBar.setObjectName(u"sideBar")
        self.sideBar.setStyleSheet(u"QWidget{\n"
            "	background-color: rgb(228, 228, 228);\n"
            "}")
        self.gridLayout = QGridLayout(self.sideBar)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.btnRegi = QToolButton(self.sideBar)
        self.btnRegi.setObjectName(u"btnRegi")
        self.btnRegi.setStyleSheet(u"/* \u9ed8\u8ba4 */\n"
            "QToolButton{   \n"
            "	border-top: 3px outset transparent;\n"
            "	border-bottom: 7px outset transparent;\n"
            "	border-right: 3px outset transparent;\n"
            "	border-left: 3px outset transparent;\n"
            "    min-width: 80px;\n"
            "    min-height: 80px;\n"
            "	background-color: rgb(228, 228, 228);\n"
            "}\n"
            "\n"
            "/* \u9f20\u6807\u60ac\u505c */\n"
            "QToolButton:hover{\n"
            "	background-color: rgb(205, 205, 205);\n"
            "}\n"
            "\n"
            "/* \u70b9\u51fb\u548c\u6309\u4e0b */\n"
            "QToolButton:pressed,QToolButton:checked{\n"
            "	border-left: 3px outset rgb(93, 95, 97);\n"
            "	background-color: rgb(246, 246, 246);	\n"
            "}\n"
            "\n"
            "QPushButton:default {\n"
            "    border-color: navy; /* make the default button prominent */\n"
            "}\n"
            "\n"
            "")
        icon = QIcon()
        icon.addFile(u"icons/Regi.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btnRegi.setIcon(icon)
        self.btnRegi.setIconSize(QSize(48, 48))
        self.btnRegi.setCheckable(True)
        self.btnRegi.setAutoExclusive(True)
        self.btnRegi.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        self.gridLayout.addWidget(self.btnRegi, 0, 0, 1, 1)

        self.btnCT = QToolButton(self.sideBar)
        self.btnCT.setObjectName(u"btnCT")
        self.btnCT.setMinimumSize(QSize(86, 90))
        self.btnCT.setStyleSheet(u"/* \u9ed8\u8ba4 */\n"
            "QToolButton{   \n"
            "	border-top: 3px outset transparent;\n"
            "	border-bottom: 7px outset transparent;\n"
            "	border-right: 3px outset transparent;\n"
            "	border-left: 3px outset transparent;\n"
            "    min-width: 80px;\n"
            "    min-height: 80px;\n"
            "	background-color: rgb(228, 228, 228);\n"
            "}\n"
            "\n"
            "/* \u9f20\u6807\u60ac\u505c */\n"
            "QToolButton:hover{\n"
            "	background-color: rgb(205, 205, 205);\n"
            "}\n"
            "\n"
            "/* \u70b9\u51fb\u548c\u6309\u4e0b */\n"
            "QToolButton:pressed,QToolButton:checked{\n"
            "	border-left: 3px outset rgb(93, 95, 97);\n"
            "	background-color: rgb(246, 246, 246);	\n"
            "}\n"
            "")
        icon1 = QIcon()
        icon1.addFile(u"icons/CT.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btnCT.setIcon(icon1)
        self.btnCT.setIconSize(QSize(48, 48))
        self.btnCT.setCheckable(True)
        self.btnCT.setAutoExclusive(True)
        self.btnCT.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        self.btnRegi.setChecked(True)

        self.gridLayout.addWidget(self.btnCT, 1, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 288, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 2, 0, 1, 1)


        self.gridLayout_2.addWidget(self.sideBar, 0, 0, 1, 1)

        self.stackedWidget = QStackedWidget(self.centralwidget)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setStyleSheet(u"")

        #页面1
        self.Regipage = QWidget()
        self.Regipage.setObjectName(u"Regipage")

        # rxFixedLabel
        self.rxFixedLabelLayout = QVBoxLayout()
        self.rxFixedLabelLayout.setObjectName(u"rxFixedLabelLayout")
        self.rxlabel = QLabel(self.Regipage)
        self.rxlabel.setObjectName(u"rxlabel")
        # rxLabel加入垂直布局
        self.rxFixedLabelLayout.addWidget(self.rxlabel, 0, Qt.AlignHCenter)
        self.rxFixedValue = QDoubleSpinBox(self.Regipage)
        self.rxFixedValue.setObjectName(u"rxFixedValue")
        self.rxFixedValue.setMinimum(-98.000000000000000)
        self.rxFixedValue.setMaximum(-82.000000000000000)
        self.rxFixedValue.setValue(-90.000000000000000)
        # rxFixedValue加入垂直布局
        self.rxFixedLabelLayout.addWidget(self.rxFixedValue)

        # ryFixedLabel
        self.ryFixedLabelLayout = QVBoxLayout()
        self.ryFixedLabelLayout.setObjectName(u"ryFixedLabelLayout")
        self.rylabel = QLabel(self.Regipage)
        self.rylabel.setObjectName(u"rylabel")
        # ryLabel加入垂直布局
        self.ryFixedLabelLayout.addWidget(self.rylabel, 0, Qt.AlignHCenter)
        self.ryFixedValue = QDoubleSpinBox(self.Regipage)
        self.ryFixedValue.setObjectName(u"ryFixedValue")
        self.ryFixedValue.setMinimum(-8.000000000000000)
        self.ryFixedValue.setMaximum(8.000000000000000)
        self.ryFixedValue.setValue(0.000000000000000)
        # ryFixedValue加入垂直布局
        self.ryFixedLabelLayout.addWidget(self.ryFixedValue)

        # rzFixedLabel
        self.rzFixedLabelLayout = QVBoxLayout()
        self.rzFixedLabelLayout.setObjectName(u"rzFixedLabelLayout")
        self.rzlabel = QLabel(self.Regipage)
        self.rzlabel.setObjectName(u"rzlabel")
        # rzLabel加入垂直布局
        self.rzFixedLabelLayout.addWidget(self.rzlabel, 0, Qt.AlignHCenter)
        self.rzFixedValue = QDoubleSpinBox(self.Regipage)
        self.rzFixedValue.setObjectName(u"rzFixedValue")
        self.rzFixedValue.setMinimum(-4.000000000000000)
        self.rzFixedValue.setMaximum(4.000000000000000)
        self.rzFixedValue.setValue(0.000000000000000)
        # rzFixedValue加入垂直布局
        self.rzFixedLabelLayout.addWidget(self.rzFixedValue)

        # txFixedLabel
        self.txFixedLabelLayout = QVBoxLayout()
        self.txFixedLabelLayout.setObjectName(u"txFixedLabelLayout")
        self.txlabel = QLabel(self.Regipage)
        self.txlabel.setObjectName(u"txlabel")
        # txLabel加入垂直布局
        self.txFixedLabelLayout.addWidget(self.txlabel, 0, Qt.AlignHCenter)
        self.txFixedValue = QDoubleSpinBox(self.Regipage)
        self.txFixedValue.setObjectName(u"txFixedValue")
        self.txFixedValue.setMinimum(-20.000000000000000)
        self.txFixedValue.setMaximum(20.000000000000000)
        self.txFixedValue.setValue(0.000000000000000)
        # txFixedValue加入垂直布局
        self.txFixedLabelLayout.addWidget(self.txFixedValue)

        # tyFixedLabel
        self.tyFixedLabelLayout = QVBoxLayout()
        self.tyFixedLabelLayout.setObjectName(u"tyFixedLabelLayout")
        self.tylabel = QLabel(self.Regipage)
        self.tylabel.setObjectName(u"tylabel")
        # tyLabel加入垂直布局
        self.tyFixedLabelLayout.addWidget(self.tylabel, 0, Qt.AlignHCenter)
        self.tyFixedValue = QDoubleSpinBox(self.Regipage)
        self.tyFixedValue.setObjectName(u"tyFixedValue")
        self.tyFixedValue.setMinimum(-8.000000000000000)
        self.tyFixedValue.setMaximum(8.000000000000000)
        self.tyFixedValue.setValue(0.000000000000000)
        # tyFixedValue加入垂直布局
        self.tyFixedLabelLayout.addWidget(self.tyFixedValue)

        # tzFixedLabel
        self.tzFixedLabelLayout = QVBoxLayout()
        self.tzFixedLabelLayout.setObjectName(u"tzFixedLabelLayout")
        self.tzlabel = QLabel(self.Regipage)
        self.tzlabel.setObjectName(u"tzlabel")
        # tzLabel加入垂直布局
        self.tzFixedLabelLayout.addWidget(self.tzlabel, 0, Qt.AlignHCenter)
        self.tzFixedValue = QDoubleSpinBox(self.Regipage)
        self.tzFixedValue.setObjectName(u"tzFixedValue")
        self.tzFixedValue.setMinimum(-20.000000000000000)
        self.tzFixedValue.setMaximum(20.000000000000000)
        self.tzFixedValue.setValue(0.000000000000000)
        # tzFixedValue加入垂直布局
        self.tzFixedLabelLayout.addWidget(self.tzFixedValue)

        # inputChoose
        self.inputChooseLayout = QVBoxLayout()
        self.inputChooseLayout.setObjectName(u"inputChooseLayout")
        self.isInputLabel = QComboBox(self.Regipage)
        self.isInputLabel.setObjectName(u"isInputLabel")
        self.isInputLabel.addItem("选择文件")
        self.isInputLabel.addItem("输入参数")
        # isInputLabel加入垂直布局
        self.inputChooseLayout.addWidget(self.isInputLabel)
        self.FileOpenButton = QPushButton(self.Regipage)
        self.FileOpenButton.setObjectName(u"FileOpenButton")
        self.FileOpenButton.setStyleSheet(u"")
        icon = QIcon()
        icon.addFile(u"icons/openfile.png", QSize(), QIcon.Normal, QIcon.Off)
        self.FileOpenButton.setIcon(icon)
        self.FileOpenButton.setCheckable(False)
        self.FileOpenButton.setAutoRepeat(False)
        # FileOpenButton加入垂直布局
        self.inputChooseLayout.addWidget(self.FileOpenButton)

        # LoadandStart
        self.LoadandStartLayout = QVBoxLayout()
        self.LoadandStartLayout.setObjectName(u"LoadandStartLayout")
        self.LoadImage = QPushButton(self.Regipage)
        self.LoadImage.setObjectName(u"LoadImage")
        icon1 = QIcon()
        icon1.addFile(u"icons/loadimage.png", QSize(), QIcon.Normal, QIcon.Off)
        self.LoadImage.setIcon(icon1)
        # LoadImage加入垂直布局
        self.LoadandStartLayout.addWidget(self.LoadImage)

        self.StartRegistrationButton = QPushButton(self.Regipage)
        self.StartRegistrationButton.setObjectName(u"StartRegistrationButton")
        icon2 = QIcon()
        icon2.addFile(u"icons/registration.png", QSize(), QIcon.Normal, QIcon.Off)
        self.StartRegistrationButton.setIcon(icon2)
        # StartRegistrationButton加入垂直布局
        self.LoadandStartLayout.addWidget(self.StartRegistrationButton)

        # RefineandAnalysis
        self.RefineandAnalysisLayout = QVBoxLayout()
        self.RefineandAnalysisLayout.setObjectName(u"RefineandAnalysisLayout")

        self.RefineButton = QPushButton(self.Regipage)
        self.RefineButton.setObjectName(u"RefineButton")
        icon3 = QIcon()
        icon3.addFile(u"icons/refine.png", QSize(), QIcon.Normal, QIcon.Off)
        self.RefineButton.setIcon(icon3)
        # RefineButton加入垂直布局
        self.RefineandAnalysisLayout.addWidget(self.RefineButton)

        self.AnalysisButton = QPushButton(self.Regipage)
        self.AnalysisButton.setObjectName(u"AnalysisButton")
        icon4 = QIcon()
        icon4.addFile(u"icons/analysis.png", QSize(), QIcon.Normal, QIcon.Off)
        self.AnalysisButton.setIcon(icon4)
        # AnalysisButton加入垂直布局
        self.RefineandAnalysisLayout.addWidget(self.AnalysisButton)

        # ImageLoadTop
        self.ImageLoadTopLayout = QHBoxLayout()
        self.ImageLoadTopLayout.setObjectName(u"ImageLoadTopLayout")
        self.ImageLoadTopLayout.addLayout(self.rxFixedLabelLayout)
        self.ImageLoadTopLayout.addLayout(self.ryFixedLabelLayout)
        self.ImageLoadTopLayout.addLayout(self.rzFixedLabelLayout)
        self.ImageLoadTopLayout.addLayout(self.txFixedLabelLayout)
        self.ImageLoadTopLayout.addLayout(self.tyFixedLabelLayout)
        self.ImageLoadTopLayout.addLayout(self.tzFixedLabelLayout)
        self.ImageLoadTopLayout.addLayout(self.inputChooseLayout)
        self.ImageLoadTopLayout.addLayout(self.LoadandStartLayout)
        self.ImageLoadTopLayout.addLayout(self.RefineandAnalysisLayout)

        self.line = QFrame(self.Regipage)
        self.line.setObjectName(u"line")
        self.line.setLineWidth(2)
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        # ImageShowMiddle
        self.Fixedlabel = QLabel(self.Regipage)
        self.Fixedlabel.setObjectName(u"Fixedlabel")
        font = QFont()
        font.setBold(True)
        self.Fixedlabel.setFont(font)
        self.Fixedlabel.setAlignment(Qt.AlignCenter)

        self.FixedImage = QLabel(self.Regipage)
        self.FixedImage.setObjectName(u"FixedImage")

        self.Movedlabel = QLabel(self.Regipage)
        self.Movedlabel.setObjectName(u"Movedlabel")
        self.Movedlabel.setFont(font)
        self.Movedlabel.setAlignment(Qt.AlignCenter)

        self.MovedImage = QLabel(self.Regipage)
        self.MovedImage.setObjectName(u"MovedImage")

        self.Differencelabel = QLabel(self.Regipage)
        self.Differencelabel.setObjectName(u"Differencelabel")
        self.Differencelabel.setFont(font)
        self.Differencelabel.setAlignment(Qt.AlignCenter)

        self.DifferenceImage = QLabel(self.Regipage)
        self.DifferenceImage.setObjectName(u"DifferenceImage")

        # self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.line_2 = QFrame(self.Regipage)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setLineWidth(2)
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        # RegistrationResultBottom
        self.RegistrationResultlabel = QLabel(self.Regipage)
        self.RegistrationResultlabel.setObjectName(u"RegistrationResultlabel")
        font1 = QFont()
        font1.setPointSize(8)
        font1.setBold(True)
        self.RegistrationResultlabel.setFont(font1)
        self.RegistrationResultlabel.setToolTipDuration(0)
        self.RegistrationResultlabel.setLayoutDirection(Qt.LeftToRight)
        self.RegistrationResultlabel.setTextFormat(Qt.AutoText)
        self.RegistrationResultlabel.setAlignment(Qt.AlignCenter)
        self.RegistrationResultlabel.setMargin(2)
        self.RegistrationResultlabel.setTextInteractionFlags(Qt.LinksAccessibleByMouse)

        self.RegistrationValuelabel = QLabel(self.Regipage)
        self.RegistrationValuelabel.setObjectName(u"RegistrationValuelabel")

        # rxMovedLayout
        self.rxMovedLayout = QHBoxLayout()
        self.rxMovedLayout.setObjectName(u"rxMovedLayout")
        self.rxlabel_2 = QLabel(self.Regipage)
        self.rxlabel_2.setObjectName(u"rxlabel_2")
        self.rxMovedLayout.addWidget(self.rxlabel_2)
        self.rxMovedValue = QLineEdit(self.Regipage)
        self.rxMovedValue.setObjectName(u"rxMovedValue")
        self.rxMovedLayout.addWidget(self.rxMovedValue)
        # ryMovedLayout
        self.ryMovedLayout = QHBoxLayout()
        self.rylabel_2 = QLabel(self.Regipage)
        self.rylabel_2.setObjectName(u"rylabel_2")
        self.ryMovedLayout.addWidget(self.rylabel_2)
        self.ryMovedValue = QLineEdit(self.Regipage)
        self.ryMovedValue.setObjectName(u"ryMovedValue")
        self.ryMovedLayout.addWidget(self.ryMovedValue)
        # rzMovedLayout
        self.rzMovedLayout = QHBoxLayout()
        self.rzMovedLayout.setObjectName(u"rzMovedLayout")
        self.rzlabel_2 = QLabel(self.Regipage)
        self.rzlabel_2.setObjectName(u"rzlabel_2")
        self.rzMovedLayout.addWidget(self.rzlabel_2)
        self.rzMovedValue = QLineEdit(self.Regipage)
        self.rzMovedValue.setObjectName(u"rzMovedValue")
        self.rzMovedLayout.addWidget(self.rzMovedValue)
        # txMovedLayout
        self.txMovedLayout = QHBoxLayout()
        self.txMovedLayout.setObjectName(u"txMovedLayout")
        self.txlabel_3 = QLabel(self.Regipage)
        self.txlabel_3.setObjectName(u"txlabel_3")
        self.txMovedLayout.addWidget(self.txlabel_3)
        self.txMovedValue = QLineEdit(self.Regipage)
        self.txMovedValue.setObjectName(u"txMovedValue")
        self.txMovedLayout.addWidget(self.txMovedValue)
        # tyMovedLayout
        self.tyMovedLayout = QHBoxLayout()
        self.tyMovedLayout.setObjectName(u"tyMovedLayout")
        self.tylabel_2 = QLabel(self.Regipage)
        self.tylabel_2.setObjectName(u"tylabel_2")
        self.tyMovedLayout.addWidget(self.tylabel_2)
        self.tyMovedValue = QLineEdit(self.Regipage)
        self.tyMovedValue.setObjectName(u"tyMovedValue")
        self.tyMovedLayout.addWidget(self.tyMovedValue)
        # tzMovedLayout
        self.tzMovedLayout = QHBoxLayout()
        self.tzMovedLayout.setObjectName(u"tzMovedLayout")
        self.tzlabel_2 = QLabel(self.Regipage)
        self.tzlabel_2.setObjectName(u"tzlabel_2")
        self.tzMovedLayout.addWidget(self.tzlabel_2)
        self.tzMovedValue = QLineEdit(self.Regipage)
        self.tzMovedValue.setObjectName(u"tzMovedValue")
        self.tzMovedLayout.addWidget(self.tzMovedValue)

        # MovedValueLayout
        self.MovedValueLayout = QHBoxLayout()
        self.MovedValueLayout.setObjectName(u"MovedValueLayout")
        self.MovedValueLayout.addLayout(self.rxMovedLayout)
        self.MovedValueLayout.addLayout(self.ryMovedLayout)
        self.MovedValueLayout.addLayout(self.rzMovedLayout)
        self.MovedValueLayout.addLayout(self.txMovedLayout)
        self.MovedValueLayout.addLayout(self.tyMovedLayout)
        self.MovedValueLayout.addLayout(self.tzMovedLayout)

        self.Evaluatelabel = QLabel(self.Regipage)
        self.Evaluatelabel.setObjectName(u"Evaluatelabel")

        # NCCLayout
        self.NCCLayout = QHBoxLayout()
        self.NCCLayout.setObjectName(u"NCCLayout")
        self.NCClabel = QLabel(self.Regipage)
        self.NCClabel.setObjectName(u"NCClabel")
        self.NCCLayout.addWidget(self.NCClabel)
        self.NCCValue = QLineEdit(self.Regipage)
        self.NCCValue.setObjectName(u"NCCValue")
        self.NCCLayout.addWidget(self.NCCValue)

        # MILayout
        self.MILayout = QHBoxLayout()
        self.MILayout.setObjectName(u"MILayout")
        self.NMIlabel = QLabel(self.Regipage)
        self.NMIlabel.setObjectName(u"NMIlabel")
        self.MILayout.addWidget(self.NMIlabel)
        self.NMIValue = QLineEdit(self.Regipage)
        self.NMIValue.setObjectName(u"NMIValue")
        self.MILayout.addWidget(self.NMIValue)

        # DICELayout
        self.DICELayout = QHBoxLayout()
        self.DICELayout.setObjectName(u"DICELayout")
        self.DICElabel = QLabel(self.Regipage)
        self.DICElabel.setObjectName(u"DICElabel")
        self.DICELayout.addWidget(self.DICElabel)
        self.DICECValue = QLineEdit(self.Regipage)
        self.DICECValue.setObjectName(u"DICECValue")
        self.DICELayout.addWidget(self.DICECValue)

        # EvaluateLayout
        self.EvaluateLayout = QHBoxLayout()
        self.EvaluateLayout.setObjectName(u"EvaluateLayout")
        self.EvaluateLayout.addLayout(self.NCCLayout)
        self.EvaluateLayout.addLayout(self.MILayout)
        self.EvaluateLayout.addLayout(self.DICELayout)

        # 整体布局
        self.gridLayout = QGridLayout(self.Regipage)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.addLayout(self.ImageLoadTopLayout, 0, 0, 2, 9)
        self.gridLayout.addWidget(self.line, 2, 0, 1, 9)
        self.gridLayout.addWidget(self.Fixedlabel, 3, 0, 1, 3)
        self.gridLayout.addWidget(self.Movedlabel, 3, 3, 1, 3)
        self.gridLayout.addWidget(self.Differencelabel, 3, 6, 1, 3)
        self.gridLayout.addWidget(self.FixedImage, 4, 0, 7, 3)
        self.gridLayout.addWidget(self.MovedImage, 4, 3, 7, 3)
        self.gridLayout.addWidget(self.DifferenceImage, 4, 6, 7, 3)
        self.gridLayout.addWidget(self.line_2, 11, 0, 1, 9)
        self.gridLayout.addWidget(self.RegistrationResultlabel, 12, 0, 1, 9)
        self.gridLayout.addWidget(self.RegistrationValuelabel, 13, 0, 1, 9)
        self.gridLayout.addLayout(self.MovedValueLayout, 14, 0, 1, 9)
        self.gridLayout.addWidget(self.Evaluatelabel, 15, 0, 1, 9)
        self.gridLayout.addLayout(self.EvaluateLayout, 16, 0, 1, 9)

        # 展示图片
        funcs.showImage(self.FixedImage, "img/FixedImageInit.png")
        funcs.showImage(self.MovedImage, "img/MovedImageInit.png")
        funcs.showImage(self.DifferenceImage, "img/DifferenceImageInit.png")

        self.rxFixedValue.setEnabled(False)
        self.ryFixedValue.setEnabled(False)
        self.rzFixedValue.setEnabled(False)
        self.txFixedValue.setEnabled(False)
        self.tyFixedValue.setEnabled(False)
        self.tzFixedValue.setEnabled(False)

        self.stackedWidget.addWidget(self.Regipage)

        #页面2
        self.CTpage = QWidget()
        self.CTpage.setObjectName(u"CTpage")

        self.CTLayout = QHBoxLayout(self.CTpage)
        self.CTLayout.setObjectName(u"CTLayout")

        #左侧信息栏
        self.LeftInfo = QVBoxLayout()
        self.LeftInfo.setObjectName(u"LeftInfo")

        self.btnOpenCT = QPushButton(self.CTpage)
        self.btnOpenCT.setObjectName(u"btnOpenCT")
        self.LeftInfo.addWidget(self.btnOpenCT)

        self.btnShow3D = QPushButton(self.CTpage)
        self.btnShow3D.setObjectName(u"btnShow3D")
        self.LeftInfo.addWidget(self.btnShow3D)

        self.btnShowInfo = QPushButton(self.CTpage)
        self.btnShowInfo.setObjectName(u"btnShowInfo")
        self.LeftInfo.addWidget(self.btnShowInfo)

        self.CTLayout.addLayout(self.LeftInfo)

        self.ListWidget = QListWidget(self.CTpage)
        self.ListWidget.setObjectName(u"QListWidget")
        self.ListWidget.setSpacing(5)
        self.ListWidget.setWordWrap(True)

        self.ListWidget.setStyleSheet("""
            QListWidget {
                background-color: rgb(245, 245, 245);
            }
            QListWidget::item:hover {
                background-color: rgb(135, 206, 250);
            }
        """)

        self.ListWidget.addItem("姓名：")
        self.ListWidget.addItem("")
        self.ListWidget.addItem("性别：")
        self.ListWidget.addItem("")
        self.ListWidget.addItem("ID：")
        self.ListWidget.addItem("")
        self.ListWidget.addItem("模态：")
        self.ListWidget.addItem("")
        self.ListWidget.addItem("图像尺寸：")
        self.ListWidget.addItem("")
        self.ListWidget.addItem("体素尺寸：")
        self.ListWidget.addItem("")
        self.ListWidget.addItem("图像原点：")
        self.ListWidget.addItem("")
        for i in range(14):
            if i%2 == 0:
                self.ListWidget.item(i).setFont(QFont("Roman times", 12, QFont.Bold))
            else:
                self.ListWidget.item(i).setTextAlignment(Qt.AlignCenter)



        self.LeftInfo.addWidget(self.ListWidget)



        #CT及三维渲染
        self.pg_3d =QVTKRenderWindowInteractor()
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

        self.stackedWidget.addWidget(self.CTpage)

        self.gridLayout_2.addWidget(self.stackedWidget, 0, 1, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 925, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"2D/3D Registration", None))
        self.btnRegi.setText(QCoreApplication.translate("MainWindow", u"配准", None))
        self.btnCT.setText(QCoreApplication.translate("MainWindow", u"影像", None))
        self.rxlabel.setText(QCoreApplication.translate("MainWindow", u"rx", None))
        self.rylabel.setText(QCoreApplication.translate("MainWindow", u"ry", None))
        self.rzlabel.setText(QCoreApplication.translate("MainWindow", u"rz", None))
        self.txlabel.setText(QCoreApplication.translate("MainWindow", u"tx", None))
        self.tylabel.setText(QCoreApplication.translate("MainWindow", u"ty", None))
        self.tzlabel.setText(QCoreApplication.translate("MainWindow", u"tz", None))
        # self.isInputLabel.setText(QCoreApplication.translate("MainWindow", u"输入参数", None))
        self.FileOpenButton.setText(QCoreApplication.translate("MainWindow", u"打开文件", None))
        self.LoadImage.setText(QCoreApplication.translate("MainWindow", u"加载图像", None))
        self.StartRegistrationButton.setText(QCoreApplication.translate("MainWindow", u"开始配准", None))
        self.RefineButton.setText(QCoreApplication.translate("MainWindow", u"精细配准", None))
        self.AnalysisButton.setText(QCoreApplication.translate("MainWindow", u"指标分析", None))
        self.Fixedlabel.setText(QCoreApplication.translate("MainWindow", u"参考图像", None))
        self.FixedImage.setText("")
        self.Movedlabel.setText(QCoreApplication.translate("MainWindow", u"配准图像", None))
        self.MovedImage.setText("")
        self.Differencelabel.setText(QCoreApplication.translate("MainWindow", u"差值图", None))
        self.DifferenceImage.setText("")
        self.RegistrationResultlabel.setText(QCoreApplication.translate("MainWindow", u"配准结果", None))
        self.RegistrationValuelabel.setText(QCoreApplication.translate("MainWindow", u"配准参数：", None))
        self.rxlabel_2.setText(QCoreApplication.translate("MainWindow", u"rx", None))
        self.rylabel_2.setText(QCoreApplication.translate("MainWindow", u"ry", None))
        self.rzlabel_2.setText(QCoreApplication.translate("MainWindow", u"rz", None))
        self.txlabel_3.setText(QCoreApplication.translate("MainWindow", u"tx", None))
        self.tylabel_2.setText(QCoreApplication.translate("MainWindow", u"ty", None))
        self.tzlabel_2.setText(QCoreApplication.translate("MainWindow", u"tz", None))
        self.Evaluatelabel.setText(QCoreApplication.translate("MainWindow", u"评估系数：", None))
        self.NCClabel.setText(QCoreApplication.translate("MainWindow", u"NCC", None))
        self.NMIlabel.setText(QCoreApplication.translate("MainWindow", u"NMI", None))
        self.DICElabel.setText(QCoreApplication.translate("MainWindow", u"DICE", None))
        self.btnOpenCT.setText(QCoreApplication.translate("MainWindow", u"打开CT", None))
        self.btnShow3D.setText(QCoreApplication.translate("MainWindow", u"三维渲染", None))
        self.btnShowInfo.setText(QCoreApplication.translate("MainWindow", u"显示信息", None))
    # retranslateUi

