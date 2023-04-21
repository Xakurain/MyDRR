
from PySide6.QtCore import (QCoreApplication, QMetaObject, QRect,
                            QSize, Qt)
from PySide6.QtGui import (QFont, QIcon)
from PySide6.QtWidgets import (QDoubleSpinBox, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMenuBar, QPushButton, QStatusBar, QVBoxLayout, QWidget,
                               QComboBox, QStackedLayout)

from modules import funcs


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(834, 506)

        #主窗口
        self.mainWidget = QWidget(MainWindow)
        self.mainWidget.setObjectName(u"mainWidget")
        # app布局
        self.appLayout = QHBoxLayout(self.mainWidget)

        #切换栏
        self.leftbar = QWidget()
        self.leftbar.setObjectName(u"leftbar")
        self.btn_regi = QPushButton(self.leftbar)
        self.btn_regi.setObjectName(u"btn_regi")
        self.btn_ct = QPushButton(self.leftbar)
        self.btn_ct.setObjectName(u"btn_ct")
        self.barLayout = QVBoxLayout(self.leftbar)
        self.barLayout.setObjectName(u"barLayout")
        self.barLayout.addWidget(self.btn_regi)
        self.barLayout.addWidget(self.btn_ct)

        #页面
        self.appStack = QStackedLayout(self.mainWidget)
        self.appStack.setObjectName(u"appStack")

        # 页面一：配准页面
        self.centralwidget = QWidget()
        self.centralwidget.setObjectName(u"centralwidget")

        #rxFixedLabel
        self.rxFixedLabelLayout = QVBoxLayout()
        self.rxFixedLabelLayout.setObjectName(u"rxFixedLabelLayout")
        self.rxlabel = QLabel(self.centralwidget)
        self.rxlabel.setObjectName(u"rxlabel")
        #rxLabel加入垂直布局
        self.rxFixedLabelLayout.addWidget(self.rxlabel, 0, Qt.AlignHCenter)
        self.rxFixedValue = QDoubleSpinBox(self.centralwidget)
        self.rxFixedValue.setObjectName(u"rxFixedValue")
        self.rxFixedValue.setMinimum(-98.000000000000000)
        self.rxFixedValue.setMaximum(-82.000000000000000)
        self.rxFixedValue.setValue(-90.000000000000000)
        #rxFixedValue加入垂直布局
        self.rxFixedLabelLayout.addWidget(self.rxFixedValue)

        #ryFixedLabel
        self.ryFixedLabelLayout = QVBoxLayout()
        self.ryFixedLabelLayout.setObjectName(u"ryFixedLabelLayout")
        self.rylabel = QLabel(self.centralwidget)
        self.rylabel.setObjectName(u"rylabel")
        #ryLabel加入垂直布局
        self.ryFixedLabelLayout.addWidget(self.rylabel, 0, Qt.AlignHCenter)
        self.ryFixedValue = QDoubleSpinBox(self.centralwidget)
        self.ryFixedValue.setObjectName(u"ryFixedValue")
        self.ryFixedValue.setMinimum(-8.000000000000000)
        self.ryFixedValue.setMaximum(8.000000000000000)
        self.ryFixedValue.setValue(0.000000000000000)
        #ryFixedValue加入垂直布局
        self.ryFixedLabelLayout.addWidget(self.ryFixedValue)

        #rzFixedLabel
        self.rzFixedLabelLayout = QVBoxLayout()
        self.rzFixedLabelLayout.setObjectName(u"rzFixedLabelLayout")
        self.rzlabel = QLabel(self.centralwidget)
        self.rzlabel.setObjectName(u"rzlabel")
        #rzLabel加入垂直布局
        self.rzFixedLabelLayout.addWidget(self.rzlabel, 0, Qt.AlignHCenter)
        self.rzFixedValue = QDoubleSpinBox(self.centralwidget)
        self.rzFixedValue.setObjectName(u"rzFixedValue")
        self.rzFixedValue.setMinimum(-4.000000000000000)
        self.rzFixedValue.setMaximum(4.000000000000000)
        self.rzFixedValue.setValue(0.000000000000000)
        #rzFixedValue加入垂直布局
        self.rzFixedLabelLayout.addWidget(self.rzFixedValue)

        #txFixedLabel
        self.txFixedLabelLayout = QVBoxLayout()
        self.txFixedLabelLayout.setObjectName(u"txFixedLabelLayout")
        self.txlabel = QLabel(self.centralwidget)
        self.txlabel.setObjectName(u"txlabel")
        #txLabel加入垂直布局
        self.txFixedLabelLayout.addWidget(self.txlabel, 0, Qt.AlignHCenter)
        self.txFixedValue = QDoubleSpinBox(self.centralwidget)
        self.txFixedValue.setObjectName(u"txFixedValue")
        self.txFixedValue.setMinimum(-20.000000000000000)
        self.txFixedValue.setMaximum(20.000000000000000)
        self.txFixedValue.setValue(0.000000000000000)
        #txFixedValue加入垂直布局
        self.txFixedLabelLayout.addWidget(self.txFixedValue)

        #tyFixedLabel
        self.tyFixedLabelLayout = QVBoxLayout()
        self.tyFixedLabelLayout.setObjectName(u"tyFixedLabelLayout")
        self.tylabel = QLabel(self.centralwidget)
        self.tylabel.setObjectName(u"tylabel")
        #tyLabel加入垂直布局
        self.tyFixedLabelLayout.addWidget(self.tylabel, 0, Qt.AlignHCenter)
        self.tyFixedValue = QDoubleSpinBox(self.centralwidget)
        self.tyFixedValue.setObjectName(u"tyFixedValue")
        self.tyFixedValue.setMinimum(-8.000000000000000)
        self.tyFixedValue.setMaximum(8.000000000000000)
        self.tyFixedValue.setValue(0.000000000000000)
        #tyFixedValue加入垂直布局
        self.tyFixedLabelLayout.addWidget(self.tyFixedValue)

        #tzFixedLabel
        self.tzFixedLabelLayout = QVBoxLayout()
        self.tzFixedLabelLayout.setObjectName(u"tzFixedLabelLayout")
        self.tzlabel = QLabel(self.centralwidget)
        self.tzlabel.setObjectName(u"tzlabel")
        #tzLabel加入垂直布局
        self.tzFixedLabelLayout.addWidget(self.tzlabel, 0, Qt.AlignHCenter)
        self.tzFixedValue = QDoubleSpinBox(self.centralwidget)
        self.tzFixedValue.setObjectName(u"tzFixedValue")
        self.tzFixedValue.setMinimum(-20.000000000000000)
        self.tzFixedValue.setMaximum(20.000000000000000)
        self.tzFixedValue.setValue(0.000000000000000)
        #tzFixedValue加入垂直布局
        self.tzFixedLabelLayout.addWidget(self.tzFixedValue)

        #inputChoose
        self.inputChooseLayout = QVBoxLayout()
        self.inputChooseLayout.setObjectName(u"inputChooseLayout")
        self.isInputLabel = QComboBox(self.centralwidget)
        self.isInputLabel.setObjectName(u"isInputLabel")
        self.isInputLabel.addItem("选择文件")
        self.isInputLabel.addItem("输入参数")
        #isInputLabel加入垂直布局
        self.inputChooseLayout.addWidget(self.isInputLabel)
        self.FileOpenButton = QPushButton(self.centralwidget)
        self.FileOpenButton.setObjectName(u"FileOpenButton")
        self.FileOpenButton.setStyleSheet(u"")
        icon = QIcon()
        icon.addFile(u"icons/openfile.png", QSize(), QIcon.Normal, QIcon.Off)
        self.FileOpenButton.setIcon(icon)
        self.FileOpenButton.setCheckable(False)
        self.FileOpenButton.setAutoRepeat(False)
        #FileOpenButton加入垂直布局
        self.inputChooseLayout.addWidget(self.FileOpenButton)

        #LoadandStart
        self.LoadandStartLayout = QVBoxLayout()
        self.LoadandStartLayout.setObjectName(u"LoadandStartLayout")
        self.LoadImage = QPushButton(self.centralwidget)
        self.LoadImage.setObjectName(u"LoadImage")
        icon1 = QIcon()
        icon1.addFile(u"icons/loadimage.png", QSize(), QIcon.Normal, QIcon.Off)
        self.LoadImage.setIcon(icon1)
        #LoadImage加入垂直布局
        self.LoadandStartLayout.addWidget(self.LoadImage)

        self.StartRegistrationButton = QPushButton(self.centralwidget)
        self.StartRegistrationButton.setObjectName(u"StartRegistrationButton")
        icon2 = QIcon()
        icon2.addFile(u"icons/registration.png", QSize(), QIcon.Normal, QIcon.Off)
        self.StartRegistrationButton.setIcon(icon2)
        #StartRegistrationButton加入垂直布局
        self.LoadandStartLayout.addWidget(self.StartRegistrationButton)

        #RefineandAnalysis
        self.RefineandAnalysisLayout = QVBoxLayout()
        self.RefineandAnalysisLayout.setObjectName(u"RefineandAnalysisLayout")

        self.RefineButton = QPushButton(self.centralwidget)
        self.RefineButton.setObjectName(u"RefineButton")
        icon3 = QIcon()
        icon3.addFile(u"icons/refine.png", QSize(), QIcon.Normal, QIcon.Off)
        self.RefineButton.setIcon(icon3)
        #RefineButton加入垂直布局
        self.RefineandAnalysisLayout.addWidget(self.RefineButton)

        self.AnalysisButton = QPushButton(self.centralwidget)
        self.AnalysisButton.setObjectName(u"AnalysisButton")
        icon4 = QIcon()
        icon4.addFile(u"icons/analysis.png", QSize(), QIcon.Normal, QIcon.Off)
        self.AnalysisButton.setIcon(icon4)
        #AnalysisButton加入垂直布局
        self.RefineandAnalysisLayout.addWidget(self.AnalysisButton)


        #ImageLoadTop
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


        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setLineWidth(2)
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        #ImageShowMiddle
        self.Fixedlabel = QLabel(self.centralwidget)
        self.Fixedlabel.setObjectName(u"Fixedlabel")
        font = QFont()
        font.setBold(True)
        self.Fixedlabel.setFont(font)
        self.Fixedlabel.setAlignment(Qt.AlignCenter)


        self.FixedImage = QLabel(self.centralwidget)
        self.FixedImage.setObjectName(u"FixedImage")


        self.Movedlabel = QLabel(self.centralwidget)
        self.Movedlabel.setObjectName(u"Movedlabel")
        self.Movedlabel.setFont(font)
        self.Movedlabel.setAlignment(Qt.AlignCenter)

        self.MovedImage = QLabel(self.centralwidget)
        self.MovedImage.setObjectName(u"MovedImage")

        self.Differencelabel = QLabel(self.centralwidget)
        self.Differencelabel.setObjectName(u"Differencelabel")
        self.Differencelabel.setFont(font)
        self.Differencelabel.setAlignment(Qt.AlignCenter)

        self.DifferenceImage = QLabel(self.centralwidget)
        self.DifferenceImage.setObjectName(u"DifferenceImage")

        # self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)



        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setLineWidth(2)
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        #RegistrationResultBottom
        self.RegistrationResultlabel = QLabel(self.centralwidget)
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

        self.RegistrationValuelabel = QLabel(self.centralwidget)
        self.RegistrationValuelabel.setObjectName(u"RegistrationValuelabel")

        #rxMovedLayout
        self.rxMovedLayout = QHBoxLayout()
        self.rxMovedLayout.setObjectName(u"rxMovedLayout")
        self.rxlabel_2 = QLabel(self.centralwidget)
        self.rxlabel_2.setObjectName(u"rxlabel_2")
        self.rxMovedLayout.addWidget(self.rxlabel_2)
        self.rxMovedValue = QLineEdit(self.centralwidget)
        self.rxMovedValue.setObjectName(u"rxMovedValue")
        self.rxMovedLayout.addWidget(self.rxMovedValue)
        #ryMovedLayout
        self.ryMovedLayout = QHBoxLayout()
        self.rylabel_2 = QLabel(self.centralwidget)
        self.rylabel_2.setObjectName(u"rylabel_2")
        self.ryMovedLayout.addWidget(self.rylabel_2)
        self.ryMovedValue = QLineEdit(self.centralwidget)
        self.ryMovedValue.setObjectName(u"ryMovedValue")
        self.ryMovedLayout.addWidget(self.ryMovedValue)
        #rzMovedLayout
        self.rzMovedLayout = QHBoxLayout()
        self.rzMovedLayout.setObjectName(u"rzMovedLayout")
        self.rzlabel_2 = QLabel(self.centralwidget)
        self.rzlabel_2.setObjectName(u"rzlabel_2")
        self.rzMovedLayout.addWidget(self.rzlabel_2)
        self.rzMovedValue = QLineEdit(self.centralwidget)
        self.rzMovedValue.setObjectName(u"rzMovedValue")
        self.rzMovedLayout.addWidget(self.rzMovedValue)
        #txMovedLayout
        self.txMovedLayout = QHBoxLayout()
        self.txMovedLayout.setObjectName(u"txMovedLayout")
        self.txlabel_3 = QLabel(self.centralwidget)
        self.txlabel_3.setObjectName(u"txlabel_3")
        self.txMovedLayout.addWidget(self.txlabel_3)
        self.txMovedValue = QLineEdit(self.centralwidget)
        self.txMovedValue.setObjectName(u"txMovedValue")
        self.txMovedLayout.addWidget(self.txMovedValue)
        #tyMovedLayout
        self.tyMovedLayout = QHBoxLayout()
        self.tyMovedLayout.setObjectName(u"tyMovedLayout")
        self.tylabel_2 = QLabel(self.centralwidget)
        self.tylabel_2.setObjectName(u"tylabel_2")
        self.tyMovedLayout.addWidget(self.tylabel_2)
        self.tyMovedValue = QLineEdit(self.centralwidget)
        self.tyMovedValue.setObjectName(u"tyMovedValue")
        self.tyMovedLayout.addWidget(self.tyMovedValue)
        #tzMovedLayout
        self.tzMovedLayout = QHBoxLayout()
        self.tzMovedLayout.setObjectName(u"tzMovedLayout")
        self.tzlabel_2 = QLabel(self.centralwidget)
        self.tzlabel_2.setObjectName(u"tzlabel_2")
        self.tzMovedLayout.addWidget(self.tzlabel_2)
        self.tzMovedValue = QLineEdit(self.centralwidget)
        self.tzMovedValue.setObjectName(u"tzMovedValue")
        self.tzMovedLayout.addWidget(self.tzMovedValue)

        #MovedValueLayout
        self.MovedValueLayout = QHBoxLayout()
        self.MovedValueLayout.setObjectName(u"MovedValueLayout")
        self.MovedValueLayout.addLayout(self.rxMovedLayout)
        self.MovedValueLayout.addLayout(self.ryMovedLayout)
        self.MovedValueLayout.addLayout(self.rzMovedLayout)
        self.MovedValueLayout.addLayout(self.txMovedLayout)
        self.MovedValueLayout.addLayout(self.tyMovedLayout)
        self.MovedValueLayout.addLayout(self.tzMovedLayout)


        self.Evaluatelabel = QLabel(self.centralwidget)
        self.Evaluatelabel.setObjectName(u"Evaluatelabel")

        #NCCLayout
        self.NCCLayout = QHBoxLayout()
        self.NCCLayout.setObjectName(u"NCCLayout")
        self.NCClabel = QLabel(self.centralwidget)
        self.NCClabel.setObjectName(u"NCClabel")
        self.NCCLayout.addWidget(self.NCClabel)
        self.NCCValue = QLineEdit(self.centralwidget)
        self.NCCValue.setObjectName(u"NCCValue")
        self.NCCLayout.addWidget(self.NCCValue)

        #MILayout
        self.MILayout = QHBoxLayout()
        self.MILayout.setObjectName(u"MILayout")
        self.NMIlabel = QLabel(self.centralwidget)
        self.NMIlabel.setObjectName(u"NMIlabel")
        self.MILayout.addWidget(self.NMIlabel)
        self.NMIValue = QLineEdit(self.centralwidget)
        self.NMIValue.setObjectName(u"NMIValue")
        self.MILayout.addWidget(self.NMIValue)

        #DICELayout
        self.DICELayout = QHBoxLayout()
        self.DICELayout.setObjectName(u"DICELayout")
        self.DICElabel = QLabel(self.centralwidget)
        self.DICElabel.setObjectName(u"DICElabel")
        self.DICELayout.addWidget(self.DICElabel)
        self.DICECValue = QLineEdit(self.centralwidget)
        self.DICECValue.setObjectName(u"DICECValue")
        self.DICELayout.addWidget(self.DICECValue)

        #EvaluateLayout
        self.EvaluateLayout = QHBoxLayout()
        self.EvaluateLayout.setObjectName(u"EvaluateLayout")
        self.EvaluateLayout.addLayout(self.NCCLayout)
        self.EvaluateLayout.addLayout(self.MILayout)
        self.EvaluateLayout.addLayout(self.DICELayout)

        #整体布局
        self.gridLayout = QGridLayout(self.centralwidget)
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

        #展示图片
        funcs.showImage(self.FixedImage, "img/FixedImageInit.png")
        funcs.showImage(self.MovedImage, "img/MovedImageInit.png")
        funcs.showImage(self.DifferenceImage, "img/DifferenceImageInit.png")

        self.rxFixedValue.setEnabled(False)
        self.ryFixedValue.setEnabled(False)
        self.rzFixedValue.setEnabled(False)
        self.txFixedValue.setEnabled(False)
        self.tyFixedValue.setEnabled(False)
        self.tzFixedValue.setEnabled(False)

        # 页面2：CT
        self.ct = QWidget()
        self.ct.setObjectName(u"ct")
        self.ctlabel = QLabel(self.ct)
        self.ctlabel.setObjectName(u"ctlabel")
        self.ctlabel.setText("CT")

        self.appLayout.addWidget(self.leftbar)
        self.appLayout.addWidget(self.appStack)
        self.appLayout.setStretch(0, 1)
        self.appLayout.setStretch(1, 8)










        # img_bgr = cv2.imread("img/FixedImageInit.png")
        # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # qt_img = QImage(img_rgb.data, #数据源
        #                 img_rgb.shape[1], #宽度
        #                 img_rgb.shape[0], #高度
        #                 # img_rgb.shape[1] * 3, #行字节数
        #                 QImage.Format_RGB888)
        #
        # pix_img = QPixmap.fromImage(qt_img).scaled(400, 400, aspectMode=Qt.KeepAspectRatio)
        # self.FixedImage.setScaledContents(True)
        # self.FixedImage.setPixmap(pix_img)

        # img_obj = ImageQt.fromqpixmap(qt_img)
        # plt.imshow(img_obj)
        # plt.show()


        MainWindow.setCentralWidget(self.mainWidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 834, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"2D/3D Registration", None))
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
    # retranslateUi


