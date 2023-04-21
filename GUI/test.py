#coding=utf-8
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

import SimpleITK as sitk
import numpy as np
import os
import vtkmodules.all as vtk
from modules import ui_main, funcs
import qdarktheme

import sys
import datetime
sys.path.append('../wrapped_modules')
from DRRGenerate import pyDRRGenerate

envpath = 'D:\\Python\\Anaconda3\\Lib\\site-packages\\PySide6\\plugins\\platforms'
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = envpath


class Worker(QObject):
    DRRFinishSignal = Signal(str)

    def DrrInit(self, dcmPath):
        print('DrrInit')
        self.p = pyDRRGenerate(dcmPath, False)
        self.p.ReadDCM()

    def RegistrationInit(self):
        self.net, self.device = funcs.predictinit()

    def GenerateDRR(self, rx, ry, rz, tx, ty, tz, path):
        sid = 400
        sx = 3
        sy = 3
        dx = 512
        dy = 512
        threshold = 0
        self.p.Drr1(path, rx, ry, rz, tx, ty, tz, sid, sx, sy, dx, dy, threshold)
        self.DRRFinishSignal.emit(path)

class WorkerforRegstration(QObject):
    over = Signal(dict)

    def Registration(self, img_path, net, device):
        print('predict')
        movedlabels = funcs.predict(net, device, img_path)
        self.over.emit(movedlabels)

class MyWindow(QMainWindow):
    DrrInitSignal = Signal(str)
    RegistrationInitSignal = Signal()
    GenerateDRRSignal = Signal(float, float, float, float, float, float, str)
    RegistrationSignal = Signal(str, object, object)

    def __init__(self):
        QMainWindow.__init__(self)

        # SET AS GLOBAL WIDGETS
        # ///////////////////////////////////////////////////////////////
        self.ui = ui_main.Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        #创建一个线程和一个工作实例
        self.worker = Worker()
        self.work_thread = QThread()
        self.worker1 = WorkerforRegstration()
        self.work_thread1 = QThread()

        #全局信号绑定工作实例
        self.DrrInitSignal.connect(self.worker.DrrInit)
        self.RegistrationInitSignal.connect(self.worker.RegistrationInit)
        self.GenerateDRRSignal.connect(self.worker.GenerateDRR)
        self.RegistrationSignal.connect(self.worker1.Registration)

        self.worker.DRRFinishSignal.connect(self.showMovedImage)
        self.worker1.over.connect(self.stopRegistration)

        #工作实例绑定线程
        self.worker.moveToThread(self.work_thread)
        self.worker1.moveToThread(self.work_thread1)

        widgets.isInputLabel.activated.connect(self.buttonClick)
        widgets.FileOpenButton.clicked.connect(self.buttonClick)
        widgets.LoadImage.clicked.connect(self.buttonClick)
        widgets.StartRegistrationButton.clicked.connect(self.buttonClick)
        widgets.AnalysisButton.clicked.connect(self.buttonClick)
        widgets.btnRegi.clicked.connect(self.buttonClick1)
        widgets.btnCT.clicked.connect(self.buttonClick1)
        widgets.btnOpenCT.clicked.connect(self.buttonClick1)
        widgets.btnShow3D.clicked.connect(self.buttonClick1)
        widgets.btnShowInfo.clicked.connect(self.buttonClick1)

        self.work_thread.start()
        self.threadStart()

    #加载dcm数据
    def threadStart(self):
        dcmPath = "F:\\dataset\\imia\\zyt303\\ScalarVolume_16"
        self.DrrInitSignal.emit(dcmPath)
        self.RegistrationInitSignal.emit()

    def buttonClick(self):
        btn = self.sender()
        btnName = btn.objectName()

        if btnName == 'isInputLabel':
            if widgets.isInputLabel.currentText() == '输入参数':
                widgets.FileOpenButton.setText('生成图像')
                widgets.rxFixedValue.setEnabled(True)
                widgets.ryFixedValue.setEnabled(True)
                widgets.rzFixedValue.setEnabled(True)
                widgets.txFixedValue.setEnabled(True)
                widgets.tyFixedValue.setEnabled(True)
                widgets.tzFixedValue.setEnabled(True)
            else:
                widgets.FileOpenButton.setText('打开文件')
                widgets.rxFixedValue.setEnabled(False)
                widgets.ryFixedValue.setEnabled(False)
                widgets.rzFixedValue.setEnabled(False)
                widgets.txFixedValue.setEnabled(False)
                widgets.tyFixedValue.setEnabled(False)
                widgets.tzFixedValue.setEnabled(False)

        elif btnName == 'FileOpenButton':
            if widgets.isInputLabel.currentText() == '输入参数':
                rx = float(widgets.rxFixedValue.text())
                ry = float(widgets.ryFixedValue.text())
                rz = float(widgets.rzFixedValue.text())
                tx = float(widgets.txFixedValue.text())
                ty = float(widgets.tyFixedValue.text())
                tz = float(widgets.tzFixedValue.text())
                curtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                self.save_dcm_name = f'F:\\code\\python\\iMIA\\MyDRR\\GUI\\img\\{curtime}.png'
                self.GenerateDRRSignal.emit(rx, ry, rz, tx, ty, tz, self.save_dcm_name)
            else:
                FilePath = QFileDialog.getOpenFileName(self, '选择文件', './', 'Image Files(*.jpg *.png *.bmp *.tif *.tiff *.dcm *.dicom *.nii *.nii.gz *.mhd *.mha)')
                if FilePath[0] != '':
                    self.FixedImagePath = FilePath[0]
        elif btnName == 'LoadImage':
            if widgets.isInputLabel.currentText() == '输入参数':
                self.nowshowimg = self.save_dcm_name
                funcs.showImage(widgets.FixedImage, self.save_dcm_name)
            else:
                self.nowshowimg = self.FixedImagePath
                funcs.showImage(widgets.FixedImage, self.FixedImagePath)

        elif btnName == 'StartRegistrationButton':
            self.work_thread1.start()
            self.RegistrationSignal.emit(self.nowshowimg, self.worker.net, self.worker.device)

        elif btnName == 'AnalysisButton':
            funcs.showImage(widgets.DifferenceImage, [self.nowshowimg, self.MovedImagePath], mode='difference')
            d, ncc, nmi = funcs.Analysis(self.nowshowimg, self.MovedImagePath)
            widgets.DICECValue.setText(str(d))
            widgets.NCCValue.setText(str(ncc))
            widgets.NMIValue.setText(str(nmi))
    def stopRegistration(self, movedlabels):
        print(movedlabels)
        self.work_thread1.quit()
        self.work_thread1.wait()
        widgets.rxMovedValue.setText(str(movedlabels['rx']))
        widgets.ryMovedValue.setText(str(movedlabels['ry']))
        widgets.rzMovedValue.setText(str(movedlabels['rz']))
        widgets.txMovedValue.setText(str(movedlabels['tx']))
        widgets.tyMovedValue.setText(str(movedlabels['ty']))
        widgets.tzMovedValue.setText(str(movedlabels['tz']))
        curtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.MovedImagePath = f'F:\\code\\python\\iMIA\\MyDRR\\GUI\\img\\{curtime}_moved.png'
        rx = float(movedlabels['rx'])
        ry = float(movedlabels['ry'])
        rz = float(movedlabels['rz'])
        tx = float(movedlabels['tx'])
        ty = float(movedlabels['ty'])
        tz = float(movedlabels['tz'])
        self.GenerateDRRSignal.emit(rx, ry, rz, tx, ty, tz, self.MovedImagePath)
    def showMovedImage(self, movedImagePath):
        if movedImagePath[-9:-4] == 'moved':
            funcs.showImage(widgets.MovedImage, movedImagePath)
    def buttonClick1(self):
        btn = self.sender()
        btnName = btn.objectName()
        if btnName == 'btnRegi':
            widgets.stackedWidget.setCurrentIndex(0)
        elif btnName == 'btnCT':
            widgets.stackedWidget.setCurrentIndex(1)
        elif btnName == 'btnOpenCT':
            self.CTPath = QFileDialog.getExistingDirectory(self, '选择文件夹', './')
            print(self.CTPath)
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(self.CTPath)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            print(image.GetSize())
            print(image.GetSpacing())
            image_array = sitk.GetArrayFromImage(image)
            print(image_array.shape)

            widgets.pg_tra.setImage(np.rot90(image_array, k=1, axes=(1, 2)))

            sag = image_array.transpose(2, 0, 1)
            sag = np.rot90(sag, k=3, axes=(1, 2))
            widgets.pg_sag.setImage(sag)

            cor = image_array.transpose(1, 0, 2)
            cor = np.rot90(cor, k=3, axes=(1, 2))
            widgets.pg_cor.setImage(cor)

            self.getInfo(dicom_names[0], image.GetSize(), image.GetSpacing(), image.GetOrigin())

        elif btnName == 'btnShow3D':
            self.show3D()
        elif btnName == 'btnShowInfo':
            self.showInfo()


    def getInfo(self, series0, size, spacing, origin):
        self.dataSize = size
        self.dataSpacing = spacing
        self.dataOrigin = origin
        self.dataName = sitk.ReadImage(series0).GetMetaData('0010|0010')      #姓名
        self.dataSex = sitk.ReadImage(series0).GetMetaData('0010|0040')       #性别
        self.dataID = sitk.ReadImage(series0).GetMetaData('0010|0020')        #ID
        self.dataModality = sitk.ReadImage(series0).GetMetaData('0008|0060')  #模态
    def showInfo(self):

        widgets.ListWidget.item(1).setText(self.dataName)
        widgets.ListWidget.item(3).setText(self.dataSex)
        widgets.ListWidget.item(5).setText(self.dataID)
        widgets.ListWidget.item(7).setText(self.dataModality)
        widgets.ListWidget.item(9).setText(str(self.dataSize[0]) + ' * ' + str(self.dataSize[1]) + ' * ' + str(self.dataSize[2]))
        widgets.ListWidget.item(11).setText(str(self.dataSpacing[0]) + ' * ' + str(self.dataSpacing[1]) + ' * ' + str(self.dataSpacing[2]))
        widgets.ListWidget.item(13).setText(str(self.dataOrigin))


    #数据源（ sourse )﹣映射器（ mapper )﹣演员（ actor )﹣渲染器（ renderer )﹣窗口（ renderwindow )﹣交互器（ RenderWindowInteractor )
    def show3D(self):
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(self.CTPath)

        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()             #GPU体绘制映射器
        volumeMapper.SetInputConnection(reader.GetOutputPort())    #设置输入数据

        volumeColor = vtk.vtkColorTransferFunction()              #颜色映射函数
        volumeColor.AddRGBPoint(0, 0.0, 0.0, 0.0)                  #设置颜色映射函数
        volumeColor.AddRGBPoint(500, 0.9, 0.5, 0.3)
        volumeColor.AddRGBPoint(1000, 0.9, 0.5, 0.3)
        volumeColor.AddRGBPoint(1150, 1.0, 1.0, 0.9)

        volumeScalarOpacity = vtk.vtkPiecewiseFunction()           #不透明度映射函数
        volumeScalarOpacity.AddPoint(0, 0.00)
        volumeScalarOpacity.AddPoint(500, 0.15)
        volumeScalarOpacity.AddPoint(1000, 0.15)
        volumeScalarOpacity.AddPoint(1150, 0.85)

        volumeproperty = vtk.vtkVolumeProperty()                  #体渲染属性
        volumeproperty.SetColor(volumeColor)                      #设置颜色映射函数
        volumeproperty.SetScalarOpacity(volumeScalarOpacity)      #设置不透明度映射函数
        volumeproperty.SetInterpolationTypeToLinear()             #设置插值类型
        volumeproperty.ShadeOn()                                  #打开阴影
        volumeproperty.SetAmbient(0.4)                            #设置环境光
        volumeproperty.SetDiffuse(0.6)                            #设置漫反射
        volumeproperty.SetSpecular(0.2)                           #设置镜面反射

        volume = vtk.vtkVolume()                                  #体渲染
        volume.SetMapper(volumeMapper)                            #设置映射器
        volume.SetProperty(volumeproperty)                        #设置体渲染属性

        renderer = vtk.vtkRenderer()                              #渲染器
        renderer.AddVolume(volume)                                #添加体渲染

        widgets.pg_3d.GetRenderWindow().AddRenderer(renderer)     #添加渲染器
        widgets.pg_3d.Initialize()                                #初始化
        widgets.pg_3d.Start()                                     #开始渲染


    #关闭窗口时关闭线程
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', "是否确定退出?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    #判断返回结果处理相应事项
        if reply == QMessageBox.Yes:
            self.work_thread.quit()
            self.work_thread.wait()
            self.work_thread1.quit()
            self.work_thread1.wait()
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication([])
    qdarktheme.setup_theme("light")

    window = MyWindow()
    window.show()
    app.exec()
