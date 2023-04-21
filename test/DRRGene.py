import itk
import SimpleITK as sitK
import numpy as np
import cv2
import sys
__all__ = [itk]

from DRRFunction import MyDRRFunction1


def convert_from_dicom_to_png(img,low_window,high_window,save_path):
    lungwin = np.array([low_window*1.,high_window * 1.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])  #归一化
    newimg = (newimg*255).astype('uint8')  #扩展像素值到【0，255】
    # print(newimg.shape)
    # cropped = newimg[0:511, 512:1023]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite(save_path,newimg)

def save_png(filter1, save_dcm_name, save_png_name):
    itk.imwrite(filter1, save_dcm_name)
    ds_array = sitK.ReadImage(save_dcm_name)
    img_array = sitK.GetArrayFromImage(ds_array)
    shape = img_array.shape
    img_array = np.reshape(img_array,(shape[1],shape[2]))
    high = np.max(img_array)
    low = np.min(img_array)
    convert_from_dicom_to_png(img_array,low,high, save_png_name)

input_dcm_name = "F:\\dataset\\imia\\zyt303\\ScalarVolume_16"
output_root = "F:\\dataset\\imia\\zyt303\\drr\\"
dimension = 3
verbose = False 
Drr_info = { 'rx': -90.0,
             'ry':  0.,
             'rz':  0.,
             'tx': 0.,
             'ty': 0.,
             'tz':  0.,
             'sid':  400.,
             'sx': 1.25,
             'sy':  1.25,
             'dx':  1024.,
             'dy':  1024.}

Drr1 = MyDRRFunction1(input_dcm_name, dimension, verbose)
image1 = Drr1.read_dicom()

for rx in range(-90, 90):
    print(rx)
    save_dcm_name = f'{output_root}dcm\\rx_{rx}.dcm'
    save_png_name = f'{output_root}png\\rx_{rx}.png'
    Drr_info['rx'] = rx
    filter1 = Drr1.drr(image1, Drr_info)
    save_png(filter1, save_dcm_name, save_png_name)





