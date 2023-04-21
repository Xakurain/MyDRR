#!/usr/bin/env python
# coding: utf-8
import itk
import math
__all__ = [itk]

import SimpleITK as sitK
import numpy as np
import cv2
import os
def convert_from_dicom_to_png(img,low_window,high_window,save_path):
    lungwin = np.array([low_window*1.,high_window * 1.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])  #归一化
    newimg = (newimg*255).astype('uint8')  #扩展像素值到【0，255】
    # print(newimg.shape)
    cropped = newimg[0:511, 512:1023]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite(save_path,cropped)
    


# input_name = "F:\\dataset\\imia\\zyt303_1\\303 Calf_0.nii.gz"
input_name = "F:\\dataset\\imia\\zyt303\\ScalarVolume_16"
output_name = "1.png"
ok = False
verbose = True
rx = -90.0
ry = 0.
rz = 0.

tx = 0.
ty = 0.
tz = 0.

cx = 0.
cy = 0.
cz = 0.

sid = 400.
sx = 1.25
sy = 1.25


dx = 1024.
dy = 1024.

o2Dx = 0
o2Dy = 0
threshold = 0

# parse arg
# TODO

Dimension = 3

PixelType = itk.UC
InputPixelType = itk.SS
OutputPixelType = itk.UC
InputImageType = itk.Image[InputPixelType, Dimension]
OutputImageType = itk.Image[OutputPixelType, Dimension]

image = InputImageType.New()
ReaderType = itk.ImageSeriesReader[InputImageType]
reader = ReaderType.New()

gdcmIO = itk.GDCMImageIO.New()
reader.SetImageIO(gdcmIO)

nameGenerator = itk.GDCMSeriesFileNames.New()
nameGenerator.SetInputDirectory(input_name)

files = nameGenerator.GetInputFileNames()
reader.SetFileNames(files)
reader.Update()
image = reader.GetOutput()


if verbose:
    print("Input: ")
    print(f"{image.GetBufferedRegion()}")
    print(f" Resolution: {image.GetSpacing()}")
    print(f"Origin: {image.GetOrigin()}")

FilterType = itk.ResampleImageFilter[InputImageType, InputImageType]
filter = FilterType.New()
print(type(image))
filter.SetInput(image)
filter.SetDefaultPixelValue(0)
TransforType = itk.CenteredEuler3DTransform[itk.D]
transform = TransforType.New()
transform.SetComputeZYX(True)
translation = [tx, ty, tz]

dtr = (math.atan(1.0) * 4.0) / 180.0
transform.SetTranslation(translation)
transform.SetRotation(dtr * rx, dtr * ry, dtr * rz)
imOrigin = image.GetOrigin()
imRes = image.GetSpacing()

imRegion = image.GetBufferedRegion()
imSize = imRegion.GetSize()
imOrigin[0] += imRes[0] * imSize[0] / 2.0
imOrigin[1] += imRes[1] * imSize[1] / 2.0
imOrigin[2] += imRes[2] * imSize[2] / 2.0

center = [cx + imOrigin[0], cy + imOrigin[1], cz + imOrigin[2]]
transform.SetCenter(center)

if verbose:
    print(f"Image Size: {imSize}")
    print(f"Resolution: {imRes}")
    print(f"origin: {imOrigin}")
    print(f"center: {center}")
    print(f"Transform: {transform}")

InterpolatorType = itk.itkRayCastInterpolateImageFunctionPython.itkRayCastInterpolateImageFunctionISS3D
interpolator = InterpolatorType.New()
interpolator.SetTransform(transform)
interpolator.SetThreshold(threshold)
focalpoint = [imOrigin[0], imOrigin[1], imOrigin[2] - sid / 2.]
interpolator.SetFocalPoint(focalpoint)
if verbose:
    print(f"Focal Point: {focalpoint[0]}, {focalpoint[1]}, {focalpoint[2]}")
print(interpolator)
filter.SetInterpolator(interpolator)
filter.SetTransform(transform)
size = itk.Size[Dimension]()
size[0] = int(dx)
size[1] = int(dy)
size[2] = 1
filter.SetSize(size)
spacing = [sx, sy, 1.0]
filter.SetOutputSpacing(spacing)

if verbose:
    print(f"Output Image Size: {size[0]}, {size[1]}, {size[2]}")
    print(f"Output Image Spacing: {spacing[0]}, {spacing[1]}, {spacing[2]}")

origin = []
origin.append(imOrigin[0] + o2Dx - sx * ((dx - 1.0) / 2.0))
origin.append(imOrigin[1] + o2Dy - sy * ((dy - 1.0) / 2.0))
origin.append(imOrigin[2] + sid / 2.0)
filter.SetOutputOrigin(origin)
if verbose:
    print(f"Output Image Origin: {origin}")

# image1 = itk.rescale_intensity_image_filter(
#     filter, output_minimum=0, output_maximum=255
# )

# print(image1)
itk.imwrite(filter, "F:\\code\\python\\iMIA\\MyDRR\\h.dcm")
ds_array = sitK.ReadImage("F:\\code\\python\\iMIA\\MyDRR\\h.dcm")
img_array = sitK.GetArrayFromImage(ds_array)
shape = img_array.shape
img_array = np.reshape(img_array,(shape[1],shape[2]))
high = np.max(img_array)
low = np.min(img_array)
convert_from_dicom_to_png(img_array,low,high,"F:\\code\\python\\iMIA\\MyDRR\\i.png")





# RescaleFilterType = itk.itkRescaleIntensityImageFilterPython.itkRescaleIntensityImageFilterISS3ISS3
# rescaler = RescaleFilterType.New()
# rescaler.SetOutputMinimum(0)
# rescaler.SetOutputMaximum(255)
# rescaler.SetInput(filter.GetOutput())
# WriterType = itk.ImageFileWriter[OutputImageType]
# writer = WriterType.New()
# itk.PNGImageIOFactory.RegisterOneFactory()
# writer.SetFileName("F:\\code\\python\\iMIA\\MyDRR\\a.png")
# writer.SetInput(filter.GetOutput())
# writer.Update()