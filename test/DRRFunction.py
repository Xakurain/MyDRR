import itk
import math
__all__ = [itk]

import SimpleITK as sitK
import numpy as np

class MyDRRFunction1:

    def __init__(self, input_dcm_name, dimension, verbose) :
        self.Input_name = input_dcm_name
        self.verbose = verbose
        self.Dimension = dimension
        self.PixelType = itk.UC
        self.InputPixelType = itk.SS
        self.InputImageType = itk.Image[self.InputPixelType, self.Dimension]     


    def read_dicom(self):
        image = self.InputImageType.New()
        ReaderType = itk.ImageSeriesReader[self.InputImageType]
        reader = ReaderType.New()

        gdcmIO = itk.GDCMImageIO.New()
        reader.SetImageIO(gdcmIO)

        nameGenerator = itk.GDCMSeriesFileNames.New()
        nameGenerator.SetInputDirectory(self.Input_name)

        files = nameGenerator.GetInputFileNames()
        reader.SetFileNames(files)
        reader.Update()
        image = reader.GetOutput()
        imOrigin = image.GetOrigin()
        imRes = image.GetSpacing()
        imRegion = image.GetBufferedRegion()
        imSize = imRegion.GetSize()
        print(f"Image Size: {imSize}")
        print(f"Resolution: {imRes}")
        print(f"origin: {imOrigin}")

        return image

    def drr(self, image, Drr_info):
        rx = Drr_info['rx']
        ry = Drr_info['ry']
        rz = Drr_info['rz']

        tx = Drr_info['tx']
        ty = Drr_info['ty']
        tz = Drr_info['tz']

        cx = 0.
        cy = 0.
        cz = 0.

        sid = Drr_info['sid']
        sx = Drr_info['sx']
        sy = Drr_info['sy']

        dx = Drr_info['dx']
        dy = Drr_info['dy']

        o2Dx = 0
        o2Dy = 0
        threshold = 200

        FilterType = itk.ResampleImageFilter[self.InputImageType, self.InputImageType]
        filter = FilterType.New()
        # print(type(image))
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

        if self.verbose:
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
        if self.verbose:
            print(f"Focal Point: {focalpoint[0]}, {focalpoint[1]}, {focalpoint[2]}")
            print(interpolator)
        filter.SetInterpolator(interpolator)
        filter.SetTransform(transform)
        size = itk.Size[self.Dimension]()
        size[0] = int(dx)
        size[1] = int(dy)
        size[2] = 1
        filter.SetSize(size)
        spacing = [sx, sy, 1.0]
        filter.SetOutputSpacing(spacing)

        if self.verbose:
            print(f"Output Image Size: {size[0]}, {size[1]}, {size[2]}")
            print(f"Output Image Spacing: {spacing[0]}, {spacing[1]}, {spacing[2]}")

        origin = []
        origin.append(imOrigin[0] + o2Dx - sx * ((dx - 1.0) / 2.0))
        origin.append(imOrigin[1] + o2Dy - sy * ((dy - 1.0) / 2.0))
        origin.append(imOrigin[2] + sid / 2.0)
        filter.SetOutputOrigin(origin)
        if self.verbose:
            print(f"Output Image Origin: {origin}")
        return filter






# input_name = "F:\\dataset\\imia\\zyt303\\ScalarVolume_16"



