"""Module for generation of Digitally Reconstructed Radiographs (DRR).

This module includes classes for generation of DRRs from either a volumetric image (CT,MRI) 
or a STL model, and a projector class factory.

Classes:
    SiddonGpu: GPU accelerated (CUDA) DRR generation from CT or MRI scan.  
    Mahfouz: binary DRR generation from CAD model in STL format.

Functions:
    projector_factory: returns a projector instance.
    
New projectors can be plugged-in and added to the projector factory
as long as they are defined as classes with the following methods:
    compute: returns a 2D image (DRR) as a numpy array.
    delete: eventually deletes the projector object (only needed to deallocate memory from GPU) 
"""

####  PYTHON MODULES
import numpy as np
import time
import sys
import torch
####  Python ITK/VTK MODULES
import itk
# import cv2
# import vtk
# from vtk.util import numpy_support


import random


####  MY MODULES
import ReadWriteImageModule as rw
import RigidMotionModule as rm

sys.path.append('../wrapped_modules/')
from SiddonGpuPy import pySiddonGpu     # Python wrapped C library for GPU accelerated DRR generation



def projector_factory(projector_info,
                      movingImageFileName,
                      PixelType = itk.F,
                      Dimension = 3,
                      ScalarType = itk.D):

    """Generates instances of the specified projectors.

    Args:
        projector_info (dict of str): includes camera intrinsic parameters and projector-specific parameters
        movingImageFileName (string): cost function returning the metric value

    Returns:
        opt: instance of the specified projector class.
    """

    p = SiddonGpu(projector_info,
                    movingImageFileName,
                    PixelType,
                    Dimension,
                    ScalarType)

    return p



class SiddonGpu():

    """GPU accelearated DRR generation from volumetric image (CT or MRI scan).

       This class renders a DRR from a volumetric image, with an accelerated GPU algorithm
       from a Python wrapped library (SiddonGpuPy), written in C++ and accelerated with Cuda.
       IMplementation is based both on the description found in the “Improved Algorithm” section in Jacob’s paper (1998): 
       https://www.researchgate.net/publication/2344985_A_Fast_Algorithm_to_Calculate_the_Exact_Radiological_Path_Through_a_Pixel_Or_Voxel_Space
       and on the implementation suggested in Greef et al 2009:
       https://www.ncbi.nlm.nih.gov/pubmed/19810482

       Methods:
            compute (function): returns a 2D image (DRR) as a numpy array.
            delete (function): deletes the projector object (needed to deallocate memory from GPU)
    """


    def __init__(self, projector_info,
                       movingImageFileName,
                       PixelType,
                       Dimension,
                       ScalarType):

        """Reads the moving image and creates a siddon projector 
           based on the camera parameters provided in projector_info (dict)
        """

        # ITK: Instantiate types
        self.Dimension = Dimension
        self.ImageType = itk.Image[PixelType, Dimension]
        self.ImageType2D = itk.Image[PixelType, 2]
        self.RegionType = itk.ImageRegion[Dimension]
        PhyImageType = itk.Image[itk.Vector[itk.F, Dimension], Dimension]  # image of physical coordinates

        # Read moving image (CT or MRI scan)
        movImage, movImageInfo = rw.ImageReader(movingImageFileName, self.ImageType)
        self.movDirection = movImage.GetDirection()
        print(self.movDirection.GetInverse())

        # Calculate side planes
        X0 = movImageInfo['Volume_center'][0] - movImageInfo['Spacing'][0] * movImageInfo['Size'][0] * 0.5
        Y0 = movImageInfo['Volume_center'][1] - movImageInfo['Spacing'][1] * movImageInfo['Size'][1] / 2.0
        Z0 = movImageInfo['Volume_center'][2] - movImageInfo['Spacing'][2] * movImageInfo['Size'][2] / 2.0
        print(movImageInfo['Spacing'][0] * movImageInfo['Size'][0])
        print(movImageInfo['Spacing'][1] * movImageInfo['Size'][1])
        print(movImageInfo['Spacing'][2] * movImageInfo['Size'][2])
        # Get 1d array for moving image
        #movImgArray_1d = np.ravel(itk.PyBuffer[self.ImageType].GetArrayFromImage(movImage), order='C') # ravel does not generate a copy of the array (it is faster than flatten)
        movImgArray_1d = np.ravel(itk.GetArrayFromImage(movImage), order='C') # ravel does not generate a copy of the array (it is faster than flatten)

        # Set parameters for GPU library SiddonGpuPy
        NumThreadsPerBlock = np.array([projector_info['threadsPerBlock_x'], projector_info['threadsPerBlock_y'],
                                       projector_info['threadsPerBlock_z']]).astype(np.int32)
        DRRsize_forGpu = np.array([projector_info['DRRsize_x'], projector_info['DRRsize_y'], 1]).astype(np.int32)
        MovSize_forGpu = np.array([movImageInfo['Size'][0], movImageInfo['Size'][1], movImageInfo['Size'][2]]).astype(
            np.int32)
        MovSpacing_forGpu = np.array(
            [movImageInfo['Spacing'][0], movImageInfo['Spacing'][1], movImageInfo['Spacing'][2]]).astype(np.float32)

        # Define source point at its initial position (at the origin = moving image center)
        self.source = [0] * Dimension
        self.source[0] = movImageInfo['Volume_center'][0]
        self.source[1] = movImageInfo['Volume_center'][1]
        self.source[2] = movImageInfo['Volume_center'][2] - projector_info['focal_lenght'] / 2.

        # Define volume center
        self.center = [0] * Dimension
        self.center[0] = movImageInfo['Volume_center'][0]
        self.center[1] = movImageInfo['Volume_center'][1]
        self.center[2] = movImageInfo['Volume_center'][2]

        # Set DRR image at initial position (at +(focal length)/2 along the z direction)
        DRR = self.ImageType.New()
        self.DRRregion = self.RegionType()

        DRRstart = itk.Index[Dimension]()
        DRRstart.Fill(0)

        self.DRRsize = [0] * Dimension
        self.DRRsize[0] = projector_info['DRRsize_x']
        self.DRRsize[1] = projector_info['DRRsize_y']
        self.DRRsize[2] = 1

        self.DRRregion.SetSize(self.DRRsize)
        self.DRRregion.SetIndex(DRRstart)

        self.DRRspacing = itk.Point[itk.F, Dimension]()
        self.DRRspacing[0] = projector_info['DRRspacing_x']
        self.DRRspacing[1] = projector_info['DRRspacing_y']
        self.DRRspacing[2] = 1.

        self.DRRorigin = itk.Point[itk.F, Dimension]()
        self.DRRorigin[0] = movImageInfo['Volume_center'][0] - projector_info['DRR_ppx'] - self.DRRspacing[0] * (
                self.DRRsize[0] - 1.) / 2.
        self.DRRorigin[1] = movImageInfo['Volume_center'][1] - projector_info['DRR_ppy'] - self.DRRspacing[1] * (
                self.DRRsize[1] - 1.) / 2.
        self.DRRorigin[2] = movImageInfo['Volume_center'][2] + projector_info['focal_lenght'] / 2.

        DRR.SetRegions(self.DRRregion)
        DRR.Allocate()
        DRR.SetSpacing(self.DRRspacing)
        DRR.SetOrigin(self.DRRorigin)
        self.movDirection.SetIdentity()
        DRR.SetDirection(self.movDirection)

        # Get array of physical coordinates for the DRR at the initial position
        PhysicalPointImagefilter = itk.PhysicalPointImageSource[PhyImageType].New()
        PhysicalPointImagefilter.SetReferenceImage(DRR)
        PhysicalPointImagefilter.SetUseReferenceImage(True)
        PhysicalPointImagefilter.Update()
        sourceDRR = PhysicalPointImagefilter.GetOutput()

        # self.sourceDRR_array_to_reshape = itk.PyBuffer[PhyImageType].GetArrayFromImage(sourceDRR)[0].copy(order = 'C') # array has to be reshaped for matrix multiplication
        self.sourceDRR_array_to_reshape = itk.GetArrayFromImage(sourceDRR)[
            0]  # array has to be reshaped for matrix multiplication
        print(self.sourceDRR_array_to_reshape.shape)

        # Generate projector object
        tGpu1 = time.time()
        self.projector = pySiddonGpu(NumThreadsPerBlock,
                                     movImgArray_1d,
                                     MovSize_forGpu,
                                     MovSpacing_forGpu,
                                     X0.astype(np.float32), Y0.astype(np.float32), Z0.astype(np.float32),
                                     DRRsize_forGpu)
        tGpu2 = time.time()
        print('\nSiddon object initialized. Time elapsed for initialization: ', tGpu2 - tGpu1, '\n')

        # Get array of physical coordinates of the transformed DRR (GPU accelerated)
        Tn = np.array([[1., 0., 0., self.center[0]],
                       [0., 1., 0., self.center[1]],
                       [0., 0., 1., self.center[2]],
                       [0., 0., 0., 1.]])
        invTn = np.linalg.inv(Tn)
        sourceDRR_array_reshaped = self.sourceDRR_array_to_reshape.reshape(
            (self.DRRsize[0] * self.DRRsize[1], self.Dimension), order='C')
        sourceDRR_array_augmented = np.dot(invTn, rm.augment_matrix_coord(sourceDRR_array_reshaped))
        invT = np.zeros((4, 4))
        self.Tn = torch.FloatTensor(Tn).cuda()
        self.sourceDRR_array_augmented = torch.FloatTensor(sourceDRR_array_augmented).cuda()
        self.invT = torch.FloatTensor(invT).cuda()

    def compute(self, transform_parameters):


        """Generates a DRR given the transform parameters.

           Args:
               transform_parameters (list of floats): rotX, rotY,rotZ, transX, transY, transZ
 
        """
        # tDRR1 = time.time()
        tic = time.time()

        # Get transform parameters
        rotx = transform_parameters[0]
        roty = transform_parameters[1]
        rotz = transform_parameters[2]
        tx = transform_parameters[3]
        ty = transform_parameters[4]
        tz = transform_parameters[5]

        
        # compute the transformation matrix and its inverse (itk always needs the inverse)
        Tr = rm.get_rigid_motion_mat_from_euler(rotz, 'z', rotx, 'x', roty, 'y', tx, ty, tz)
        invT = np.array(Tr).astype(np.float32)  # very important conversion to float32, otherwise the code crashes

        for x in range(4):
            for y in range(4):
                self.invT[x, y] = np.float64(invT[x, y])

        # Move source point with transformation matrix, transform around volume center (subtract volume center point)
        source_transformed = np.dot(invT, np.array(
            [self.source[0] - self.center[0], self.source[1] - self.center[1], self.source[2] - self.center[2], 1.]).T)[
                             0:3]
        source_forGpu = np.array([source_transformed[0] + self.center[0], source_transformed[1] + self.center[1],
                                  source_transformed[2] + self.center[2]], dtype=np.float32)

        # Get array of physical coordinates of the transformed DRR (GPU accelerated)
        sourceDRR_array_augmented_transformed_gpu = torch.matmul(self.invT, self.sourceDRR_array_augmented)
        sourceDRR_array_transformed_gpu = torch.transpose(
            torch.matmul(self.Tn, sourceDRR_array_augmented_transformed_gpu)[0:3], 0, 1)
        sourceDRR_array_transformed = sourceDRR_array_transformed_gpu.cpu().numpy()
        sourceDRR_array_transf_to_ravel = sourceDRR_array_transformed.reshape(
            (self.DRRsize[0], self.DRRsize[1], self.Dimension), order='C')
        DRRPhy_array = np.ravel(sourceDRR_array_transf_to_ravel, order='C').astype(np.float32)

        # Update DRR
        output = self.projector.generateDRR(source_forGpu, DRRPhy_array)
        output_reshaped = np.reshape(output, (self.DRRsize[1], self.DRRsize[0]),
                                     order='C')
        toc = time.time()
        # print("tdrr time:", toc - tic)

        return output_reshaped
        


    def delete(self):
        
        """Deletes the projector object >>> GPU is freed <<<"""

        self.projector.delete()