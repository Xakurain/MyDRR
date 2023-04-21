from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("SiddonGpuPy",
                             sources=["SiddonGpuPy.pyx"],
                             include_dirs=[numpy.get_include(), 
                                           "F:\\code\\python\\iMIA\\CUDA_DigitallyReconstructedRadiographs-master\\SiddonPythonModule\\include"
                                           "D:\\Program Files\\NVIDIA\CUDA\\v11.6\\include"],
                             library_dirs = ["F:\\code\python\\iMIA\\CUDA_DigitallyReconstructedRadiographs-master\\SiddonPythonModule\\lib",
                                             "D:\\Program Files\\NVIDIA\CUDA\\v11.6\\lib\\x64"],
                             libraries = ["SiddonGpu", "cudart_static"],
                             language = "c++")]
)