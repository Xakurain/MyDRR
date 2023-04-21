import cython
from ctypes import *
# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np
from libcpp cimport bool
from libcpp.string cimport string

# declare the interface to the C++ code
cdef extern from "include\DRRGenerateClass.h" :
    cdef cppclass DRRGenerate:
        DRRGenerate(string m_dcmfilepath,
                  bool m_verbose)
        int ReadDCM()
        int Drr1(string m_save_path,
		        double m_rx, double m_ry, double m_rz, double m_tx, double m_ty, double m_tz,
		        float m_sid, float m_sx, float m_sy, float m_dx, float m_dy, double m_threshold)

cdef class pyDRRGenerate:
    cdef DRRGenerate* thisptr # hold a C++ instance
    def __cinit__(self, 
                    m_dcmfilepath, 
                    bool m_verbose):
        cdef string s = m_dcmfilepath.encode(encoding='utf-8',errors= 'strict')
        self.thisptr = new DRRGenerate(s, m_verbose)

    def ReadDCM(self):

        return self.thisptr.ReadDCM()

    def Drr1(self, m_save_path,
             m_rx, m_ry, m_rz, m_tx, m_ty, m_tz, m_sid, m_sx, m_sy, m_dx, m_dy, m_threshold):

        cdef string s = m_save_path.encode(encoding='utf-8',errors= 'strict')   
        return self.thisptr.Drr1(s, m_rx, m_ry, m_rz, m_tx, m_ty, m_tz, m_sid, m_sx, m_sy, m_dx, m_dy, m_threshold)

    def delete(self) :
        if self.thisptr is not NULL :
            "C++ object being destroyed"
            del self.thisptr