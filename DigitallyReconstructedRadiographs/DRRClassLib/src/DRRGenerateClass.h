#pragma once

#ifndef DRRGENERATECLASS_H
#define DRRGENERATECLASS_H
#include <itkResampleImageFilter.h>
#include <itkCenteredEuler3DTransform.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkPNGImageIO.h>
#include <itkPNGImageIOFactory.h>
#include <itkImageSeriesWriter.h>
#include <itkNumericSeriesFileNames.h>
#include <itkGiplImageIOFactory.h>
#include <itkCastImageFilter.h>
#include <itkNIFTIImageIO.h>
#include <itkNIFTIImageIOFactory.h>
#include "itkImageSeriesReader.h" 
#include "itkGDCMImageIO.h" 
#include "itkGDCMSeriesFileNames.h"
#include <itkMetaImageIOFactory.h>
#include <itkRayCastInterpolateImageFunction.h>
#include <iostream>
#include <io.h>
using namespace std;

class DRRGenerate
{
public:
	string dcmfilepath;
	string save_path;
	bool verbose;
	double rx, ry, rz, tx, ty, tz;
	double cx, cy, cz;
	float sid;
	float sx, sy, dx, dy;
	float o2Dx, o2Dy;
	double threshold;
	itk::Image<float, 3>::Pointer image;


	DRRGenerate(string m_dcmfilepath, bool m_verbose);

	int ReadDCM(void);
	int Drr1(string m_save_path,
		double m_rx, double m_ry, double m_rz, double m_tx, double m_ty, double m_tz,
		float m_sid, float m_sx, float m_sy, float m_dx, float m_dy, double m_threshold);

};



#endif // !DRRGENERATECLASS_H




