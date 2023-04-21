#include "DRRGenerateClass.h"


DRRGenerate::DRRGenerate(string m_dcmfilepath, bool m_verbose)
{
	
	dcmfilepath = m_dcmfilepath;
	verbose = m_verbose;
}


int DRRGenerate::ReadDCM(void)
{
	using InputPixelType = float;
	using OutputPixelType = unsigned char;
	using InputImageType = itk::Image<InputPixelType, 3>;
	using OutputImageType = itk::Image<OutputPixelType, 3>;
	//InputImageType::Pointer image;

	//using ReaderType = itk::ImageFileReader<InputImageType>;
	//ReaderType::Pointer reader = ReaderType::New();
	//itk::NiftiImageIOFactory::RegisterOneFactory();
	//reader->SetFileName(file_path);
	using ReaderType = itk::ImageSeriesReader< InputImageType >;
	ReaderType::Pointer reader = ReaderType::New();

	using ImageIOType = itk::GDCMImageIO;
	ImageIOType::Pointer dicomIO = ImageIOType::New();
	reader->SetImageIO(dicomIO);

	using NamesGeneratorType = itk::GDCMSeriesFileNames;
	NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
	nameGenerator->SetInputDirectory(dcmfilepath);

	using FileNamesContainer = vector<string>;
	FileNamesContainer fileNames = nameGenerator->GetInputFileNames();
	reader->SetFileNames(fileNames);

	try {
		reader->Update();
	}
	catch (itk::ExceptionObject& err) {
		cerr << "error exception object caught!" << endl;
		cerr << err.what() << endl;
		return EXIT_FAILURE;
	}
	image = reader->GetOutput();

	if (verbose) {
		const InputImageType::SpacingType spacing = image->GetSpacing();
		cout << endl << "Input: ";

		InputImageType::RegionType region = image->GetBufferedRegion();
		region.Print(cout);

		cout << " Resolution: [";
		for (int i = 0; i < 3; i++) {
			cout << spacing[i];
			if (i < 3 - 1) cout << ", ";
		}
		cout << "]" << endl;

		const InputImageType::PointType origin = image->GetOrigin();
		cout << " Origin: [";
		for (int i = 0; i < 3; i++) {
			cout << origin[i];
			if (i < 3 - 1) cout << ", ";
		}
		cout << "]" << endl << endl;
	}
	return EXIT_SUCCESS;
}

int DRRGenerate::Drr1(string m_save_path,
	double m_rx, double m_ry, double m_rz, double m_tx, double m_ty, double m_tz,
	float m_sid, float m_sx, float m_sy, float m_dx, float m_dy, double m_threshold)
{
	using InputPixelType = float;
	using OutputPixelType = unsigned char;
	using InputImageType = itk::Image<InputPixelType, 3>;
	using OutputImageType = itk::Image<OutputPixelType, 3>;

	float o2Dx = 0;
	float o2Dy = 0;
	double cx = 0.;
	double cy = 0.;
	double cz = 0.;

	rx = m_rx;
	ry = m_ry;
	rz = m_rz;
	tx = m_tx;
	ty = m_ty;
	tz = m_tz;

	sid = m_sid;
	sx = m_sx;
	sy = m_sy;
	dx = m_dx;
	dy = m_dy;
	threshold = m_threshold;

	
	save_path = m_save_path;

	using FilterType = itk::ResampleImageFilter<InputImageType, InputImageType>;
	FilterType::Pointer filter = FilterType::New();
	filter->SetInput(image);
	filter->SetDefaultPixelValue(0);

	using TransformType = itk::CenteredEuler3DTransform<double>;
	TransformType::Pointer transform = TransformType::New();

	transform->SetComputeZYX(true);
	TransformType::OutputVectorType translation;

	translation[0] = tx;
	translation[1] = ty;
	translation[2] = tz;

	const double dtr = atan(1.0) * 4.0 / 180.0;

	transform->SetTranslation(translation);
	transform->SetRotation(dtr * rx, dtr * ry, dtr * rz);

	InputImageType::PointType imOrigin = image->GetOrigin();
	InputImageType::SpacingType imRes = image->GetSpacing();

	using InputImageRegionType = InputImageType::RegionType;
	using InputImageSizeType = InputImageRegionType::SizeType;

	InputImageRegionType imRegion = image->GetBufferedRegion();
	InputImageSizeType imSize = imRegion.GetSize();

	imOrigin[0] += imRes[0] * static_cast<double>(imSize[0]) / 2.0;
	imOrigin[1] += imRes[1] * static_cast<double>(imSize[1]) / 2.0;
	imOrigin[2] += imRes[2] * static_cast<double>(imSize[2]) / 2.0;

	TransformType::InputPointType center;
	center[0] = cx + imOrigin[0];
	center[1] = cy + imOrigin[1];
	center[2] = cz + imOrigin[2];
	transform->SetCenter(center);

	if (verbose) {
		cout << "Image size: " << imSize[0] << ", " << imSize[1] << ", " << imSize[2]
			<< endl << " resolution: " << imRes[0] << ", " << imRes[1] << ", " << imRes[2]
			<< endl << " origin: " << imOrigin[0] << ", " << imOrigin[1] << ", " <<
			imOrigin[2] << endl << " center: " << center[0] << ", " << center[1]
			<< ", " << center[2] << endl << "Transform: " << transform << endl;
	}

	using InterpolatorType = itk::RayCastInterpolateImageFunction<InputImageType, double>;
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	interpolator->SetTransform(transform);

	interpolator->SetThreshold(threshold);
	InterpolatorType::InputPointType focalpoint;

	focalpoint[0] = imOrigin[0];
	focalpoint[1] = imOrigin[1];
	focalpoint[2] = imOrigin[2] - sid / 2.0;

	interpolator->SetFocalPoint(focalpoint);

	if (verbose) {
		cout << "Focal Point: "
			<< focalpoint[0] << ", "
			<< focalpoint[1] << ", "
			<< focalpoint[2] << endl;
	}

	//interpolator->Print(std::cout);

	filter->SetInterpolator(interpolator);
	filter->SetTransform(transform);

	// setup the scene
	InputImageType::SizeType   size;
	size[0] = dx;  // number of pixels along X of the 2D DRR image
	size[1] = dy;  // number of pixels along Y of the 2D DRR image
	size[2] = 1;   // only one slice

	filter->SetSize(size);

	InputImageType::SpacingType spacing;

	spacing[0] = sx;  // pixel spacing along X of the 2D DRR image [mm]
	spacing[1] = sy;  // pixel spacing along Y of the 2D DRR image [mm]
	spacing[2] = 1.0; // slice thickness of the 2D DRR image [mm]
	filter->SetOutputSpacing(spacing);

	if (verbose)
	{
		std::cout << "Output image size: "
			<< size[0] << ", "
			<< size[1] << ", "
			<< size[2] << std::endl;

		std::cout << "Output image spacing: "
			<< spacing[0] << ", "
			<< spacing[1] << ", "
			<< spacing[2] << std::endl;
	}

	double origin[3];
	origin[0] = imOrigin[0] + o2Dx - sx * ((double)dx - 1.) / 2.;
	origin[1] = imOrigin[1] + o2Dy - sy * ((double)dy - 1.) / 2.;
	origin[2] = imOrigin[2] + sid / 2.;
	filter->SetOutputOrigin(origin);
	if (verbose)
	{
		std::cout << "Output image origin: "
			<< origin[0] << ", "
			<< origin[1] << ", "
			<< origin[2] << std::endl;
	}

	// create writer
	using RescaleFilterType = itk::RescaleIntensityImageFilter<InputImageType, OutputImageType>;
	RescaleFilterType::Pointer rescaler = RescaleFilterType::New();
	rescaler->SetOutputMinimum(0);
	rescaler->SetOutputMaximum(255);
	rescaler->SetInput(filter->GetOutput());

	using WriterType = itk::ImageFileWriter<OutputImageType>;
	WriterType::Pointer writer = WriterType::New();

	using pngType = itk::PNGImageIO;
	pngType::Pointer pngIO1 = pngType::New();
	itk::PNGImageIOFactory::RegisterOneFactory();
	writer->SetFileName(save_path);
	writer->SetImageIO(pngIO1);
	writer->SetImageIO(itk::PNGImageIO::New());
	writer->SetInput(rescaler->GetOutput());

	try
	{
		// std::cout << "Writing image: " << save_path << std::endl;
		writer->Update();
	}
	catch (itk::ExceptionObject& err)
	{
		std::cerr << "ERROR: ExceptionObject caught !" << std::endl;
		std::cerr << err << std::endl;
	}
	return 0;

}