#include <iostream>
#include <CL/cl.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <CL\cl.h>
#include <iterator>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <tbb\tbb.h>
#include <tbb\parallel_for.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static clock_t openCLExecutionGPUBegin, openCLExecutionGPUEnd, openCLExecutionHDGraphicsBegin,
openCLExecutionHDGraphicsEnd, openCLExecutionOnCpuBegin, openCLExecutionOnCpuEnd;
using namespace tbb;
/// It is a InnerBlur Function without a main loop that is called n times by tbbParallelBlur function
void innerBlur(int i, IplImage* img, IplImage* dst, int innerSize) {
	uchar* data = (uchar *)img->imageData;
	uchar* dstData = (uchar *)dst->imageData;
	int total = 0;
	int j, x, y, tx, ty;
	/// Width of Original Matrix
	for (j = 0; j < img->width; j++) {
		int ksize = innerSize;
		total = 0;
		/// Height of Inner Matrix 
		for (x = -ksize / 2; x <= ksize / 2; x++)
			/// Width of Inner Matrix 
			for (y = -ksize / 2; y <= ksize / 2; y++)
			{
				tx = i + x;
				ty = j + y;
				if (tx >= 0 && tx < img->height && ty >= 0 && ty < img->width)
				{
					/// Adds color values of pixels
					total += data[tx*img->widthStep + ty];
				}
			}
		/// Takes avarage of total value which is summation of color values and writes as a new value
		dstData[i*img->widthStep + j] = total / ksize / ksize;
	}
}
using namespace cv;
///It is a tbbParallelBlur function that will call parallel_for in tbb library with compact lambda function
void tbbParallelBlur(IplImage* img, IplImage* dst, size_t n, int innerSize) {
	parallel_for(blocked_range<size_t>(0, n), [=](const blocked_range<size_t>& r) {
		/// Parallel for function that runs our innerBlur function in different cores, calls n times and gives i parameter
		for (size_t i = r.begin(); i != r.end(); ++i) {
			innerBlur(i, img, dst, innerSize);
		}
	}
	);
}
/// It is a standartBlur function without any parallelization
void standartBlur(IplImage* img, IplImage* dst, int innerSize) {
	unsigned char* data = (unsigned char *)img->imageData;
	unsigned char* dstData = (unsigned char *)dst->imageData;
	int total = 0;
	int i, j, x, y, tx, ty;
	/// Height of Original Matrix
	for (i = 0; i < img->height; i++) {
		/// Width of Original Matrix
		for (j = 0; j < img->width; j++) {
			int ksize = innerSize;
			total = 0;
			/// Height of Inner Matrix
			for (x = -ksize / 2; x <= ksize / 2; x++)
				/// Width of Inner Matrix 
				for (y = -ksize / 2; y <= ksize / 2; y++)
				{
					tx = i + x;
					ty = j + y;
					if (tx >= 0 && tx < img->height && ty >= 0 && ty < img->width)
					{
						/// Adds color values of pixels
						total += data[tx*img->widthStep + ty];
					}
				}
			/// Takes avarage of total value which is summation of color values and writes as a new value
			dstData[i*img->widthStep + j] = total / ksize / ksize;
		}
	}
}
/// It is a OpenCLOnGPU function which sends our data to memory of graphic card, execute kernel code and then get results

void openCLOnGPU(IplImage* img, IplImage* dst, int innerSize) {
	int height, width, step, channels;
	int height2, width2, step2, channels2;
	height = img->height;
	width = img->width;
	step = img->widthStep;
	channels = img->nChannels;
	unsigned char* data = (unsigned char *)img->imageData;

	height2 = dst->height;
	width2 = dst->width;
	step2 = dst->widthStep;
	channels2 = dst->nChannels;
	unsigned char* dstData = (unsigned char *)dst->imageData;


	/// Gets all platforms (drivers) and set default platform
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}

	cl::Platform default_platform = all_platforms[1];


	/// Sets default device of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device = all_devices[0];


	cl::Context context({ default_device });

	cl::Program::Sources sources;

	/// Kernel function that runs on GPU and calcute values
	std::string kernel_code =
		"void kernel simple_add(global const int* size, global unsigned char* A, global unsigned char* B) {"
		"	int ksize = size[3];"
		"	int total = 0;"
		"	int x, y, tx, ty;"
		"	for (x = -ksize / 2; x <= ksize / 2; x++)"
		"		for (y = -ksize / 2; y <= ksize / 2; y++)"
		"		{"
		"			tx = get_global_id(1) + x;"
		"			ty = get_global_id(0) + y;"
		"			if (tx >= 0 && tx < size[0] && ty >= 0 && ty < size[1])"
		"			{"
		"				total += B[tx*size[2] + ty];"
		"			}"
		"		}"
		"	A[get_global_id(1)*size[2] + get_global_id(0)] = total / ksize / ksize;"
		"}	";


	/// Sends our kernel code into sources object
	sources.push_back({ kernel_code.c_str(),kernel_code.length() });

	/// Builds kernel code 
	cl::Program program(context, sources);
	if (program.build({ default_device }) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		exit(1);
	}


	/// Creates buffers on the device
	cl::Buffer buffer_size(context, CL_MEM_READ_WRITE, sizeof(int) * 4);
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * img->imageSize);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * img->imageSize);

	/// Creates an array that stores data for buffer
	int *size;
	size = (int *)malloc(sizeof(int) * 4);
	size[0] = height;
	size[1] = width;
	size[2] = step;
	size[3] = innerSize;


	/// Creates queue to which we will push commands for the device.
	cl::CommandQueue queue(context, default_device);

	/// Writes size to buffer_size, dstData to buffer_A and dstData to buffer_B
	queue.enqueueWriteBuffer(buffer_size, CL_TRUE, 0, sizeof(int) * 4, size);
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(unsigned char) * img->imageSize, dstData);
	queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(unsigned char) * img->imageSize, dstData);



	openCLExecutionGPUBegin = clock();

	/// Prepares kernel_add function to execute
	cl::Kernel kernel_add = cl::Kernel(program, "simple_add");
	kernel_add.setArg(0, buffer_size);
	kernel_add.setArg(1, buffer_A);
	kernel_add.setArg(2, buffer_B);


	/// Sets the dimension (2D) and execute the kernel
	cl::NDRange global_work_size(width, height);
	queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, global_work_size, cl::NullRange);
	queue.finish();
	openCLExecutionGPUEnd = clock();

	/// Reads buffer_A from the device and write to unsigned char dstData
	queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, sizeof(unsigned char) * img->imageSize, dstData);
}
void openClOnCPU(IplImage* img, IplImage* dst, int innerSize) {
	int height, width, step, channels;
	int height2, width2, step2, channels2;
	height = img->height;
	width = img->width;
	step = img->widthStep;
	channels = img->nChannels;
	unsigned char* data = (unsigned char *)img->imageData;

	height2 = dst->height;
	width2 = dst->width;
	step2 = dst->widthStep;
	channels2 = dst->nChannels;
	unsigned char* dstData = (unsigned char *)dst->imageData;


	/// Gets all platforms (drivers) and set default platform
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}

	cl::Platform default_platform = all_platforms[0];

	/// Sets default device of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device = all_devices[0];

	cl::Context context({ default_device });

	cl::Program::Sources sources;

	/// Kernel function that runs on GPU and calcute values
	std::string kernel_code =
		"void kernel simple_add(global const int* size, global unsigned char* A, global unsigned char* B) {"
		"	int ksize = size[3];"
		"	int total = 0;"
		"	int x, y, tx, ty;"
		"	for (x = -ksize / 2; x <= ksize / 2; x++)"
		"		for (y = -ksize / 2; y <= ksize / 2; y++)"
		"		{"
		"			tx = get_global_id(1) + x;"
		"			ty = get_global_id(0) + y;"
		"			if (tx >= 0 && tx < size[0] && ty >= 0 && ty < size[1])"
		"			{"
		"				total += B[tx*size[2] + ty];"
		"			}"
		"		}"
		"	A[get_global_id(1)*size[2] + get_global_id(0)] = total / ksize / ksize;"
		"}	";


	/// Sends our kernel code into sources object
	sources.push_back({ kernel_code.c_str(),kernel_code.length() });

	/// Builds kernel code 
	cl::Program program(context, sources);
	if (program.build({ default_device }) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		exit(1);
	}


	/// Creates buffers on the device
	cl::Buffer buffer_size(context, CL_MEM_READ_WRITE, sizeof(int) * 4);
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * img->imageSize);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * img->imageSize);

	/// Creates an array that stores data for buffer
	int *size;
	size = (int *)malloc(sizeof(int) * 4);
	size[0] = height;
	size[1] = width;
	size[2] = step;
	size[3] = innerSize;


	/// Creates queue to which we will push commands for the device.
	cl::CommandQueue queue(context, default_device);

	/// Writes size to buffer_size, dstData to buffer_A and dstData to buffer_B
	queue.enqueueWriteBuffer(buffer_size, CL_TRUE, 0, sizeof(int) * 4, size);
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(unsigned char) * img->imageSize, dstData);
	queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(unsigned char) * img->imageSize, dstData);



	openCLExecutionOnCpuBegin = clock();

	/// Prepares kernel_add function to execute
	cl::Kernel kernel_add = cl::Kernel(program, "simple_add");
	kernel_add.setArg(0, buffer_size);
	kernel_add.setArg(1, buffer_A);
	kernel_add.setArg(2, buffer_B);


	/// Sets the dimension (2D) and execute the kernel
	cl::NDRange global_work_size(width, height);
	queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, global_work_size, cl::NullRange);
	queue.finish();
	openCLExecutionOnCpuEnd = clock();

	/// Reads buffer_A from the device and write to unsigned char dstData
	queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, sizeof(unsigned char) * img->imageSize, dstData);
}
void openCLOnIntelHdGraphicCard(IplImage* img, IplImage* dst, int innerSize) {
	int height, width, step, channels;
	int height2, width2, step2, channels2;
	height = img->height;
	width = img->width;
	step = img->widthStep;
	channels = img->nChannels;
	unsigned char* data = (unsigned char *)img->imageData;

	height2 = dst->height;
	width2 = dst->width;
	step2 = dst->widthStep;
	channels2 = dst->nChannels;
	unsigned char* dstData = (unsigned char *)dst->imageData;


	/// Gets all platforms (drivers) and set default platform
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}

	cl::Platform default_platform = all_platforms[0];


	/// Sets default device of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device = all_devices[1];

	cl::Context context({ default_device });

	cl::Program::Sources sources;

	/// Kernel function that runs on GPU and calcute values
	std::string kernel_code =
		"void kernel simple_add(global const int* size, global unsigned char* A, global unsigned char* B) {"
		"	int ksize = size[3];"
		"	int total = 0;"
		"	int x, y, tx, ty;"
		"	for (x = -ksize / 2; x <= ksize / 2; x++)"
		"		for (y = -ksize / 2; y <= ksize / 2; y++)"
		"		{"
		"			tx = get_global_id(1) + x;"
		"			ty = get_global_id(0) + y;"
		"			if (tx >= 0 && tx < size[0] && ty >= 0 && ty < size[1])"
		"			{"
		"				total += B[tx*size[2] + ty];"
		"			}"
		"		}"
		"	A[get_global_id(1)*size[2] + get_global_id(0)] = total / ksize / ksize;"
		"}	";


	/// Sends our kernel code into sources object
	sources.push_back({ kernel_code.c_str(),kernel_code.length() });

	/// Builds kernel code 
	cl::Program program(context, sources);
	if (program.build({ default_device }) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		exit(1);
	}


	/// Creates buffers on the device
	cl::Buffer buffer_size(context, CL_MEM_READ_WRITE, sizeof(int) * 4);
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * img->imageSize);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * img->imageSize);

	/// Creates an array that stores data for buffer
	int *size;
	size = (int *)malloc(sizeof(int) * 4);
	size[0] = height;
	size[1] = width;
	size[2] = step;
	size[3] = innerSize;


	/// Creates queue to which we will push commands for the device.
	cl::CommandQueue queue(context, default_device);

	/// Writes size to buffer_size, dstData to buffer_A and dstData to buffer_B
	queue.enqueueWriteBuffer(buffer_size, CL_TRUE, 0, sizeof(int) * 4, size);
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(unsigned char) * img->imageSize, dstData);
	queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(unsigned char) * img->imageSize, dstData);



	openCLExecutionHDGraphicsBegin = clock();

	/// Prepares kernel_add function to execute
	cl::Kernel kernel_add = cl::Kernel(program, "simple_add");
	kernel_add.setArg(0, buffer_size);
	kernel_add.setArg(1, buffer_A);
	kernel_add.setArg(2, buffer_B);


	/// Sets the dimension (2D) and execute the kernel
	cl::NDRange global_work_size(width, height);
	queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, global_work_size, cl::NullRange);
	queue.finish();
	openCLExecutionHDGraphicsEnd = clock();

	/// Reads buffer_A from the device and write to unsigned char dstData
	queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, sizeof(unsigned char) * img->imageSize, dstData);
}
/// It is a general function that takes high resolution images, calls proper functions and print the results.
void highResolution(IplImage* img, IplImage* dstStandart, IplImage* dstParallel, IplImage* dstGPU, IplImage* dstHdGraph, IplImage* dstCpu, Mat src) {
	Mat out;

	/// Sets properties of windows that we use to show images
	namedWindow("imgHigh", CV_WINDOW_NORMAL);
	cvMoveWindow("imgHigh", 540, 125);
	resizeWindow("imgHigh", 800, 800);

	namedWindow("dstStandartHigh", CV_WINDOW_NORMAL);
	cvMoveWindow("dstStandartHigh", 530, 125);
	resizeWindow("dstStandartHigh", 800, 800);

	namedWindow("dstTbbHigh", CV_WINDOW_NORMAL);
	cvMoveWindow("dstTbbHigh", 520, 125);
	resizeWindow("dstTbbHigh", 800, 800);

	namedWindow("opencvBlurHigh", CV_WINDOW_NORMAL);
	cvMoveWindow("opencvBlurHigh", 510, 125);
	resizeWindow("opencvBlurHigh", 800, 800);

	namedWindow("dstGPUHigh", CV_WINDOW_NORMAL);
	cvMoveWindow("dstGPUHigh", 500, 125);
	resizeWindow("dstGPUHigh", 800, 800);

	namedWindow("dstHdGraphHigh", CV_WINDOW_NORMAL);
	cvMoveWindow("dstHdGraphHigh", 500, 125);
	resizeWindow("dstHdGraphHigh", 800, 800);

	namedWindow("dstCpuHigh", CV_WINDOW_NORMAL);
	cvMoveWindow("dstCpuHigh", 500, 125);
	resizeWindow("dstCpuHigh", 800, 800);

	cvShowImage("imgHigh", img);

	/// Calls standartBlur function with calculating time and print results
	clock_t totalStandartBegin = clock();
	standartBlur(img, dstStandart, 3);
	clock_t totalStandartEnd = clock();
	cvShowImage("dstStandartHigh", dstStandart);
	std::cout << "\n\n	Standart Blur Function (Without Parallelization) Execution Time :  "
		<< (totalStandartEnd - totalStandartBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Device: Intel CPU " << "\n\n";


	/// Calls tbbParallelBlur function with calculating time and print results
	clock_t totalTbbBegin = clock();
	tbbParallelBlur(img, dstParallel, img->height, 3);
	clock_t totalTbbEnd = clock();
	cvShowImage("dstTbbHigh", dstParallel);
	std::cout << "\n\n	Blur Function (With Parallelization) With Thread Building Blocks Execution Time  :  "
		<< (totalTbbEnd - totalTbbBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Device: Intel CPU " << "\n\n";

	/// Calls already implemented blur function in OpenCV with calculating time and print results
	clock_t totalOpenCvBegin = clock();
	blur(src, out, Size(3, 3));
	clock_t totalOpenCvEnd = clock();
	imshow("opencvBlurHigh", out);
	std::cout << "\n\n	Already Implemented Blur (With Parallelization) Method In OpenCV Execution Time  :  "
		<< (totalOpenCvEnd - totalOpenCvBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Device: Intel CPU" << "\n\n";

	clock_t totalHdGraphBegin = clock();
	openCLOnIntelHdGraphicCard(img, dstHdGraph, 3);
	clock_t totalHdGraphEnd = clock();
	cvShowImage("dstHdGraphHigh", dstHdGraph);
	std::cout << "\n\n	Blur Function (With Parallelization) With OpenCL On Intel HD Graphics Total Time :  " << (totalHdGraphEnd - totalHdGraphBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Only Execution Time :  "
		<< (openCLExecutionHDGraphicsEnd - openCLExecutionHDGraphicsBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Device: Intel HD Graphics" << "\n\n";

	clock_t totalCpuBegin = clock();
	openClOnCPU(img, dstCpu, 3);
	clock_t totalCpuEnd = clock();
	cvShowImage("dstHdGraphHigh", dstCpu);
	std::cout << "\n\n	Blur Function (With Parallelization) With OpenCL On Intel CPU Total Time :  " << (totalCpuEnd - totalCpuBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Only Execution Time :  "
		<< (openCLExecutionOnCpuEnd - openCLExecutionOnCpuBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Device: Intel CPU" << "\n\n";

	/// Calls openCLOnGPU function with calculating time and print results
	clock_t totalGpuBegin = clock();
	openCLOnGPU(img, dstGPU, 3);
	clock_t totalGpuEnd = clock();
	cvShowImage("dstGPUHigh", dstGPU);
	std::cout << "\n\n	Blur Function (With Parallelization) With OpenCL On GPU Total Time :  " << (totalGpuEnd - totalGpuBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Only Execution Time :  "
		<< (openCLExecutionGPUEnd - openCLExecutionGPUBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Device: Nvidia Graphic Card" << "\n\n";
}

/// It is a general function that takes low resolution images, calls proper functions and print the results.
void lowResolution(IplImage* img, IplImage* dstStandart, IplImage* dstParallel, IplImage* dstGPU, IplImage* dstHdGraph, IplImage* dstCpu, Mat src) {
	Mat out;

	/// Sets properties of windows that we use to show images
	namedWindow("imgLow", CV_WINDOW_NORMAL);
	cvMoveWindow("imgLow", 490, 125);
	resizeWindow("imgLow", 800, 800);

	namedWindow("dstStandartLow", CV_WINDOW_NORMAL);
	cvMoveWindow("dstStandartLow", 480, 125);
	resizeWindow("dstStandartLow", 800, 800);

	namedWindow("dstTbbLow", CV_WINDOW_NORMAL);
	cvMoveWindow("dstTbbLow", 470, 125);
	resizeWindow("dstTbbLow", 800, 800);

	namedWindow("opencvBlurLow", CV_WINDOW_NORMAL);
	cvMoveWindow("opencvBlurLow", 460, 125);
	resizeWindow("opencvBlurLow", 800, 800);

	namedWindow("dstGPULow", CV_WINDOW_NORMAL);
	cvMoveWindow("dstGPULow", 450, 125);
	resizeWindow("dstGPULow", 800, 800);

	namedWindow("dstHdGraphLow", CV_WINDOW_NORMAL);
	cvMoveWindow("dstHdGraphLow", 450, 125);
	resizeWindow("dstHdGraphLow", 800, 800);

	namedWindow("dstCpuHigh", CV_WINDOW_NORMAL);
	cvMoveWindow("dstCpuHigh", 500, 125);
	resizeWindow("dstCpuHigh", 800, 800);

	cvShowImage("imgLow", img);

	/// Calls standartBlur function with calculating time and print results
	clock_t totalStandartBegin = clock();
	standartBlur(img, dstStandart, 3);
	clock_t totalStandartEnd = clock();
	cvShowImage("dstStandartLow", dstStandart);
	std::cout << "\n\n	Standart Blur Function (Without Parallelization) Execution Time :  "
		<< (totalStandartEnd - totalStandartBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Device: Intel CPU " << "\n\n";

	/// Calls tbbParallelBlur function with calculating time and print results
	clock_t totalTbbBegin = clock();
	tbbParallelBlur(img, dstParallel, img->height, 3);
	clock_t totalTbbEnd = clock();
	cvShowImage("dstTbbLow", dstParallel);
	std::cout << "\n\n	Blur Function (With Parallelization) With Thread Building Blocks Execution Time  :  "
		<< (totalTbbEnd - totalTbbBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Device: Intel CPU " << "\n\n";

	/// Calls already implemented blur function in OpenCV with calculating time and print results
	clock_t totalOpenCvBegin = clock();
	blur(src, out, Size(3, 3));
	clock_t totalOpenCvEnd = clock();
	imshow("opencvBlurLow", out);
	std::cout << "\n\n	Already Implemented Blur (With Parallelization) Method In OpenCV Execution Time  :  "
		<< (totalOpenCvEnd - totalOpenCvBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Device: Intel CPU" << "\n\n";

	clock_t totalHdGraphBegin = clock();
	openCLOnIntelHdGraphicCard(img, dstHdGraph, 3);
	clock_t totalHdGraphEnd = clock();
	cvShowImage("dstHdGraphLow", dstHdGraph);
	std::cout << "\n\n	Blur Function (With Parallelization) With OpenCL On Intel HD Graphics Total Time :  " << (totalHdGraphEnd - totalHdGraphBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Only Execution Time :  "
		<< (openCLExecutionHDGraphicsEnd - openCLExecutionHDGraphicsBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Device: Intel HD Graphics" << "\n\n";

	clock_t totalCpuBegin = clock();
	openClOnCPU(img, dstCpu, 3);
	clock_t totalCpuEnd = clock();
	cvShowImage("dstHdGraphHigh", dstCpu);
	std::cout << "\n\n	Blur Function (With Parallelization) With OpenCL On Intel CPU Total Time :  " << (totalCpuEnd - totalCpuBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Only Execution Time :  "
		<< (openCLExecutionOnCpuEnd - openCLExecutionOnCpuBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Device: Intel CPU" << "\n\n";

	/// Calls openCLOnGPU function with calculating time and print results
	clock_t totalGpuBegin = clock();
	openCLOnGPU(img, dstGPU, 3);
	clock_t totalGpuEnd = clock();
	cvShowImage("dstGPULow", dstGPU);
	std::cout << "\n\n	Blur Function (With Parallelization) With OpenCL On GPU Total Time :  " << (totalGpuEnd - totalGpuBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Only Execution Time :  "
		<< (openCLExecutionGPUEnd - openCLExecutionGPUBegin) / (CLOCKS_PER_SEC / 1000) << " ms\n	Device: Nvidia Graphic Card" << "\n\n";
}
/// It is a main function that reads images and calls proper functions.
int main() {
	/// Reads images and takes clones of it
	IplImage* imgHigh = cvLoadImage("C:\\Users\\MertLaptop\\Desktop\\h.jpg", 0);
	IplImage* dstStandartHigh = cvCloneImage(imgHigh);
	IplImage* dstParallelHigh = cvCloneImage(imgHigh);
	IplImage* dstGPUHigh = cvCloneImage(imgHigh);
	IplImage* dstHdGraphHigh = cvCloneImage(imgHigh);
	IplImage* dstCpuHigh = cvCloneImage(imgHigh);

	Mat srcHigh = imread("C:\\Users\\MertLaptop\\Desktop\\h.jpg", 0);


	/// Reads images and takes clones of it
	IplImage* imgLow = cvLoadImage("C:\\Users\\MertLaptop\\Desktop\\a.jpeg", 0);
	IplImage* dstStandartLow = cvCloneImage(imgLow);
	IplImage* dstParallelLow = cvCloneImage(imgLow);
	IplImage* dstGPULow = cvCloneImage(imgLow);
	IplImage* dstHdGraphLow = cvCloneImage(imgLow);
	IplImage* dstCpuLow = cvCloneImage(imgLow);
	Mat srcLow = imread("C:\\Users\\MertLaptop\\Desktop\\a.jpeg", 0);


	/// Prints properties of images
	std::cout << "\n		<High Resolution Picture (" << imgHigh->width << " x " << imgHigh->height << ")>";
	highResolution(imgHigh, dstStandartHigh, dstParallelHigh, dstGPUHigh, dstHdGraphHigh, dstCpuHigh, srcHigh);
	std::cout << "\n\n----------------------------------------------------------------------------------------------------\n\n";
	std::cout << "\n\n		<Low Resolution Picture (" << imgLow->width << " x " << imgLow->height << ")>";
	lowResolution(imgLow, dstStandartLow, dstParallelLow, dstGPULow, dstHdGraphLow, dstCpuLow, srcLow);


	cvWaitKey(0);
	return 0;
}