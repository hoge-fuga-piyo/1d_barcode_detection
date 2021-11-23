#include <iostream>
#include <opencv2/opencv.hpp>

#include "BarcodeDetector.h"

int main() {
	//cv::Mat image = cv::imread("../../../data/test.jpg");
	//cv::Mat image = cv::imread("../../../data/test2.jpg");
	//cv::Mat image = cv::imread("../../../data/test3.jpg");
	cv::Mat image = cv::imread("../../../data/test4.jpg");

	BarcodeDetector decoder;
	decoder.detect(image);

	return 0;
}
