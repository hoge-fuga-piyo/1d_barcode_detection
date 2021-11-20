#include <iostream>
#include <opencv2/opencv.hpp>

#include "BarcodeDetector.h"

int main() {
	cv::Mat image = cv::imread("../../../data/test.jpg");
	//cv::Mat image = cv::imread("../../../data/test2.jpg");
	//cv::Mat image = cv::imread("../../../data/test3.jpg");

	BarcodeDetector decoder;
	decoder.detect(image);

	//cv::imshow("image", image);
	//cv::waitKey(0);

	return 0;
}
