#include <iostream>
#include <opencv2/opencv.hpp>

#include "BarcodeDetector.h"

int main() {
	//cv::Mat image = cv::imread("../../../data/test.jpg");
	cv::Mat image = cv::imread("../../../data/test2.jpg");
	//cv::Mat image = cv::imread("../../../data/test3.jpg");
	//cv::Mat image = cv::imread("../../../data/test4.jpg");

	BarcodeDetector decoder;
	cv::Mat result_image = image.clone();
	std::array<cv::Point, 4> corner = decoder.detect(image);
	for (const auto& point : corner) {
		cv::circle(result_image, point, 10, cv::Scalar(0, 0, 255), -1);
	}

	cv::imshow("result", result_image);
	cv::waitKey(0);

	return 0;
}
