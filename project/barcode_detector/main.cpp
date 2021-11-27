#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/barcode.hpp>

#include "BarcodeDetector.h"

int main() {
	//cv::Mat image = cv::imread("../../../data/test.jpg");
	//cv::Mat image = cv::imread("../../../data/test2.jpg");
	cv::Mat image = cv::imread("../../../data/test3.jpg");
	//cv::Mat image = cv::imread("../../../data/test4.jpg");
	//cv::Mat image = cv::imread("../../../data/test5.jpg");

	// �o�[�R�[�h���o
	BarcodeDetector decoder;
	cv::Mat result_image = image.clone();
	std::array<cv::Point, 4> corner = decoder.detect(image);
	for (const auto& point : corner) {
		cv::circle(result_image, point, 5, cv::Scalar(0, 0, 255), -1);
	}

	cv::imshow("result", result_image);
	cv::waitKey(0);

	// �o�[�R�[�h�̃f�R�[�h
	std::vector<cv::Point> tmp_corner{ corner[0], corner[1], corner[2], corner[3] };
	cv::barcode::BarcodeDetector barcode_decoder;
	std::vector<std::string> decoded_info;
	std::vector<cv::barcode::BarcodeType> decoded_type;
	barcode_decoder.decode(image, tmp_corner, decoded_info, decoded_type);
	for (uint i = 0; i < decoded_info.size(); i++) {
		std::cout << decoded_info.at(i) << std::endl;
		std::cout << decoded_type.at(i) << std::endl;
	}

	return 0;
}
