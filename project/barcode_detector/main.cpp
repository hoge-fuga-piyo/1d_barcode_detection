#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/barcode.hpp>

#include "BarcodeDetector.h"

int main() {
	//cv::Mat image = cv::imread("../../../data/test.jpg");
	//cv::Mat image = cv::imread("../../../data/test2.jpg");
	//cv::Mat image = cv::imread("../../../data/test3.jpg");
	//cv::Mat image = cv::imread("../../../data/test4.jpg");
	//cv::Mat image = cv::imread("../../../data/test5.jpg");
	cv::Mat image = cv::imread("../../../data/test6.jpg");

	// バーコード検出
	BarcodeDetector decoder;
	cv::Mat result_image = image.clone();
	std::array<cv::Point, 4> corner = decoder.detect(image);
	for (const auto& point : corner) {
		cv::circle(result_image, point, 5, cv::Scalar(0, 0, 255), -1);
	}

	cv::imshow("result", result_image);
	cv::waitKey(0);

	// バーコードのデコード
	std::vector<cv::Point> tmp_corner{ corner[0], corner[1], corner[2], corner[3] };
	cv::barcode::BarcodeDetector barcode_decoder;
	std::vector<std::string> decoded_info;
	std::vector<cv::barcode::BarcodeType> decoded_type;
	barcode_decoder.decode(image, tmp_corner, decoded_info, decoded_type);
	std::cout << "implement barcode detector result" << std::endl;
	for (uint i = 0; i < decoded_info.size(); i++) {
		std::cout << decoded_info.at(i) << std::endl;
		std::cout << decoded_type.at(i) << std::endl;
	}

	// 比較用にOpenCVにデフォルトで実装されてるバーコード検出
	cv::barcode::BarcodeDetector default_barcode_detector;
	std::vector<cv::Point2f> default_corner;
	default_barcode_detector.detect(image, default_corner);
	std::vector<std::string> default_decoded_info;
	std::vector<cv::barcode::BarcodeType> default_decoded_type;
	default_barcode_detector.decode(image, default_corner, default_decoded_info, default_decoded_type);
	std::cout << "default barcode detector result" << std::endl;
	for (uint i = 0; i < default_decoded_info.size(); i++) {
		std::cout << default_decoded_info.at(i) << std::endl;
		std::cout << default_decoded_type.at(i) << std::endl;
	}

	return 0;
}
