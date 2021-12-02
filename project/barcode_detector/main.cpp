#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/barcode.hpp>

#include "BarcodeDetector.h"
#include "BarcodeDetector2.h"
#include "DefaultBarcodeDetector.h"

int main() {
	//cv::Mat image = cv::imread("../../../data/test.jpg");
	//cv::Mat image = cv::imread("../../../data/test2.jpg");
	//cv::Mat image = cv::imread("../../../data/test3.jpg");
	//cv::Mat image = cv::imread("../../../data/test4.jpg");
	//cv::Mat image = cv::imread("../../../data/test5.jpg");
	//cv::Mat image = cv::imread("../../../data/test6.jpg");
	//cv::Mat image = cv::imread("../../../data/test15.jpg");
	//cv::Mat image = cv::imread("../../../data/test16.jpg");
	//cv::Mat image = cv::imread("../../../data/test18.jpg");
	cv::Mat image = cv::imread("../../../data/test20.jpg");

	const bool use_original_decoder = false;
	const bool use_default_decoder = false;
	const bool use_original_decoder2 = true;

	// �o�[�R�[�h���o
	if (use_original_decoder) {
		BarcodeDetector decoder;
		cv::Mat result_image = image.clone();
		std::vector<cv::Point> corner = decoder.detect(image);
		for (const auto& point : corner) {
			cv::circle(result_image, point, 3, cv::Scalar(0, 0, 255), -1);
		}

		cv::imshow("result", result_image);

		// �o�[�R�[�h�̃f�R�[�h
		if (corner.size() < 4) {
			std::cout << "barcode detector cannot find barcode" << std::endl;
		} else {
			cv::barcode::BarcodeDetector barcode_decoder;
			std::vector<std::string> decoded_info;
			std::vector<cv::barcode::BarcodeType> decoded_type;
			barcode_decoder.decode(image, corner, decoded_info, decoded_type);
			std::cout << "implement barcode detector result" << std::endl;
			for (uint i = 0; i < decoded_info.size(); i++) {
				std::cout << decoded_info.at(i) << std::endl;
				std::cout << decoded_type.at(i) << std::endl;
			}
		}
	}

	// ��r�p��OpenCV�Ƀf�t�H���g�Ŏ�������Ă�o�[�R�[�h���o
	if (use_default_decoder) {
		// �o�[�R�[�h���o
		cv::Mat gray_image;
		cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
		DefaultBarcodeDetector detector;
		std::vector<cv::Point2f> corners = detector.detect(gray_image);
		if (corners.size() > 0) {
			std::cout << "Find barcodes: " << corners.size() / 4 << std::endl;
		} else {
			std::cout << "Cannot find barcodes" << std::endl;
		}

		// �o�[�R�[�h�f�R�[�h
		std::vector<ArticleNumber> article_numbers = detector.decode(gray_image, corners);
		if (article_numbers.size() > 0) {
			for (const auto& number : article_numbers) {
				std::cout << number.method_type << " " << number.type << " : " << number.article_number << std::endl;
			}
		} else {
			std::cout << "Failed to decode" << std::endl;
		}
	}

	if (use_original_decoder2) {
		BarcodeDetector2 detector;
		detector.detect(image);
	}

	cv::waitKey(0);
	return 0;
}

