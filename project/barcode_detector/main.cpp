#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/barcode.hpp>

#include "BarcodeDetector.h"
#include "BarcodeDetector2.h"
#include "BarcodeDetector3.h"
#include "BarcodeDetector4.h"
#include "BarcodeDetector5.h"
#include "DefaultBarcodeDetector.h"

#include "BarcodeDecoder.h"

int main() {
	//cv::Mat image = cv::imread("../../../data/test.jpg");
	//cv::Mat image = cv::imread("../../../data/test2.jpg");
	//cv::Mat image = cv::imread("../../../data/test3.jpg");
	//cv::Mat image = cv::imread("../../../data/test4.jpg");
	//cv::Mat image = cv::imread("../../../data/test5.jpg");
	//cv::Mat image = cv::imread("../../../data/test6.jpg");
	//cv::Mat image = cv::imread("../../../data/test7.jpg");
	//cv::Mat image = cv::imread("../../../data/test8.jpg");
	//cv::Mat image = cv::imread("../../../data/test9.jpg");
	//cv::Mat image = cv::imread("../../../data/test10.jpg");
	//cv::Mat image = cv::imread("../../../data/test11.jpg");
	//cv::Mat image = cv::imread("../../../data/test12.jpg");
	//cv::Mat image = cv::imread("../../../data/test13.jpg");
	//cv::Mat image = cv::imread("../../../data/test14.jpg");
	//cv::Mat image = cv::imread("../../../data/test15.jpg");
	//cv::Mat image = cv::imread("../../../data/test16.jpg");
	//cv::Mat image = cv::imread("../../../data/test17.jpg");
	//cv::Mat image = cv::imread("../../../data/test18.jpg");
	//cv::Mat image = cv::imread("../../../data/test19.jpg");
	//cv::Mat image = cv::imread("../../../data/test20.jpg");
	//cv::Mat image = cv::imread("../../../data/test21.jpg");
	//cv::Mat image = cv::imread("../../../data/test22.jpg");
	//cv::Mat image = cv::imread("../../../data/test23.jpg");	// 検出できるようにしたい
	cv::Mat image = cv::imread("../../../data/test24.jpg");

	const bool use_original_decoder = false;
	const bool use_default_decoder = false;
	const bool use_original_decoder2 = false;
	const bool use_original_decoder3 = false;
	const bool use_original_decoder4 = false;
	const bool use_original_decoder5 = true;

	// バーコード検出
	if (use_original_decoder) {
		BarcodeDetector decoder;
		cv::Mat result_image = image.clone();
		std::vector<cv::Point> corner = decoder.detect(image);
		for (const auto& point : corner) {
			cv::circle(result_image, point, 3, cv::Scalar(0, 0, 255), -1);
		}

		cv::imshow("result", result_image);

		// バーコードのデコード
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

	// 比較用にOpenCVにデフォルトで実装されてるバーコード検出
	if (use_default_decoder) {
		std::cout << "=============================" << std::endl;
		std::cout << "=  OpenCV default detector  =" << std::endl;
		std::cout << "=============================" << std::endl;

		// バーコード検出
		cv::Mat gray_image;
		cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
		DefaultBarcodeDetector detector;
		std::vector<cv::Point2f> corners = detector.detect(gray_image);
		if (corners.size() > 0) {
			std::cout << "Find barcodes: " << corners.size() / 4 << std::endl;
		} else {
			std::cout << "Cannot find barcodes" << std::endl;
		}

		// バーコードデコード
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
		std::cout << "============================================" << std::endl;
		std::cout << "=  OpenCV default detector with preprocess =" << std::endl;
		std::cout << "============================================" << std::endl;

		// バーコード検出
		BarcodeDetector2 detector;
		std::vector<cv::Point2f> corners = detector.detect(image);
		if (corners.size() > 0) {
			std::cout << "Find barcodes: " << corners.size() / 4 << std::endl;
		} else {
			std::cout << "Cannot find barcodes" << std::endl;
		}

		// バーコードデコード
		cv::Mat gray_image;
		cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
		std::vector<ArticleNumber> article_numbers = detector.decode(gray_image, corners);
		if (article_numbers.size() > 0) {
			for (const auto& number : article_numbers) {
				std::cout << number.method_type << " " << number.type << " : " << number.article_number << std::endl;
			}
		} else {
			std::cout << "Failed to decode" << std::endl;
		}

	}

	if (use_original_decoder3) {
		std::cout << "============================================" << std::endl;
		std::cout << "=  OpenCV default detector with preprocess2 =" << std::endl;
		std::cout << "============================================" << std::endl;

		// バーコード検出
		BarcodeDetector3 detector;
		std::vector<cv::Point2f> corners = detector.detect(image);
		if (corners.size() > 0) {
			std::cout << "Find barcodes: " << corners.size() / 4 << std::endl;
		} else {
			std::cout << "Cannot find barcodes" << std::endl;
		}
	}

	if (use_original_decoder4) {
		// バーコード検出
		BarcodeDetector4 detector;
		std::vector<std::array<cv::Point, 4>> corners = detector.detect(image);
		detector.decode(corners);
		if (corners.size() > 0) {
			std::cout << "Find barcodes: " << corners.size() / 4 << std::endl;
		} else {
			std::cout << "Cannot find barcodes" << std::endl;
		}
	}

	if (use_original_decoder5) {
		BarcodeDetector5 detector;
		const auto& barcode_info = detector.detect(image);
		if (barcode_info.size() > 0) {
			std::cout << "Find barcodes: " << barcode_info.size() << std::endl;
		} else {
			std::cout << "Cannot find barcodes" << std::endl;
		}

		BarcodeDecoder decoder;
		for (const auto& info : barcode_info) {
			decoder.decode(image, std::get<0>(info), std::get<1>(info));
		}
	}

	cv::waitKey(0);
	return 0;
}

