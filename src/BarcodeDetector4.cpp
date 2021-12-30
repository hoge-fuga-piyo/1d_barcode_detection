#define _USE_MATH_DEFINES
#include <cmath>
#include "BarcodeDetector4.h"

cv::Mat BarcodeDetector4::derivative(const cv::Mat& gray_image) const {
	const int filter_size = 3;
	cv::Mat sobel_x_image, sobel_y_image;
	cv::Sobel(gray_image, sobel_x_image, CV_16S, 1, 0, filter_size);
	cv::Sobel(gray_image, sobel_y_image, CV_16S, 0, 1, filter_size);

	cv::Mat sobel_abs_x_image, sobel_abs_y_image;
	cv::convertScaleAbs(sobel_x_image, sobel_abs_x_image);
	cv::convertScaleAbs(sobel_y_image, sobel_abs_y_image);

	//cv::imshow("sobel_x", sobel_abs_x_image);
	//cv::imshow("sobel_y", sobel_abs_y_image);

	return sobel_abs_x_image - sobel_abs_y_image;
}

cv::Mat BarcodeDetector4::smoothedMap(const cv::Mat& image) const {
	const int filter_size = 15;
	cv::Mat blur_image;
	//cv::medianBlur(image, blur_image, filter_size);
	cv::boxFilter(image, blur_image, -1, cv::Size(filter_size, filter_size), cv::Point(-1, -1), true);

	return blur_image;
}

cv::Mat BarcodeDetector4::binalize(const cv::Mat& image) const {
	cv::Mat binary_image;
	cv::threshold(image, binary_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	return binary_image;
}

std::vector<std::vector<cv::Point>> BarcodeDetector4::contoursDetection(const cv::Mat& binary_image) const {
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binary_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	//cv::findContours(binary_image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	return contours;
}

std::vector<std::vector<cv::Point>> BarcodeDetector4::barcodeBlobDetection(const cv::Mat& smoothed_map, const cv::Mat& binary_image) const {
	// バーコード領域を抽出
	const int detect_num = 2;
	const double avg_pixel_value_threshold = 60.0;

	std::vector<std::vector<cv::Point>> contours = contoursDetection(binary_image);
	cv::Mat contours_image = cv::Mat::zeros(binary_image.rows, binary_image.cols, CV_8UC3);
	cv::drawContours(contours_image, contours, -1, cv::Scalar(0, 0, 255));
	cv::imshow("contours", contours_image);
	std::cout << "contour num: " << contours.size() << std::endl;

	std::vector<std::tuple<int, double>> each_contour_pixel_avg;
	for (size_t i = 0; i < contours.size(); i++) {
		cv::Mat contour_image = cv::Mat::zeros(binary_image.rows, binary_image.cols, CV_8UC1);
		cv::drawContours(contour_image, contours, i, cv::Scalar(255), -1);

		cv::Mat comvolution_mat;
		cv::bitwise_and(smoothed_map, contour_image, comvolution_mat);

		const double pixel_sum = cv::sum(comvolution_mat)[0];
		const int non_zero_pixel_num = cv::countNonZero(comvolution_mat);
		const double avg_pixel = pixel_sum / non_zero_pixel_num;

		std::cout << "avg pixel: " << avg_pixel << std::endl;

		if (avg_pixel < avg_pixel_value_threshold) {
			continue;
		}

		each_contour_pixel_avg.push_back(std::tuple<int, double>(i, avg_pixel));
	}

	std::sort(each_contour_pixel_avg.begin(), each_contour_pixel_avg.end(), [](auto const& obj1, auto const& obj2) {
		return std::get<1>(obj1) > std::get<1>(obj2);
	});

	std::vector<std::vector<cv::Point>> barcode_contours;
	const size_t barcode_num = each_contour_pixel_avg.size() >= detect_num ? detect_num : each_contour_pixel_avg.size();
	for (size_t i = 0; i < barcode_num; i++) {
		barcode_contours.push_back(contours.at(std::get<0>(each_contour_pixel_avg.at(i))));
	}

	return barcode_contours;
}

std::vector<std::array<cv::Point, 4>> BarcodeDetector4::barcodeRectDetection(const cv::Mat& gray_image, const std::vector<std::vector<cv::Point>>& contours) const {
	cv::Mat draw_image;
	cv::cvtColor(gray_image, draw_image, cv::COLOR_GRAY2BGR);

	std::vector<cv::Rect> barcode_rects;
	std::vector<std::array<cv::Point, 4>> result_corners;
	for (const auto& contour : contours) {
		cv::Rect barcode_rect = cv::boundingRect(contour);
		cv::Point center = (barcode_rect.br() + barcode_rect.tl()) * 0.5;
		cv::Point i_l = cv::Point(barcode_rect.tl().x, center.y);
		cv::Point i_r = cv::Point(barcode_rect.br().x, center.y);
		barcode_rects.push_back(barcode_rect);

		cv::rectangle(draw_image, barcode_rect, cv::Scalar(0, 0, 255));
		cv::circle(draw_image, center, 3, cv::Scalar(0, 255, 0), -1);
		cv::circle(draw_image, i_l, 3, cv::Scalar(0, 255, 0), -1);
		cv::circle(draw_image, i_r, 3, cv::Scalar(0, 255, 0), -1);


		// olの導出
		cv::Point o_l = i_l;
		const int rect_length_half = (barcode_rect.br().x - barcode_rect.tl().x) * 0.5;
		double brightness_sum_ol = gray_image.at<uchar>(i_l.y, i_l.x);
		int brightness_cnt_ol = 1;
		const uchar il_brightness_value = gray_image.at<uchar>(i_l.y, i_l.x);
		for (int i = 1; i < rect_length_half; i++) {
			const double weighted_avg_brightness = 0.85 * brightness_sum_ol / ((i_l.x + i) - i_l.x );
			if (weighted_avg_brightness > gray_image.at<uchar>(i_l.y, i_l.x + i)) {
				cv::circle(draw_image, cv::Point(i_l.x + i - 1, i_l.y), 3, cv::Scalar(125, 0, 125), -1);
				o_l = cv::Point(i_l.x + i - 1, i_l.y);
				break;
			}

			brightness_sum_ol += gray_image.at<uchar>(i_l.y, i_l.x + i);
			brightness_cnt_ol++;
		}

		// orの導出
		cv::Point o_r = i_r;
		double brightness_sum_or = gray_image.at<uchar>(i_r.y, i_r.x);
		double brightness_cnt_or = 1;
		const uchar ir_brightness_value = gray_image.at<uchar>(i_r.y, i_l.x);
		for (int i = 1; i < rect_length_half; i++) {
			const double weighted_avg_brightness = 0.85 * brightness_sum_or / (i_r.x - (i_r.x - i));
			if (weighted_avg_brightness > gray_image.at<uchar>(i_r.y, i_r.x - i)) {
				cv::circle(draw_image, cv::Point(i_r.x - i + 1, i_r.y), 3, cv::Scalar(125, 0, 125), -1);
				o_r = cv::Point(i_r.x - i + 1, i_r.y);
				break;
			}

			brightness_sum_or += gray_image.at<uchar>(i_r.y, i_r.x - i);
			brightness_cnt_or++;
		}

		std::array<cv::Point, 4> corners;
		corners[0] = cv::Point(o_l.x, barcode_rect.tl().y);
		corners[1] = cv::Point(o_l.x, barcode_rect.br().y);
		corners[2] = cv::Point(o_r.x, barcode_rect.br().y);
		corners[3] = cv::Point(o_r.x, barcode_rect.tl().y);
		result_corners.push_back(corners);
	}

	cv::imshow("barcode rect", draw_image);

	return result_corners;
}

double BarcodeDetector4::digitStartPoint(int digit_index, int barcode_start_point, int base_width) const {
	double o = barcode_start_point + (3 * base_width) + (7 * base_width * (digit_index));

	return o;
}

double BarcodeDetector4::base_width(const cv::Point& o_l, const cv::Point& o_r) const {
	double w = (o_r.x - o_l.x) / 95.0;
	return w;
}

void BarcodeDetector4::decode(const std::array<cv::Point, 4>& corner) const {
	const cv::Point center = (corner[0] + corner[2]) * 0.5;
	const cv::Point o_l = cv::Point(corner[0].x, center.y);
	const cv::Point o_r = cv::Point(corner[2].x, center.y);

	std::array<std::array<int, 7>, 10> m = {
		std::array<int, 7>{1, 1, 1, -1, -1, 1, -1},	// m0
		std::array<int, 7>{1, 1, -1, -1, 1, 1, -1},	// m1
		std::array<int, 7>{1, 1, -1, 1, 1, -1, -1},	// m2
		std::array<int, 7>{1, -1, -1, -1, -1, 1, -1},	// m3
		std::array<int, 7>{1, -1, 1, 1, 1, -1, -1},
		std::array<int, 7>{1, -1, -1, 1, 1, 1, -1},
		std::array<int, 7>{1, -1, 1, -1, -1, -1, -1},
		std::array<int, 7>{1, -1, -1, -1, 1, -1, -1},
		std::array<int, 7>{1, -1, -1, 1, -1, -1, -1},
		std::array<int, 7>{1, 1, 1, -1, 1, -1, -1}
	};

	const double w = (o_r.x - o_l.x) / 95.0;
}

std::vector<std::array<cv::Point, 4>> BarcodeDetector4::detect(const cv::Mat& image) const {
	bool is_draw_image = true;

	cv::Mat gray_image;
	cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

	// 勾配を求める
	auto start = std::chrono::system_clock::now();
	cv::Mat derivative_mat = derivative(gray_image);
	auto end = std::chrono::system_clock::now();
	std::cout << "derivative: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

	cv::imshow("sobel_diff", derivative_mat);

	// smoothed mapの作成
	start = std::chrono::system_clock::now();
	cv::Mat smoothed_map = smoothedMap(derivative_mat);
	end = std::chrono::system_clock::now();
	std::cout << "smoothed map: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;
	cv::imshow("smooth", smoothed_map);

	// 二値化
	start = std::chrono::system_clock::now();
	cv::Mat binary_image = binalize(smoothed_map);
	end = std::chrono::system_clock::now();
	std::cout << "binalization: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;
	cv::imshow("binalize", binary_image);

	// バーコード領域の検出
	start = std::chrono::system_clock::now();
	std::vector<std::vector<cv::Point>> barcode_contours = barcodeBlobDetection(smoothed_map, binary_image);
	end = std::chrono::system_clock::now();
	std::cout << "barcode area detection: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

	cv::Mat draw_image = image.clone();
	//cv::cvtColor(smoothed_map, draw_image, cv::COLOR_GRAY2BGR);
	cv::drawContours(draw_image, barcode_contours, -1, cv::Scalar(0, 255, 0), 1);
	cv::imshow("barcode contours", draw_image);

	start = std::chrono::system_clock::now();
	std::vector<std::array<cv::Point, 4>> barcode_corners = barcodeRectDetection(gray_image, barcode_contours);
	end = std::chrono::system_clock::now();
	std::cout << "barcode rect detection: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

	draw_image = image.clone();
	for (const auto& corner : barcode_corners) {
		cv::line(draw_image, corner[0], corner[1], cv::Scalar(0, 0, 255), 1);
		cv::line(draw_image, corner[1], corner[2], cv::Scalar(0, 0, 255), 1);
		cv::line(draw_image, corner[2], corner[3], cv::Scalar(0, 0, 255), 1);
		cv::line(draw_image, corner[3], corner[0], cv::Scalar(0, 0, 255), 1);
	}
	cv::imshow("barcode border", draw_image);

	return barcode_corners;
}

void BarcodeDetector4::decode(const std::vector<std::array<cv::Point, 4>>& corners) const {
	for (const auto& corner : corners) {
		decode(corner);
	}
}
