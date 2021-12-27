#ifndef _BARCODE_DETECTOR4_H_
#define _BARCODE_DETECTOR4_H_

#include <opencv2/opencv.hpp>

class BarcodeDetector4 {
private:
	// for detect
	cv::Mat derivative(const cv::Mat& gray_image) const;
	cv::Mat smoothedMap(const cv::Mat& image) const;
	cv::Mat binalize(const cv::Mat& image) const;
	std::vector<std::vector<cv::Point>> contoursDetection(const cv::Mat& binary_image) const;
	std::vector<std::vector<cv::Point>> barcodeBlobDetection(const cv::Mat& smoothed_map, const cv::Mat& binary_image) const;
	std::vector<std::array<cv::Point, 4>> barcodeRectDetection(const cv::Mat& gray_image, const std::vector<std::vector<cv::Point>>& contours) const;

	// for decode
	void decode(const std::array<cv::Point, 4>& corner) const;
public:
	std::vector<std::array<cv::Point, 4>> detect(const cv::Mat& image) const;
	void decode(const std::vector<std::array<cv::Point, 4>>& corners) const;
};

#endif
