#ifndef _BARCODE_DETECTOR5_H_
#define _BARCODE_DETECTOR5_H_

#include <opencv2/opencv.hpp>
#include "Bar5.h"

class BarcodeDetector5 {
private:
	class TreeElement {
	public:
		int parent_index;
		cv::Point2f point;
		std::vector<int> indexes;
	};

	const int min_barcode_bar_num;

	std::tuple<std::vector<cv::Rect>, std::vector<std::vector<cv::Point>>> detectMserRegions(const cv::Mat& gray_image) const;
	std::vector<Bar5> removeInvalidRegions(const std::vector<Bar5>& bars) const;
	cv::Point2f conputeRepresentationPoint(const Bar5& bar) const;
	std::vector<std::vector<Bar5>> clustering(const std::vector<Bar5>& bars) const;
	std::vector<std::vector<Bar5>> removeOutlierPositionBars(const std::vector<std::vector<Bar5>>& bars) const;
	std::vector<std::vector<Bar5>> removeOutlierLengthBars(const std::vector<std::vector<Bar5>>& bars) const;
	std::vector<std::vector<Bar5>> removeInvalidBars(const std::vector<std::vector<Bar5>>& bars) const;
	std::vector<std::array<cv::Point2f, 4>> mergeBars(const std::vector<std::vector<Bar5>>& bars) const;
public:
	BarcodeDetector5();
	void detect(const cv::Mat& image) const;
};

#endif
