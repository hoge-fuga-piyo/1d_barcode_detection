#ifndef _DEFAULT_BARCODE_DETECTOR_H
#define _DEFAULT_BARCODE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "ArticleNumber.h"

class DefaultBarcodeDetector {
private:
  cv::Mat flatten(const cv::Mat& image, const std::vector<cv::Point2f>& corners) const;

public:
  std::vector<cv::Point2f> detect(const cv::Mat& image) const;
  std::vector<ArticleNumber> decode(const cv::Mat& image, const std::vector<cv::Point2f>& corners) const;
};

#endif
