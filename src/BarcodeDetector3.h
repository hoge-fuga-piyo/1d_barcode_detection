#ifndef _BARCODE_DETECTOR3_H_
#define _BARCODE_DETECTOR3_H_

#include <opencv2/opencv.hpp>
#include "Bar3.h"

class BarcodeDetector3 {
private:
  cv::Mat preprocess(const cv::Mat& gray_image) const;
  std::vector<std::vector<cv::Point>> contoursDetection(const cv::Mat& binary_image) const; 
  std::vector<Bar3> removeSmallContours(int image_long_length, const std::vector<Bar3>& bars) const;
  std::vector<Bar3> removeNotBarRect(const std::vector<Bar3>& bars) const;
  std::vector<Bar3> removeInvalidContours(int image_long_length, const std::vector<Bar3>& bars) const;
  std::vector<std::vector<Bar3>> detectParallelContours(const std::vector<Bar3>& bars) const;

  // for DEBUG
  cv::Mat drawBars(const std::vector<Bar3>& bars, const cv::Mat& image) const;
public:
  std::vector<cv::Point2f> detect(const cv::Mat& image) ;
};

#endif
