#ifndef _BARCODE_DETECTOR2_
#define _BARCODE_DETECTOR2_

#include <opencv2/opencv.hpp>

class BarcodeDetector2 {
private:
  const int detect_num;
  const int minimum_bar_num;

  cv::Mat preprocess(const cv::Mat& image) const;
  std::vector<std::vector<cv::Point>> contoursDetection(const cv::Mat& binary_image) const; 
  std::array<cv::Point, 4> getMinMaxPoint(const std::vector<cv::Point>& contour) const;
  std::vector<std::vector<cv::Point>> removeSmallContours(const std::vector<std::vector<cv::Point>>& contours) const;
  std::vector<std::vector<cv::Point>> removeInvalidContours(const std::vector<std::vector<cv::Point>>& contours) const;
  cv::Point2d getCenter(const std::vector<cv::Point>& contour) const;
  double getBarLength(const std::vector<cv::Point>& contour) const;
  std::vector<std::vector<std::vector<cv::Point>>> detectParallelContours(const std::vector<std::vector<cv::Point>>& contours) const;
  std::vector<std::vector<std::vector<cv::Point>>> detectSameLengthContours(const std::vector<std::vector<std::vector<cv::Point>>>& all_parallel_contours) const;

  // for DEBUG
  cv::Mat drawContourGroup(const cv::Mat& image, const std::vector<std::vector<std::vector<cv::Point>>>& all_parallel_contours) const;

public:
  BarcodeDetector2();
  void detect(const cv::Mat& image) const;

};

#endif