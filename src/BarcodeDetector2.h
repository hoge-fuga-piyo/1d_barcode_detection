#ifndef _BARCODE_DETECTOR2_
#define _BARCODE_DETECTOR2_

#include <opencv2/opencv.hpp>
#include "ArticleNumber.h"

class BarcodeDetector2 {
private:
  const int detect_num;
  const int minimum_bar_num;

  enum class Direction {
    Vertical,
    Horizontal
  };

  cv::Mat preprocess(const cv::Mat& gray_image) const;
  std::vector<std::vector<cv::Point>> contoursDetection(const cv::Mat& binary_image) const; 
  std::array<cv::Point, 4> getMinMaxPoint(const std::vector<cv::Point>& contour) const;
  std::array<cv::Point, 4> getMinMaxPoint(const std::vector<std::vector<cv::Point>>& contours) const;
  std::vector<std::vector<cv::Point>> removeSmallContours(const std::vector<std::vector<cv::Point>>& contours) const;
  std::vector<std::vector<cv::Point>> removeInvalidContours(const std::vector<std::vector<cv::Point>>& contours) const;
  cv::Point2d getCenter(const std::vector<cv::Point>& contour) const;
  double getBarLength(const std::vector<cv::Point>& contour) const;
  std::vector<std::vector<std::vector<cv::Point>>> detectParallelContours(const std::vector<std::vector<cv::Point>>& contours) const;
  std::vector<std::vector<std::vector<cv::Point>>> detectParallelContours(const std::vector<std::vector<std::vector<cv::Point>>>& all_parallel_contours) const;
  std::vector<std::vector<std::vector<cv::Point>>> detectSameLengthContours(const std::vector<std::vector<std::vector<cv::Point>>>& all_parallel_contours) const;
  std::vector<std::vector<std::vector<cv::Point>>> detectNearContours(const std::vector<std::vector<std::vector<cv::Point>>>& all_parallel_contours) const;
  std::vector<std::vector<std::vector<cv::Point>>> getResultContours(const std::vector<std::vector<std::vector<cv::Point>>>& all_parallel_contours) const;
  cv::Mat createBarcodeImage(const cv::Mat& gray_image, const std::vector<std::vector<std::vector<cv::Point>>>& parallel_contours) const;

  // for DEBUG
  cv::Mat drawContourGroup(const cv::Mat& image, const std::vector<std::vector<std::vector<cv::Point>>>& all_parallel_contours) const;

public:
  BarcodeDetector2();
  std::vector<cv::Point2f> detect(const cv::Mat& image) const;
  std::vector<ArticleNumber> decode(const cv::Mat& image, const std::vector<cv::Point2f>& corners) const;

};

#endif