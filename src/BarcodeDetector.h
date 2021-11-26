#ifndef _BARCODE_DETECTOR_H_
#define _BARCODE_DETECTOR_H_

#include <opencv2/opencv.hpp>
#include "Bar.h"

class BarcodeDetector {
private:
  enum class Direction {
    Vertical,
    Horizontal
  };

  static int pdf_interval_t;

  cv::Mat preprocessing(const cv::Mat& image) const;
  std::vector<std::vector<cv::Point>> contoursDetection(const cv::Mat& binary_image) const;
  double barcodeAngleDetermine(const std::vector<Bar>& bars) const;
  void updateValidityWithAngle(std::vector<Bar>& bars, double degree) const;

  // for DEBUG
  cv::Mat drawLines(const cv::Mat& image, std::vector<std::vector<cv::Point>> lines, cv::Scalar color) const;
  cv::Mat drawLines(const cv::Mat& image, std::vector<std::vector<cv::Point2d>> lines, cv::Scalar color) const;
  cv::Mat drawLines(const cv::Mat& image, std::vector<std::vector<std::vector<cv::Point>>> lines, cv::Scalar color) const;
  cv::Mat drawLines(const cv::Mat& image, std::vector<std::vector<std::vector<cv::Point2d>>> lines, cv::Scalar color) const;
  cv::Mat drawLine(const cv::Mat& image, std::vector<cv::Point> line, cv::Scalar color) const;
  cv::Mat drawLine(const cv::Mat& image, std::vector<cv::Point2d> line, cv::Scalar color) const;
public:
  void detect(const cv::Mat& image) const;
};

#endif