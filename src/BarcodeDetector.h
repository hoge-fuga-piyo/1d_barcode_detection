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
  static double pdf_length_ratio;

  cv::Mat preprocessing(const cv::Mat& image) const;
  std::vector<std::vector<cv::Point>> contoursDetection(const cv::Mat& binary_image) const;
  double barcodeAngleDetermine(const std::vector<Bar>& bars) const;
  void updateValidityWithAngle(std::vector<Bar>& bars, double degree) const;
  double barcodeLengthDetermine(const std::vector<Bar>& bars) const;
  void updateValidityWithLength(std::vector<Bar>& bars, double length) const;
  void removeFewBarDirection(std::vector<Bar>& bars, double degree) const;
  std::array<cv::Point, 4> getBarcodeCorner(std::vector<Bar>& bars) const;

  // for DEBUG
  cv::Mat drawLines(const cv::Mat& image, std::vector<std::vector<cv::Point>> lines, cv::Scalar color) const;
  cv::Mat drawLines(const cv::Mat& image, std::vector<std::vector<cv::Point2d>> lines, cv::Scalar color) const;
  cv::Mat drawLines(const cv::Mat& image, std::vector<std::vector<std::vector<cv::Point>>> lines, cv::Scalar color) const;
  cv::Mat drawLines(const cv::Mat& image, std::vector<std::vector<std::vector<cv::Point2d>>> lines, cv::Scalar color) const;
  cv::Mat drawLine(const cv::Mat& image, std::vector<cv::Point> line, cv::Scalar color) const;
  cv::Mat drawLine(const cv::Mat& image, std::vector<cv::Point2d> line, cv::Scalar color) const;
  cv::Mat drawBars(const cv::Mat& image, const std::vector<Bar>& bars, cv::Scalar color) const;
public:
  std::array<cv::Point, 4> detect(const cv::Mat& image) const;
};

#endif