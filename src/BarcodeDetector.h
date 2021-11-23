#ifndef _BARCODE_DETECTOR_H_
#define _BARCODE_DETECTOR_H_

#include <opencv2/opencv.hpp>

class BarcodeDetector {
private:
  enum class Direction {
    Vertical,
    Horizontal
  };

  cv::Mat preprocessing(const cv::Mat& image) const;
  std::vector<std::vector<cv::Point>> contoursDetection(const cv::Mat& binary_image) const;
  bool isBarcodePart(const std::vector<cv::Point>& contour) const;
  double getDiagonal(const std::vector<cv::Point>& contour) const;
  std::vector<cv::Point> cutEdge(const std::vector<cv::Point>& contour) const;
  std::vector<std::vector<cv::Point>> getLines(const std::vector<cv::Point>& contour) const;
  std::vector<std::vector<cv::Point>> getBarcodeCandidateLines(const std::vector<cv::Point>& contour) const;
  std::vector<cv::Point2d> samplingLine(const std::vector<cv::Point>& line) const;
  cv::Point2d sampling(const std::vector<cv::Point>& line_part) const;
  std::tuple<Direction, int> getDirection(const std::vector<cv::Point>& line) const;

  // for DEBUG
  cv::Mat drawLines(const cv::Mat& image, std::vector<std::vector<cv::Point>> lines, cv::Scalar color) const;
  cv::Mat drawLines(const cv::Mat& image, std::vector<std::vector<cv::Point2d>> lines, cv::Scalar color) const;
  cv::Mat drawLines(const cv::Mat& image, std::vector<std::vector<std::vector<cv::Point>>> lines, cv::Scalar color) const;
  cv::Mat drawLine(const cv::Mat& image, std::vector<cv::Point> line, cv::Scalar color) const;
  cv::Mat drawLine(const cv::Mat& image, std::vector<cv::Point2d> line, cv::Scalar color) const;
public:
  void detect(const cv::Mat& image) const;
};

#endif