#ifndef _BAR_H_
#define _BAR_H_

#include <vector>
#include <opencv2/opencv.hpp>

class Bar {
private:
  enum class Direction {
    Vertical,
    Horizontal
  };

  bool is_valid;
  std::vector<cv::Point> contour;
  double degree;
  std::array<cv::Point, 4> corner;
  cv::Point center;

  bool isBarcodeElement();
  std::array<cv::Point, 4> detectCorner(const std::vector<cv::Point>& contour) const;
  cv::Point2d detectCenter(const std::vector<cv::Point>& contour) const;
  double getDiagonalLength(const std::vector<cv::Point>& contour) const;
  std::vector<cv::Point> cutEdge(const std::vector<cv::Point>& contour) const;
  std::vector<std::vector<cv::Point>> detectAllLines(const std::vector<cv::Point>& contour) const;
  std::array<std::vector<cv::Point>, 2> detectLines(const std::vector<cv::Point>& contour) const;
  std::tuple<Direction, int> getDirection(const std::vector<cv::Point>& line) const;
  cv::Point2d samplingPoint(const std::vector<cv::Point>& line_part) const;
  std::vector<cv::Point2d> samplingLine(const std::vector<cv::Point>& line) const;
  double lineDegree(const std::vector<cv::Point2d>& sampling_points) const;

  // for DEBUG
  std::array<std::vector<cv::Point>, 2> lines;
  std::array<std::vector<cv::Point2d>, 2> sampling_lines;

public:
  Bar(const std::vector<cv::Point>& contour);

  bool isValid() const;
  void setIsValid(bool is_valid);
  double getDegree() const;
  double getBarLength() const;
  std::array<cv::Point, 4> getCorner() const;
  cv::Point2d getCenter() const;
  void lineFitting();
  std::array<std::vector<cv::Point>, 2> getLines() const;

  // for DEBUG
  std::vector<cv::Point> getContour() const;
  std::array<std::vector<cv::Point2d>, 2> getSamplingLines() const;
};

#endif