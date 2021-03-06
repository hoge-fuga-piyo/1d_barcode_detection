#ifndef _BAR3_H_
#define _BAR3_H_

#include <opencv2/opencv.hpp>

class Bar3 {
private:
  std::vector<cv::Point> contour;
  cv::Point2d center; // 図心
  double short_length, long_length;
  cv::Vec2f direction;  // バーの方向(正規化済み)

  cv::Point2d computeCenter(const std::vector<cv::Point>& contour) const;
  void computeRectLength(const std::vector<cv::Point>& contour);
public:
  Bar3(const std::vector<cv::Point>& contour);
  double getLongLength() const;
  double getShortLength() const;
  std::vector<cv::Point> getContour() const;
  cv::Vec2f getDirection() const;
};

#endif
