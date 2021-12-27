#include "Bar3.h"

Bar3::Bar3(const std::vector<cv::Point>& contour) {
  this->contour = contour;
  this->center = computeCenter(contour);
  computeRectLength(contour);
}

double Bar3::getLongLength() const {
  return long_length;
}

double Bar3::getShortLength() const {
  return short_length;
}

std::vector<cv::Point> Bar3::getContour() const {
  return contour;
}

cv::Vec2f Bar3::getDirection() const {
  return direction;
}

cv::Point2d Bar3::computeCenter(const std::vector<cv::Point>& contour) const {
  cv::Moments moment = cv::moments(contour, false);
  double mx = moment.m10 / moment.m00;
  double my = moment.m01 / moment.m00;

  return cv::Point2d(mx, my);
}

void Bar3::computeRectLength(const std::vector<cv::Point>& contour) {
  const cv::RotatedRect rect = cv::minAreaRect(contour);
  cv::Point2f corner[4];
  rect.points(corner);

  // bottomLeft to topLeft
  const double line_length1 = cv::norm(corner[1] - corner[0]);

  // topLeft to topRight
  const double line_length2 = cv::norm(corner[2] - corner[1]);

  if (line_length1 > line_length2) {
    short_length = line_length2;
    long_length = line_length1;

    direction = (corner[1] - corner[0]) / line_length1;
  } else {
    short_length = line_length1;
    long_length = line_length2;

    direction = (corner[2] - corner[1]) / line_length2;
  }

}

