#ifndef _BAR5_H_
#define _BAR5_H_

#include <opencv2/opencv.hpp>

class Bar5 {
private:
	cv::Rect box;
	std::vector<cv::Point> region;
	cv::RotatedRect rotated_rect;
	std::array<cv::Point2f, 4> rotated_rect_corner;
	double length;
public:
	Bar5(const cv::Rect& boxes, const std::vector<cv::Point>& regions);

	double getAspectRatio() const;
	std::vector<cv::Point> getRegion() const;
	cv::Rect getBox() const;
	cv::Point2f getCenter() const;
	cv::Vec2f getVerticalVector() const;
	cv::Vec2f getBarDirectionVector() const;
	double getAngleRadian() const;
	double getLength() const;
	std::array<cv::Point2f, 4> getCorner() const;
};

#endif