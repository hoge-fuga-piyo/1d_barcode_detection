#define _USE_MATH_DEFINES
#include <cmath>
#include "Bar5.h"

Bar5::Bar5(const cv::Rect& box, const std::vector<cv::Point>& region) {
	this->box = box;
	this->region = region;

	rotated_rect = cv::minAreaRect(region);
	rotated_rect.points(rotated_rect_corner);

	cv::Size2f size = rotated_rect.size;
	length = size.height > size.width ? size.height : size.width;
}

double Bar5::getAspectRatio() const {
	cv::Size2f size = rotated_rect.size;

	const double long_length = size.height > size.width ? size.height : size.width;
	const double short_length = size.height > size.width ? size.width : size.height;

	return long_length / short_length;
}

std::vector<cv::Point> Bar5::getRegion() const {
	return region;
}

cv::Rect Bar5::getBox() const {
	return box;
}

cv::Point2f Bar5::getCenter() const {
	return rotated_rect.center;
}

cv::Vec2f Bar5::getVerticalVector() const {
	// topLeft to bottomLeft
	const cv::Vec2f vector1 = rotated_rect_corner[0] - rotated_rect_corner[1];

	// topLeft to topRight
	const cv::Vec2f vector2 = rotated_rect_corner[2] - rotated_rect_corner[1];

	// 短い方がバーに垂直なベクトルなはず
	const cv::Vec2f vertical_vector = cv::norm(vector1) > cv::norm(vector2) ? vector2 : vector1;

	return vertical_vector / cv::norm(vertical_vector);
}

cv::Vec2f Bar5::getBarDirectionVector() const {
	// topLeft to bottomLeft
	const cv::Vec2f vector1 = rotated_rect_corner[0] - rotated_rect_corner[1];

	// topLeft to topRight
	const cv::Vec2f vector2 = rotated_rect_corner[2] - rotated_rect_corner[1];

	// 長い方をバーの向きとみなす
	const cv::Vec2f direction_vector = cv::norm(vector1) > cv::norm(vector2) ? vector1 : vector2;

	return direction_vector / cv::norm(direction_vector);
}

double Bar5::getAngleRadian() const {
	// topLeft to bottomLeft
	const cv::Vec2f vector1 = rotated_rect_corner[0] - rotated_rect_corner[1];

	// topLeft to topRight
	const cv::Vec2f vector2 = rotated_rect_corner[2] - rotated_rect_corner[1];

	// バーの方向を示すベクトル
	const cv::Vec2f bar_vector = cv::norm(vector1) > cv::norm(vector2) ? vector1 : vector2;

	// 角度の導出
	const cv::Vec2f base_vector(1.0, 0.0);
	const double cos_theta = base_vector.dot(bar_vector) / (cv::norm(bar_vector) * cv::norm(base_vector));
	const double radian = std::acos(cos_theta);

	return radian;
}

double Bar5::getLength() const {
	return length;
}

