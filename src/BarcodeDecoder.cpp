#define _USE_MATH_DEFINES
#include <cmath>
#include "BarcodeDecoder.h"

std::array<cv::Point2f, 4> BarcodeDecoder::getWithQuietZone(const std::array<cv::Point2f, 4>& corner, const cv::Vec2f& direction) const {
	const cv::Vec2f vector1 = corner[0] - corner[1]; // topLeft to bottomLeft
	const cv::Vec2f vector2 = corner[2] - corner[1]; // topLeft to topRight

    bool is_reverse1 = false;
	double cos_theta1 = direction.dot(vector1) / (cv::norm(direction) * cv::norm(vector1));
    if (cos_theta1 > 1.0) { // 2つのベクトルが完全に同一方向を向いている場合は理論上は1.0になるが、計算誤差の関係で1.0を超える場合がある。その場合、std::acosの結果がnanになってしまうため無理やり丸める
        cos_theta1 = 1.0;
    }
	double radian1 = std::acos(cos_theta1);
	if (radian1 > M_PI / 2.0) {
		radian1 = M_PI - radian1;
        is_reverse1 = true;
	}

    bool is_reverse2 = false;
	double cos_theta2 = direction.dot(vector2) / (cv::norm(direction) * cv::norm(vector2));
    if (cos_theta2 > 1.0) {
        cos_theta2 = 1.0;
    }
	double radian2 = std::acos(cos_theta2);
	if (radian2 > M_PI / 2.0) {
		radian2 = M_PI - radian2;
        is_reverse2 = true;
	}

    std::array<cv::Point2f, 4> dst_corner;
    if (radian1 < radian2) {
        const double unit_pixel = cv::norm(vector1) / 95.0; // EAN13は95モジュールからなるので、1モジュールあたりの長さを求める
        cv::Vec2f unit_vector = vector1 / cv::norm(vector1);
        if (is_reverse1) {
            unit_vector = -unit_vector;
        }

        dst_corner[0] = corner[0] + cv::Point2f(unit_vector * unit_pixel * 9); // bottomLeft
        dst_corner[1] = corner[1] - cv::Point2f(unit_vector * unit_pixel * 9); // topLeft
        dst_corner[2] = corner[2] - cv::Point2f(unit_vector * unit_pixel * 9); // topRight
        dst_corner[3] = corner[3] + cv::Point2f(unit_vector * unit_pixel * 9); // bottomRight
    } else {
        const double unit_pixel = cv::norm(vector2) / 95.0; // EAN13は95モジュールからなるので、1モジュールあたりの長さを求める
        cv::Vec2f unit_vector = vector2 / cv::norm(vector2);
        if (is_reverse2) {
            unit_vector = -unit_vector;
        }

        dst_corner[0] = corner[0] - cv::Point2f(unit_vector * unit_pixel * 9); // bottomLeft
        dst_corner[1] = corner[1] - cv::Point2f(unit_vector * unit_pixel * 9); // topLeft
        dst_corner[2] = corner[2] + cv::Point2f(unit_vector * unit_pixel * 9); // topRight
        dst_corner[3] = corner[3] + cv::Point2f(unit_vector * unit_pixel * 9); // bottomRight
    }

    return dst_corner;
}

std::string BarcodeDecoder::decode(const cv::Mat& image, const std::array<cv::Point2f, 4>& corners, const cv::Vec2f& direction) const {
    bool is_draw_image = true;

    const std::array<cv::Point2f, 4> corner_with_quiet_zone = getWithQuietZone(corners, direction);

    if (is_draw_image) {
        cv::Mat draw_image = image.clone();

        cv::line(draw_image, corner_with_quiet_zone[0], corner_with_quiet_zone[1], cv::Scalar(0, 0, 255), 2);
        cv::line(draw_image, corner_with_quiet_zone[1], corner_with_quiet_zone[2], cv::Scalar(0, 0, 255), 2);
        cv::line(draw_image, corner_with_quiet_zone[2], corner_with_quiet_zone[3], cv::Scalar(0, 0, 255), 2);
        cv::line(draw_image, corner_with_quiet_zone[3], corner_with_quiet_zone[0], cv::Scalar(0, 0, 255), 2);

        cv::imshow("quiet zone", draw_image);
    }

    return "";
}