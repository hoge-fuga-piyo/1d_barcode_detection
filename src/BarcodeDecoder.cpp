#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/ximgproc.hpp>
#include <opencv2/barcode.hpp>
#include "BarcodeDecoder.h"

std::array<cv::Point2f, 4> BarcodeDecoder::getWithQuietZone(const cv::Size& image_size, const std::array<cv::Point2f, 4>& corner, const cv::Vec2f& direction) const {
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

        // 画像サイズをはみ出るようなら画像内に収まるよう調整する
        if (dst_corner[0].x < 0.0 || dst_corner[3].x < 0.0) {
            const float min_value = dst_corner[0].x > dst_corner[3].x ? dst_corner[3].x : dst_corner[0].x;
            const cv::Vec2f shift_vector = unit_vector * std::abs(min_value / unit_vector[0]);
            dst_corner[0] = dst_corner[0] - cv::Point2f(shift_vector);
            dst_corner[3] = dst_corner[3] - cv::Point2f(shift_vector);
        }
        if (dst_corner[0].y < 0.0 || dst_corner[3].y < 0.0) {
            const float min_value = dst_corner[0].y > dst_corner[3].y ? dst_corner[3].y : dst_corner[0].y;
            const cv::Vec2f shift_vector = unit_vector * std::abs(min_value / unit_vector[1]);
            dst_corner[0] = dst_corner[0] - cv::Point2f(shift_vector);
            dst_corner[3] = dst_corner[3] - cv::Point2f(shift_vector);
        }
        if (dst_corner[0].x > image_size.width - 1 || dst_corner[3].x > image_size.width - 1) {
            const float max_x = dst_corner[0].x > dst_corner[3].x ? dst_corner[0].x : dst_corner[3].x;
            const float max_diff = max_x - image_size.width;
            const cv::Vec2f shift_vector = unit_vector * std::abs(max_diff / unit_vector[0]);
            dst_corner[0] = dst_corner[0] - cv::Point2f(shift_vector);
            dst_corner[3] = dst_corner[3] - cv::Point2f(shift_vector);
        }
        if (dst_corner[0].y > image_size.height - 1 || dst_corner[3].y > image_size.height - 1) {
            const float max_y = dst_corner[0].y > dst_corner[3].y ? dst_corner[0].y : dst_corner[3].y;
            const float max_diff = max_y - image_size.height;
            const cv::Vec2f shift_vector = unit_vector * std::abs(max_diff / unit_vector[1]);
            dst_corner[0] = dst_corner[0] - cv::Point2f(shift_vector);
            dst_corner[3] = dst_corner[3] - cv::Point2f(shift_vector);
        }
        if (dst_corner[1].x < 0.0 || dst_corner[2].x < 0.0) {
            const float min_value = dst_corner[1].x > dst_corner[2].x ? dst_corner[2].x : dst_corner[1].x;
            const cv::Vec2f shift_vector = unit_vector * std::abs(min_value / unit_vector[0]);
            dst_corner[1] = dst_corner[1] + cv::Point2f(shift_vector);
            dst_corner[2] = dst_corner[2] + cv::Point2f(shift_vector);
        }
        if (dst_corner[1].y < 0.0 || dst_corner[2].y < 0.0) {
            const float min_value = dst_corner[1].y > dst_corner[2].y ? dst_corner[2].y : dst_corner[1].y;
            const cv::Vec2f shift_vector = unit_vector * std::abs(min_value / unit_vector[1]);
            dst_corner[1] = dst_corner[1] + cv::Point2f(shift_vector);
            dst_corner[2] = dst_corner[2] + cv::Point2f(shift_vector);
        }
        if (dst_corner[1].x > image_size.width - 1 || dst_corner[2].x > image_size.width - 1) {
            const float max_x = dst_corner[1].x > dst_corner[2].x ? dst_corner[1].x : dst_corner[2].x;
            const float max_diff = max_x - image_size.width;
            const cv::Vec2f shift_vector = unit_vector * std::abs(max_diff / unit_vector[0]);
            dst_corner[1] = dst_corner[1] + cv::Point2f(shift_vector);
            dst_corner[2] = dst_corner[2] + cv::Point2f(shift_vector);
        }
        if (dst_corner[1].y > image_size.height - 1 || dst_corner[2].y > image_size.height - 1) {
            const float max_y = dst_corner[1].y > dst_corner[2].y ? dst_corner[1].y : dst_corner[2].y;
            const float max_diff = max_y - image_size.height;
            const cv::Vec2f shift_vector = unit_vector * std::abs(max_diff / unit_vector[1]);
            dst_corner[1] = dst_corner[1] + cv::Point2f(shift_vector);
            dst_corner[2] = dst_corner[2] + cv::Point2f(shift_vector);
        }
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

        // 画像サイズをはみ出るようなら画像内に収まるよう調整する
        if (dst_corner[2].x < 0.0 || dst_corner[3].x < 0.0) {
            const float min_value = dst_corner[2].x > dst_corner[3].x ? dst_corner[3].x : dst_corner[2].x;
            const cv::Vec2f shift_vector = unit_vector * std::abs(min_value / unit_vector[0]);
            dst_corner[2] = dst_corner[2] - cv::Point2f(shift_vector);
            dst_corner[3] = dst_corner[3] - cv::Point2f(shift_vector);
        }
        if (dst_corner[2].y < 0.0 || dst_corner[3].y < 0.0) {
            const float min_value = dst_corner[2].y > dst_corner[3].y ? dst_corner[3].y : dst_corner[2].y;
            const cv::Vec2f shift_vector = unit_vector * std::abs(min_value / unit_vector[1]);
            dst_corner[2] = dst_corner[2] - cv::Point2f(shift_vector);
            dst_corner[3] = dst_corner[3] - cv::Point2f(shift_vector);
        }
        if (dst_corner[2].x > image_size.width - 1 || dst_corner[3].x > image_size.width - 1) {
            const float max_x = dst_corner[2].x > dst_corner[3].x ? dst_corner[2].x : dst_corner[3].x;
            const float max_diff = max_x - image_size.width;
            const cv::Vec2f shift_vector = unit_vector * std::abs(max_diff / unit_vector[0]);
            dst_corner[2] = dst_corner[2] - cv::Point2f(shift_vector);
            dst_corner[3] = dst_corner[3] - cv::Point2f(shift_vector);
        }
        if (dst_corner[2].y > image_size.height - 1 || dst_corner[3].y > image_size.height - 1) {
            const float max_y = dst_corner[2].y > dst_corner[3].y ? dst_corner[2].y : dst_corner[3].y;
            const float max_diff = max_y - image_size.height;
            const cv::Vec2f shift_vector = unit_vector * std::abs(max_diff / unit_vector[1]);
            dst_corner[2] = dst_corner[2] - cv::Point2f(shift_vector);
            dst_corner[3] = dst_corner[3] - cv::Point2f(shift_vector);
        }
        if (dst_corner[0].x < 0.0 || dst_corner[1].x < 0.0) {
            const float min_value = dst_corner[0].x > dst_corner[1].x ? dst_corner[1].x : dst_corner[0].x;
            const cv::Vec2f shift_vector = unit_vector * std::abs(min_value / unit_vector[0]);
            dst_corner[0] = dst_corner[0] + cv::Point2f(shift_vector);
            dst_corner[1] = dst_corner[1] + cv::Point2f(shift_vector);
        }
        if (dst_corner[0].y < 0.0 || dst_corner[1].y < 0.0) {
            const float min_value = dst_corner[0].y > dst_corner[1].y ? dst_corner[1].y : dst_corner[0].y;
            const cv::Vec2f shift_vector = unit_vector * std::abs(min_value / unit_vector[1]);
            dst_corner[0] = dst_corner[0] + cv::Point2f(shift_vector);
            dst_corner[1] = dst_corner[1] + cv::Point2f(shift_vector);
        }
        if (dst_corner[0].x > image_size.width - 1 || dst_corner[1].x > image_size.width - 1) {
            const float max_x = dst_corner[0].x > dst_corner[1].x ? dst_corner[0].x : dst_corner[1].x;
            const float max_diff = max_x - image_size.width;
            const cv::Vec2f shift_vector = unit_vector * std::abs(max_diff / unit_vector[0]);
            dst_corner[0] = dst_corner[0] + cv::Point2f(shift_vector);
            dst_corner[1] = dst_corner[1] + cv::Point2f(shift_vector);
        }
        if (dst_corner[0].y > image_size.height - 1 || dst_corner[1].y > image_size.height - 1) {
            const float max_y = dst_corner[0].y > dst_corner[1].y ? dst_corner[0].y : dst_corner[1].y;
            const float max_diff = max_y - image_size.height;
            const cv::Vec2f shift_vector = unit_vector * std::abs(max_diff / unit_vector[1]);
            dst_corner[0] = dst_corner[0] + cv::Point2f(shift_vector);
            dst_corner[1] = dst_corner[1] + cv::Point2f(shift_vector);
        }
    }

    return dst_corner;
}

cv::Mat BarcodeDecoder::cropBarcodeArea(const cv::Mat& image, const std::array<cv::Point2f, 4>& corner, const cv::Vec2f& direction) const {
    // バーコード領域を含んだ矩形領域の切り出し
    double min_x = image.cols;
    double max_x = 0.0;
    double min_y = image.rows;
    double max_y = 0.0;
    for (const auto& point: corner) {
        if (min_x > point.x) {
            min_x = point.x;
        }
        if (max_x < point.x) {
            max_x = point.x;
        }
        if (min_y > point.y) {
            min_y = point.y;
        }
        if (max_y < point.y) {
            max_y = point.y;
        }
    }

    const cv::Rect barcode_rect(cv::Point(min_x, min_y), cv::Size(max_x - min_x, max_y - min_y));
    const cv::Mat barcode_image = image(barcode_rect);

    std::array<cv::Point2f, 4> dst_corner;
    for (int i = 0; i < 4; i++) {
        dst_corner[i] = corner[i] - cv::Point2f(min_x, min_y);
    }

    // 回転させてもバーコードが途切れないように余白を埋める
    const double length = cv::norm(cv::Point(barcode_image.cols, barcode_image.rows));

    cv::Mat barcode_image_with_margin = cv::Mat::zeros(length, length, image.type());
    const int x_margin = (barcode_image_with_margin.cols - barcode_image.cols) / 2.0;
    const int y_margin = (barcode_image_with_margin.rows - barcode_image.rows) / 2.0;
    barcode_image.copyTo(barcode_image_with_margin(cv::Rect(cv::Point(x_margin, y_margin), cv::Size(barcode_image.cols, barcode_image.rows))));
    for (int i = 0; i < 4; i++) {
        dst_corner[i] = dst_corner[i] + cv::Point2f(x_margin, y_margin);
    }

    //cv::line(barcode_image_with_margin, dst_corner[0], dst_corner[1], cv::Scalar(0, 0, 255), 2);
    //cv::line(barcode_image_with_margin, dst_corner[1], dst_corner[2], cv::Scalar(0, 0, 255), 2);
    //cv::line(barcode_image_with_margin, dst_corner[2], dst_corner[3], cv::Scalar(0, 0, 255), 2);
    //cv::line(barcode_image_with_margin, dst_corner[3], dst_corner[0], cv::Scalar(0, 0, 255), 2);
    //cv::imshow("tmp_barcode2", barcode_image_with_margin);

    // 画像を回転
    double cos_theta = cv::Vec2f(0.0, 1.0).dot(direction) / cv::norm(direction);
    if (cos_theta > 1.0) { // 2つのベクトルが完全に同一方向を向いている場合は理論上は1.0になるが、計算誤差の関係で1.0を超える場合がある。その場合、std::acosの結果がnanになってしまうため無理やり丸める
        cos_theta = 1.0;
    }
	const double radian = std::acos(cos_theta);

    const double rotation_angle_radian = radian - (M_PI / 2.0);
    const double rotation_angle_degree = rotation_angle_radian * (180.0 / M_PI);

    const cv::Size2f image_center(barcode_image_with_margin.size().width * 0.5, barcode_image_with_margin.size().height * 0.5);
    const cv::Mat affine_mat = cv::getRotationMatrix2D(image_center, rotation_angle_degree, 1.0);
    cv::Mat rotated_image;
    cv::warpAffine(barcode_image_with_margin, rotated_image, affine_mat, barcode_image_with_margin.size(), cv::INTER_CUBIC);

    // バーコード部分のみ抽出
    const cv::Matx22f rotation_mat(std::cos(rotation_angle_radian), -std::sin(rotation_angle_radian)
        , std::sin(rotation_angle_radian), std::cos(rotation_angle_radian));
    for (int i = 0; i < 4; i++) {
        cv::Mat1d transformed_point = affine_mat * cv::Matx31d(dst_corner[i].x, dst_corner[i].y, 1.0);
        dst_corner[i] = cv::Point2f(transformed_point(0, 0), transformed_point(1, 0));
    }
    //cv::line(rotated_image, dst_corner[0], dst_corner[1], cv::Scalar(0, 0, 255), 2);
    //cv::line(rotated_image, dst_corner[1], dst_corner[2], cv::Scalar(0, 0, 255), 2);
    //cv::line(rotated_image, dst_corner[2], dst_corner[3], cv::Scalar(0, 0, 255), 2);
    //cv::line(rotated_image, dst_corner[3], dst_corner[0], cv::Scalar(0, 0, 255), 2);
    //cv::imshow("tmp_barcode3", rotated_image);

    min_x = rotated_image.cols;
    max_x = 0.0;
    min_y = rotated_image.rows;
    max_y = 0.0;
    for (const auto& point : dst_corner) {
        if (min_x > point.x) {
            min_x = point.x;
        }
        if (max_x < point.x) {
            max_x = point.x;
        }
        if (min_y > point.y) {
            min_y = point.y;
        }
        if (max_y < point.y) {
            max_y = point.y;
        }
    }

    const cv::Mat result_image = rotated_image(cv::Rect(cv::Point(min_x, min_y), cv::Size(max_x - min_x, max_y - min_y))).clone();

    return result_image;
}

std::string BarcodeDecoder::decode(const cv::Mat& image, const std::array<cv::Point2f, 4>& corner, const cv::Vec2f& direction) const {
    bool is_draw_image = true;

    // 余白込の領域を抽出
    const std::array<cv::Point2f, 4> corner_with_quiet_zone = getWithQuietZone(image.size(), corner, direction);

    if (is_draw_image) {
        cv::Mat draw_image = image.clone();

        cv::line(draw_image, corner_with_quiet_zone[0], corner_with_quiet_zone[1], cv::Scalar(0, 0, 255), 2);
        cv::line(draw_image, corner_with_quiet_zone[1], corner_with_quiet_zone[2], cv::Scalar(0, 0, 255), 2);
        cv::line(draw_image, corner_with_quiet_zone[2], corner_with_quiet_zone[3], cv::Scalar(0, 0, 255), 2);
        cv::line(draw_image, corner_with_quiet_zone[3], corner_with_quiet_zone[0], cv::Scalar(0, 0, 255), 2);

        cv::imshow("quiet zone", draw_image);
    }

    // バーコードの領域をx軸に並行な状態に回転する
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    const cv::Mat barcode_image = cropBarcodeArea(gray_image, corner_with_quiet_zone, direction);

    if (is_draw_image) {
        cv::imshow("barcode_area", barcode_image);
    }

    // デコード
    std::vector<std::string> decoded_info;
    std::vector<cv::barcode::BarcodeType> decoded_type;
    cv::barcode::BarcodeDetector detector;
    std::vector<cv::Point2f> corners {
        cv::Point2f(0, barcode_image.rows - 1),
        cv::Point2f(0, 0),
        cv::Point2f(barcode_image.cols - 1, 0),
        cv::Point2f(barcode_image.cols - 1, barcode_image.rows - 1)
    };
    detector.decode(barcode_image, corners, decoded_info, decoded_type);

    for (auto tmp : decoded_info) {
        std::cout << tmp << std::endl;
    }

    //// デコードが失敗した場合は、二値化の結果スキャンライン上のガードバーがくっついてしまっているパターンの場合があるので、スキャンラインをずらして再実行
    //bool is_failure = true;
    //for (const auto& str : decoded_info) {
    //    if (str != "") {
    //        is_failure = false;
    //    }
    //}
    //if (is_failure) {
    //    corners = std::vector<cv::Point2f>{
    //        cv::Point2f(0, (barcode_image.rows - 1) * (1.0/2.0)),
    //        cv::Point2f(0, 0),
    //        cv::Point2f(barcode_image.cols - 1, 0),
    //        cv::Point2f(barcode_image.cols - 1, (barcode_image.rows - 1) * (1.0/2.0))
    //        //cv::Point2f(0, barcode_image.rows - 1),
    //        //cv::Point2f(0, barcode_image.rows * 1.0 / 3.0),
    //        //cv::Point2f(barcode_image.cols - 1, barcode_image.rows * 1.0 / 3.0),
    //        //cv::Point2f(barcode_image.cols - 1, barcode_image.rows - 1)
    //    };
    //    decoded_info.clear();
    //    decoded_type.clear();
    //    detector.decode(barcode_image, corners, decoded_info, decoded_type);

    //    std::cout << "2nd try" << std::endl;
    //    for (auto tmp : decoded_info) {
    //        std::cout << tmp << std::endl;
    //    }

    //    cv::Mat tmp_image = barcode_image.clone();
    //    cv::line(tmp_image, corners[0], corners[1], cv::Scalar(0, 0, 255), 2);
    //    cv::line(tmp_image, corners[1], corners[2], cv::Scalar(0, 0, 255), 2);
    //    cv::line(tmp_image, corners[2], corners[3], cv::Scalar(0, 0, 255), 2);
    //    cv::line(tmp_image, corners[3], corners[0], cv::Scalar(0, 0, 255), 2);
    //    cv::imshow("tmp_barcode", barcode_image);
    //}

    return "";
}