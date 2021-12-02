#define _USE_MATH_DEFINES
#include <cmath>
#include "BarcodeDetector2.h"

BarcodeDetector2::BarcodeDetector2() : detect_num(2), minimum_bar_num(5) {}

cv::Mat BarcodeDetector2::preprocess(const cv::Mat& image) const {
  // グレースケール変換
  cv::Mat gray_image;
  cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

  // DoGフィルタでエッジ抽出
  cv::Mat gaussian_small, gaussian_large;
  cv::GaussianBlur(gray_image, gaussian_small, cv::Size(3, 3), 0, 0);
  cv::GaussianBlur(gray_image, gaussian_large, cv::Size(5, 5), 0, 0);
  cv::Mat dog_image = gaussian_small - gaussian_large;

  // 二値化
  cv::Mat binary_image;
  cv::threshold(dog_image, binary_image, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

  return binary_image;
}

std::vector<std::vector<cv::Point>> BarcodeDetector2::contoursDetection(const cv::Mat& binary_image) const {
  // 輪郭抽出
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

  return contours;
}

std::array<cv::Point, 4> BarcodeDetector2::getMinMaxPoint(const std::vector<cv::Point>& contour) const {
  cv::Point max_x_point = *std::max_element(contour.begin(), contour.end(), [](const cv::Point& p1, const cv::Point& p2) {
    return p1.x < p2.x;
  });

  cv::Point min_x_point = *std::min_element(contour.begin(), contour.end(), [](const cv::Point& p1, const cv::Point& p2) {
    return p1.x < p2.x;
  });

  cv::Point max_y_point = *std::max_element(contour.begin(), contour.end(), [](const cv::Point& p1, const cv::Point& p2) {
    return p1.y < p2.y;
  });

  cv::Point min_y_point = *std::min_element(contour.begin(), contour.end(), [](const cv::Point& p1, const cv::Point& p2) {
    return p1.y < p2.y;
  });

  return std::array<cv::Point, 4>{
    min_x_point, 
    max_x_point, 
    min_y_point, 
    max_y_point
  };
}

std::vector<std::vector<cv::Point>> BarcodeDetector2::removeSmallContours(const std::vector<std::vector<cv::Point>>& contours) const {
  const int threshold = 20;

  std::vector<std::vector<cv::Point>> dst_contours;
  for (const auto& contour : contours) {
    const auto min_max_point = getMinMaxPoint(contour);
    const cv::Point min_x_point = min_max_point.at(0);
    const cv::Point max_x_point = min_max_point.at(1);
    const cv::Point min_y_point = min_max_point.at(2);
    const cv::Point max_y_point = min_max_point.at(3);

    const int length = max_x_point.x - min_x_point.x > max_y_point.y - min_y_point.y ? (max_x_point.x - min_x_point.x) : (max_y_point.y - min_y_point.y);
    if (length > 20) {
      dst_contours.push_back(contour);
    }
  }

  return dst_contours;
}

std::vector<std::vector<cv::Point>> BarcodeDetector2::removeInvalidContours(const std::vector<std::vector<cv::Point>>& contours) const {
  std::vector<std::vector<cv::Point>> dst_contours = removeSmallContours(contours);

  return dst_contours;
}

cv::Point2d BarcodeDetector2::getCenter(const std::vector<cv::Point>& contour) const {
  const auto min_max_point = getMinMaxPoint(contour);
  const cv::Point min_x_point = min_max_point.at(0);
  const cv::Point max_x_point = min_max_point.at(1);
  const cv::Point min_y_point = min_max_point.at(2);
  const cv::Point max_y_point = min_max_point.at(3);

  const double x = (min_x_point.x + max_x_point.x) / 2.0;
  const double y = (min_y_point.y + max_y_point.y) / 2.0;

  return cv::Point2d(x, y);
}

double BarcodeDetector2::getBarLength(const std::vector<cv::Point>& contour) const {
  const auto min_max_point = getMinMaxPoint(contour);
  const cv::Point min_x_point = min_max_point.at(0);
  const cv::Point max_x_point = min_max_point.at(1);
  const cv::Point min_y_point = min_max_point.at(2);
  const cv::Point max_y_point = min_max_point.at(3);

  // バーコードの一部ならほぼ長方形のはずなので、この値は対角線の近似値になるはず
  const cv::Point min_point(min_x_point.x, min_y_point.y);
  const cv::Point max_point(max_x_point.x, max_y_point.y);

  return cv::norm(max_point - min_point);
}

std::vector<std::vector<std::vector<cv::Point>>> BarcodeDetector2::detectParallelContours(const std::vector<std::vector<cv::Point>>& contours) const {
  const double radian_threshold = 5.0 * (M_PI / 180.0);

  std::vector<std::vector<std::vector<cv::Point>>> all_parallel_contours;
  for (uint i = 0; i < contours.size() - 1; i++) {
    const cv::Point2d base_center_point1 = getCenter(contours.at(i));
    const cv::Point2d base_center_point2 = getCenter(contours.at(i + 1));
    const cv::Vec2d base_vector(base_center_point2 - base_center_point1);
    std::vector<std::vector<cv::Point>> parallel_contours{ contours.at(i), contours.at(i + 1) };
    for (uint j = 0; j < contours.size(); j++) {
      if (i == j || i + 1 == j) {
        continue;
      }

      const cv::Vec2d target_vector(getCenter(contours.at(j)) - base_center_point1);
      const double cos_theta = base_vector.dot(target_vector) / (cv::norm(base_vector) * cv::norm(target_vector));
      const double radian = std::acos(cos_theta);
      if (radian < radian_threshold || (180.0 - radian) < radian_threshold) {
        parallel_contours.push_back(contours.at(j));
      }
    }

    if (parallel_contours.size() >= minimum_bar_num) {
      all_parallel_contours.push_back(parallel_contours);
    }
  }

  return all_parallel_contours;
}

std::vector<std::vector<std::vector<cv::Point>>> BarcodeDetector2::detectSameLengthContours(const std::vector<std::vector<std::vector<cv::Point>>>& all_parallel_contours) const {
  const double ratio_threshold = 0.12;

  std::vector<std::vector<std::vector<cv::Point>>> new_parallel_contours;
  for (const auto& parallel_contours : all_parallel_contours) {

    std::vector<std::vector<cv::Point>> max_parallel_contours;
    for (uint i = 0; i < parallel_contours.size(); i++) {
      double base_length = getBarLength(parallel_contours.at(i));
      std::vector<std::vector<cv::Point>> tmp_parallel_contours{ parallel_contours.at(i) };
      for (uint j = 0; j < parallel_contours.size(); j++) {
        if (i == j) {
          continue;
        }
        
        double target_length = getBarLength(parallel_contours.at(j));
        if (std::abs(base_length - target_length) < base_length * ratio_threshold) {
          tmp_parallel_contours.push_back(parallel_contours.at(j));
        }
      }

      if (tmp_parallel_contours.size() > max_parallel_contours.size()) {
        max_parallel_contours = tmp_parallel_contours;
      }
    }

    if (max_parallel_contours.size() >= minimum_bar_num) {
      new_parallel_contours.push_back(max_parallel_contours);
    }
  }

  return new_parallel_contours;
}

cv::Mat BarcodeDetector2::drawContourGroup(const cv::Mat& image, const std::vector<std::vector<std::vector<cv::Point>>>& all_parallel_contours) const {
  cv::Mat dst_image = image.clone();
  for (const auto& parallel_contours : all_parallel_contours) {
    int min_x = 10000;
    int max_x = 0;
    int min_y = 10000;
    int max_y = 0;
    cv::drawContours(dst_image, parallel_contours, -1, cv::Scalar(0, 0, 255));
    for (const auto& contour : parallel_contours) {
      const auto min_max_point = getMinMaxPoint(contour);
      const cv::Point min_x_point = min_max_point.at(0);
      const cv::Point max_x_point = min_max_point.at(1);
      const cv::Point min_y_point = min_max_point.at(2);
      const cv::Point max_y_point = min_max_point.at(3);
      if (min_x > min_x_point.x) {
        min_x = min_x_point.x;
      }
      if (max_x < max_x_point.x) {
        max_x = max_x_point.x;
      }
      if (min_y > min_y_point.y) {
        min_y = min_y_point.y;
      }
      if (max_y < max_y_point.y) {
        max_y = max_y_point.y;
      }
    }
    cv::rectangle(dst_image, cv::Point(min_x, min_y), cv::Point(max_x, max_y), cv::Scalar(0, 255, 0));
  }

  return dst_image;
}

void BarcodeDetector2::detect(const cv::Mat& image) const {
  // 前処理
  cv::Mat filtered_image = preprocess(image);

  // 輪郭抽出
  std::vector<std::vector<cv::Point>> contours = contoursDetection(filtered_image);

  cv::Mat draw_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
  cv::drawContours(draw_image, contours, -1, cv::Scalar(0, 0, 255));
  cv::imshow("draw1", draw_image);

  // 不要な輪郭を削除
  contours = removeInvalidContours(contours);

  draw_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
  cv::drawContours(draw_image, contours, -1, cv::Scalar(0, 0, 255));
  cv::imshow("draw2", draw_image);

  // 平行な輪郭ごとに分ける
  std::vector<std::vector<std::vector<cv::Point>>> all_parallel_contours = detectParallelContours(contours);

  std::vector<std::vector<cv::Point>> tmp_contours;
  for (const auto& parallel_contours : all_parallel_contours) {
    for (const auto& contour : parallel_contours) {
      tmp_contours.push_back(contour);
    }
  }
  draw_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
  cv::drawContours(draw_image, tmp_contours, -1, cv::Scalar(0, 0, 255));
  cv::imshow("draw3", draw_image);

  draw_image = drawContourGroup(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), all_parallel_contours);
  cv::imshow("draw3_2", draw_image);

  // 長さが不揃いな輪郭は捨てる
  std::vector<std::vector<std::vector<cv::Point>>> all_same_length_parallel_contours = detectSameLengthContours(all_parallel_contours);

  tmp_contours.clear();
  for (const auto& parallel_contours : all_same_length_parallel_contours) {
    std::cout << "size: " << parallel_contours.size() << std::endl;
    for (const auto& contour : parallel_contours) {
      tmp_contours.push_back(contour);
    }
  }
  std::cout << "tmp contours:" << tmp_contours.size() << std::endl;
  draw_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
  cv::drawContours(draw_image, tmp_contours, -1, cv::Scalar(0, 0, 255));
  cv::imshow("draw4", draw_image);

  draw_image = drawContourGroup(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), all_same_length_parallel_contours);
  cv::imshow("draw4_2", draw_image);
}
