#define _USE_MATH_DEFINES
#include <cmath>
#include "BarcodeDetector3.h"

cv::Mat BarcodeDetector3::preprocess(const cv::Mat& gray_image) const {
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

std::vector<std::vector<cv::Point>> BarcodeDetector3::contoursDetection(const cv::Mat& binary_image) const {
  // 輪郭抽出
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

  return contours;
}

std::vector<Bar3> BarcodeDetector3::removeSmallContours(int image_long_length, const std::vector<Bar3>& bars) const {
  const double threshold = image_long_length * 0.02;

  std::vector<Bar3> dst_bars;
  dst_bars.reserve(bars.size());
  for (const auto& bar : bars) {
    if (bar.getLongLength() > threshold) {
      dst_bars.push_back(bar);
    }
  }

  return dst_bars;
}

std::vector<Bar3> BarcodeDetector3::removeNotBarRect(const std::vector<Bar3>& bars) const {
  double ratio_threshold = 0.3;

  std::vector<Bar3> dst_bars;
  for (const auto& bar : bars) {
    double long_length = bar.getLongLength();
    double short_length = bar.getShortLength();
    if (short_length / long_length < ratio_threshold) {
      dst_bars.push_back(bar);
    }
  }

  return dst_bars;
}

std::vector<Bar3> BarcodeDetector3::removeInvalidContours(int image_long_length, const std::vector<Bar3>& bars) const {
  std::vector<Bar3> dst_bars = removeSmallContours(image_long_length, bars);
  dst_bars = removeNotBarRect(dst_bars);

  return dst_bars;
}

std::vector<std::vector<Bar3>> BarcodeDetector3::detectParallelContours(const std::vector<Bar3>& bars) const {
  const double radian_threshold = 8.0 * (M_PI / 180.0);

  const cv::Matx22f rotation_mat(std::cos(M_PI / 2.0), -std::sin(M_PI / 2.0),
                                 std::sin(M_PI / 2.0), std::cos(M_PI / 2.0));

  for (uint i = 0; i < bars.size(); i++) {
    const cv::Vec2f base_bar_vec = bars.at(i).getDirection();
    const cv::Vec2f vertical_vec = rotation_mat * base_bar_vec;
    const double base_length = bars.at(i).getLongLength();
    for (uint j = 0; j < bars.size(); j++) {
      if (i == j) {
        continue;
      }

      // そもそもバー同士の方向が異なればスキップ
      const cv::Vec2f target_bar_vec = bars.at(j).getDirection();
      const double cos_theta = base_bar_vec.dot(target_bar_vec);
      const double radian = std::acos(cos_theta);
      if (radian > radian_threshold || (M_PI - radian) > radian_threshold) {
        continue;
      }

      // 明らかにバーの長さが異なればスキップ
      const double target_length = bars.at(j).getLongLength();
      const double ratio = base_length > target_length ? base_length / target_length : target_length / base_length;
      if (ratio > 1.5) {
        continue;
      }

      // 直線と図心の距離を比較
    }
  }

  return std::vector<std::vector<Bar3>>();
}

cv::Mat BarcodeDetector3::drawBars(const std::vector<Bar3>& bars, const cv::Mat& image) const {
  std::vector<std::vector<cv::Point>> contours;
  for (const auto& bar : bars) {
    contours.push_back(bar.getContour());
  }

  cv::Mat dst_image = image.clone();
  cv::drawContours(dst_image, contours, -1, cv::Scalar(0, 0, 255));

  return dst_image;
}

std::vector<cv::Point2f> BarcodeDetector3::detect(const cv::Mat& image) {
  bool draw_image_flag = true;

  // 前処理
  // グレースケール変換
  auto start = std::chrono::system_clock::now();
  cv::Mat gray_image;
  cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
  const cv::Mat filtered_image = preprocess(gray_image);
  auto end = std::chrono::system_clock::now();
  std::cout << "preprocess : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;
  cv::imshow("gray", gray_image);

  // 輪郭抽出
  start = std::chrono::system_clock::now();
  std::vector<std::vector<cv::Point>> contours = contoursDetection(filtered_image);
  end = std::chrono::system_clock::now();
  std::cout << "find contours : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::Mat draw_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    cv::drawContours(draw_image, contours, -1, cv::Scalar(0, 0, 255));
    cv::imshow("draw1", draw_image);
  }

  // 明らかに不要な輪郭は削除
  start = std::chrono::system_clock::now();
  std::vector<Bar3> bars;
  bars.reserve(contours.size());
  for (const auto& contour : contours) {
    bars.push_back(Bar3(contour));
  }
  const int image_length = image.rows > image.cols ? image.rows : image.cols;
  bars = removeInvalidContours(image_length, bars);
  end = std::chrono::system_clock::now();
  std::cout << "remove short contours : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::Mat draw_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    draw_image = drawBars(bars, draw_image);
    cv::imshow("draw2", draw_image);
  }

  // 平行な輪郭毎に分ける
  start = std::chrono::system_clock::now();
  std::vector<std::vector<Bar3>> parallel_bars = detectParallelContours(bars);
  end = std::chrono::system_clock::now();
  std::cout << "parallel contours : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  // 残った輪郭以外の部分は塗りつぶしてしまう

  return std::vector<cv::Point2f>();
}
