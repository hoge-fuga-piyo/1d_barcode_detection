#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>
#include "BarcodeDetector.h"
#include "Bar.h"

cv::Mat BarcodeDetector::preprocessing(const cv::Mat& image) const {
  // ガウシアンフィルタでノイズ除去
  cv::Mat gaussian_image;
  cv::GaussianBlur(image, gaussian_image, cv::Size(3, 3), 0, 0);
  cv::imshow("gaussian", gaussian_image);

  // 大津の二値化で二値画像に変換
  // TODO: adaptiveThresholdで二値化した方が汎用性高いかも
  // TODO: ガウシアンフィルタ -> 二値化 -> 輪郭抽出じゃなくて、DoGフィルタ -> 二値化 -> 輪郭抽出の方がいいかも。要検証
  cv::Mat gray_image;
  cv::cvtColor(gaussian_image, gray_image, cv::COLOR_BGR2GRAY);
  cv::Mat binary_image;
  double threshold = cv::threshold(gray_image, binary_image, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  //double threshold = cv::threshold(gray_image, binary_image, 175, 255, cv::THRESH_BINARY_INV);
  //std::cout << "threshold: " << threshold << std::endl;

  return binary_image;
}

std::vector<std::vector<cv::Point>> BarcodeDetector::contoursDetection(const cv::Mat& binary_image) const {
  // 輪郭抽出
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
  //cv::findContours(binary_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  return contours;
}

cv::Mat BarcodeDetector::drawLines(const cv::Mat& image, std::vector<std::vector<cv::Point>> lines, cv::Scalar color) const {
  cv::Mat dst_image = image.clone();
  for (const auto& line : lines) {
    for (const auto& point : line) {
      dst_image.at<cv::Vec3b>(point.y, point.x)[0] = color[0];
      dst_image.at<cv::Vec3b>(point.y, point.x)[1] = color[1];
      dst_image.at<cv::Vec3b>(point.y, point.x)[2] = color[2];
    }
  }

  return dst_image;
}

cv::Mat BarcodeDetector::drawLines(const cv::Mat& image, std::vector<std::vector<cv::Point2d>> lines, cv::Scalar color) const {
  std::vector<std::vector<cv::Point>> tmp_lines;
  for (const auto& line : lines) {
    std::vector<cv::Point> tmp_line;
    for (const auto& point : line) {
      tmp_line.push_back(point);
    }
    tmp_lines.push_back(tmp_line);
  }

  return drawLines(image, tmp_lines, color);
}

cv::Mat BarcodeDetector::drawLines(const cv::Mat& image, std::vector<std::vector<std::vector<cv::Point>>> lines, cv::Scalar color) const {
  cv::Mat dst_image = image.clone();
  for (const auto& tmp_lines : lines) {
    for (const auto& line : tmp_lines) {
      for (const auto& point : line) {
        dst_image.at<cv::Vec3b>(point.y, point.x)[0] = color[0];
        dst_image.at<cv::Vec3b>(point.y, point.x)[1] = color[1];
        dst_image.at<cv::Vec3b>(point.y, point.x)[2] = color[2];
      }
    }
  }

  return dst_image;
}

cv::Mat BarcodeDetector::drawLines(const cv::Mat& image, std::vector<std::vector<std::vector<cv::Point2d>>> lines, cv::Scalar color) const {
  std::vector<std::vector<std::vector<cv::Point>>> barcode_lines;
  for (const auto& tmp_lines : lines) {
    std::vector<std::vector<cv::Point>> tmp_bar_line;
    for (const auto& line : tmp_lines) {
      std::vector<cv::Point> tmp_line;
      for (const auto& point : line) {
        tmp_line.push_back(point);
      }
      tmp_bar_line.push_back(tmp_line);
    }
    barcode_lines.push_back(tmp_bar_line);
  }

  return drawLines(image, barcode_lines, color);
}

cv::Mat BarcodeDetector::drawLine(const cv::Mat& image, std::vector<cv::Point> line, cv::Scalar color) const {
  cv::Mat dst_image = image.clone();
  for (const auto& point : line) {
    dst_image.at<cv::Vec3b>(point.y, point.x)[0] = color[0];
    dst_image.at<cv::Vec3b>(point.y, point.x)[1] = color[1];
    dst_image.at<cv::Vec3b>(point.y, point.x)[2] = color[2];
  }

  return dst_image;
}

cv::Mat BarcodeDetector::drawLine(const cv::Mat& image, std::vector<cv::Point2d> line, cv::Scalar color) const {
  std::vector<cv::Point> points;
  for (const auto& point : line) {
    points.push_back(point);
  }

  return drawLine(image, points, color);
}

void BarcodeDetector::detect(const cv::Mat& image) const {
  //
  // Barcode Detection Method
  //
  cv::Mat filtered_image = preprocessing(image);
  std::vector<std::vector<cv::Point>> contours = contoursDetection(filtered_image);

  cv::imshow("original", image);
  cv::imshow("filtered", filtered_image);

  cv::Mat draw_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
  cv::drawContours(draw_image, contours, -1, cv::Scalar(0, 0, 255));
  cv::imshow("contours", draw_image);

  std::vector<std::vector<cv::Point>> draw_contours;
  std::vector<Bar> bars;
  for (const auto& contour : contours) {
    Bar bar = Bar(contour);
    bars.push_back(bar);

    // for DEBUG
    if (bar.isValid()) {
      draw_contours.push_back(bar.getContour());
    }
  }

  cv::Mat draw_image2 = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
  cv::drawContours(draw_image2, draw_contours, -1, cv::Scalar(0, 255, 0));
  cv::imshow("contours2", draw_image2);

  //
  // Outer contour to line transformation
  //
  for (Bar& bar : bars) {
    if (!bar.isValid()) {
      continue;
    }
    bar.lineFitting();
  }

  std::vector<std::vector<cv::Point>> draw_lines;
  std::vector<std::vector<cv::Point2d>> draw_sampling_lines;
  for (const Bar& bar : bars) {
    if (!bar.isValid()) {
      continue;
    }
    // for DEBUG
    const auto lines = bar.getLines();
    for (const auto& line : lines) {
      draw_lines.push_back(line);
    }
    const auto sampling_lines = bar.getSamplingLines();
    for (const auto& line : sampling_lines) {
      draw_sampling_lines.push_back(line);
    }
  }
  cv::Mat draw_image3 = drawLines(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), draw_lines, cv::Scalar(255, 255, 0));
  cv::imshow("line", draw_image3);
  cv::Mat draw_image4 = drawLines(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), draw_sampling_lines, cv::Scalar(0, 255, 255));
  cv::imshow("sampling_line", draw_image4);

  // PDF
  std::array<int, 180> degree_distribution{};
  std::fill(degree_distribution.begin(), degree_distribution.end(), 0);
  for (const Bar& bar : bars) {
    if (!bar.isValid()) {
      continue;
    }

    const int floored_degree = static_cast<int>(std::floor(bar.getDegree()));
    degree_distribution[floored_degree]++;
  }

  for (int i = 0; i < 180; i++) {
    std::cout << i << ": " << degree_distribution[i] << std::endl;
  }

  // 確率密度が最も大きなところを選択
  const int t = 4;
  int dentist_degree = 0;
  int dentist_degree_count = 0;
  for (int i = 0; i <= 180 - t; i++) {
    int degree_count = 0;
    for (int j = i; j < i + 4; j++) {
      degree_count += degree_distribution[j];
    }

    if (dentist_degree_count < degree_count) {
      dentist_degree_count = degree_count;
      dentist_degree = i;
    }
  }

  // バーコードの向き
  double m_lambda = 0.0;
  for (int i = dentist_degree; i < dentist_degree + t; i++) {
    m_lambda += i * degree_distribution[i];
  }
  m_lambda /= (double)dentist_degree_count;

  std::cout << "index: " << dentist_degree << std::endl;
  std::cout << "count: " << dentist_degree_count << std::endl;
  std::cout << "m_lambda: " << m_lambda << std::endl;

  cv::waitKey(0);
}
