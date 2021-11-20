#include <algorithm>

#include "BarcodeDetector.h"

const double BarcodeDetector::k_th = 2.34;

cv::Mat BarcodeDetector::preprocessing(const cv::Mat& image) const {
  // ガウシアンフィルタでノイズ除去
  cv::Mat gaussian_image;
  cv::GaussianBlur(image, gaussian_image, cv::Size(3, 3), 0, 0);
  cv::imshow("gaussian", gaussian_image);

  // 大津の二値化で二値画像に変換
  // TODO: adaptiveThresholdで二値化した方が汎用性高いかも
  cv::Mat gray_image;
  cv::cvtColor(gaussian_image, gray_image, cv::COLOR_BGR2GRAY);
  cv::Mat binary_image;
  double threshold = cv::threshold(gray_image, binary_image, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  //double threshold = cv::threshold(gray_image, binary_image, 175, 255, cv::THRESH_BINARY_INV);
  //std::cout << "threshold: " << threshold << std::endl;

  //cv::imshow("gray", gray_image);
  //cv::imshow("binary", binary_image);
  //cv::waitKey(0);

  return binary_image;
}

std::vector<std::vector<cv::Point>> BarcodeDetector::contoursDetection(const cv::Mat& binary_image) const {
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
  //cv::findContours(binary_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  return contours;
}


bool BarcodeDetector::isBarcodePart(const std::vector<cv::Point>& contour) const {
  uint pc = contour.size();

  double d = getDiagonal(contour);

  std::cout << (pc <= k_th * d) << ": " << pc << ", " << k_th * d << std::endl;
  if (pc <= k_th * d) {
    return true;
  }

  return false;
}

double BarcodeDetector::getDiagonal(const std::vector<cv::Point>& contour) const {
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

  double distance = std::sqrt(std::pow((double)(max_x_point.x - min_x_point.x), 2.0) + std::pow((double)(max_y_point.y - min_y_point.y), 2.0));

  return distance;
}

void BarcodeDetector::detect(const cv::Mat& image) const {
  cv::Mat filtered_image = preprocessing(image);
  std::vector<std::vector<cv::Point>> contours = contoursDetection(filtered_image);

  cv::imshow("original", image);
  cv::imshow("filtered", filtered_image);

  cv::Mat draw_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
  //cv::drawContours(draw_image, contours, -1, cv::Scalar(0, 0, 255), -1);
  cv::drawContours(draw_image, contours, -1, cv::Scalar(0, 0, 255));
  cv::imshow("contours", draw_image);

  std::vector<std::vector<cv::Point>> barcode_contours;
  for (const auto& contour : contours) {
    if (isBarcodePart(contour)) {
      barcode_contours.push_back(contour);
    }
  }

  cv::Mat draw_image2 = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
  cv::drawContours(draw_image2, barcode_contours, -1, cv::Scalar(255, 0, 0));
  cv::imshow("contours2", draw_image2);
  cv::waitKey(0);

  std::cout << contours.size() << ", " << barcode_contours.size() << std::endl;
}
