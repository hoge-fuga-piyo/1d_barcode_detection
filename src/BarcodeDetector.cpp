#include "BarcodeDetector.h"

cv::Mat BarcodeDetector::preprocessing(const cv::Mat& image) const {
  // ガウシアンフィルタでノイズ除去
  cv::Mat gaussian_image;
  cv::GaussianBlur(image, gaussian_image, cv::Size(3, 3), 3, 0);

  // 大津の二値化で二値画像に変換
  cv::Mat gray_image;
  cv::cvtColor(gaussian_image, gray_image, cv::COLOR_BGR2GRAY);
  cv::Mat binary_image;
  double threshold = cv::threshold(gray_image, binary_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  //std::cout << "threshold: " << threshold << std::endl;

  //cv::imshow("gray", gray_image);
  //cv::imshow("binary", binary_image);
  //cv::waitKey(0);

  return binary_image;
}

std::vector<std::vector<cv::Point>> BarcodeDetector::contoursDetection(const cv::Mat& binary_image) const {
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

  return contours;
}

void BarcodeDetector::detect(const cv::Mat& image) const {
  cv::Mat filtered_image = preprocessing(image);
  std::vector<std::vector<cv::Point>> contours = contoursDetection(filtered_image);

  cv::Mat draw_image = image.clone();
  cv::drawContours(draw_image, contours, -1, cv::Scalar(0, 0, 255));
  cv::imshow("contours", draw_image);
  cv::waitKey(0);
}
