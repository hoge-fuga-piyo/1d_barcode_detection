#ifndef _BARCODE_DETECTOR_H_
#define _BARCODE_DETECTOR_H_

#include <opencv2/opencv.hpp>

class BarcodeDetector {
private:
  cv::Mat preprocessing(const cv::Mat& image) const;
  std::vector<std::vector<cv::Point>> contoursDetection(const cv::Mat& binary_image) const;
public:
  void detect(const cv::Mat& image) const;
};

#endif