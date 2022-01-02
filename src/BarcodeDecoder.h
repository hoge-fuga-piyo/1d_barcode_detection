#ifndef _BARCODE_DECODER_H_
#define _BARCODE_DECODER_H_

#include <opencv2/opencv.hpp>

class BarcodeDecoder {
private:
    std::array<cv::Point2f, 4> getWithQuietZone(const std::array<cv::Point2f, 4>& corner, const cv::Vec2f& direction) const;
    cv::Mat cropBarcodeArea(const cv::Mat& image, const std::array<cv::Point2f, 4>& corner, const cv::Vec2f& direction) const;
public:
    std::string decode(const cv::Mat& image, const std::array<cv::Point2f, 4>& corner, const cv::Vec2f& direction) const;
};

#endif