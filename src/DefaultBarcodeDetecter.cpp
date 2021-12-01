#include "DefaultBarcodeDetector.h"
#include <opencv2/barcode.hpp>

cv::Mat DefaultBarcodeDetector::flatten(const cv::Mat& image, const std::vector<cv::Point2f>& corners) const {
  const int clahe_filter_size = 8;

  if (image.cols < clahe_filter_size || image.rows < clahe_filter_size) {
    return image;
  }

  cv::Mat dst_image = image.clone();

	// �o�[�R�[�h�����̉摜��؂���
	cv::Point min_x_point = *std::min_element(corners.begin(), corners.end(), [](const cv::Point& p1, const cv::Point& p2) {
	  return p1.x < p2.x;
	});
	cv::Point max_x_point = *std::max_element(corners.begin(), corners.end(), [](const cv::Point& p1, const cv::Point& p2) {
	  return p1.x < p2.x;
	});
	cv::Point min_y_point = *std::min_element(corners.begin(), corners.end(), [](const cv::Point& p1, const cv::Point& p2) {
	  return p1.y < p2.y;
	});
	cv::Point max_y_point = *std::max_element(corners.begin(), corners.end(), [](const cv::Point& p1, const cv::Point& p2) {
	  return p1.y < p2.y;
	});
	cv::Mat barcode_part(dst_image, cv::Rect(min_x_point.x, min_y_point.y, (max_x_point.x - min_x_point.x), (max_y_point.y - min_y_point.y)));

  // �R���g���X�g���R��
  cv::Mat clahe_image;
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0, cv::Size(clahe_filter_size, clahe_filter_size));
  clahe->apply(barcode_part, clahe_image);

  // DoG�t�B���^��p���ĉ摜��sharpness�����s��
  // DoG�t�B���^�̓����I�ɂ�����x�R���g���X�g�����R�������͂�
  cv::Mat gaussian_small, gaussian_large;
  cv::GaussianBlur(clahe_image, gaussian_small, cv::Size(3, 3), 0.0, 0.0);
  cv::GaussianBlur(clahe_image, gaussian_large, cv::Size(5, 5), 0.0, 0.0);
  const double p = 5.0;
  cv::Mat dog_sharp_image = (1.0 + p) * gaussian_small - p * gaussian_large;

  dog_sharp_image.copyTo(barcode_part);

  //cv::imshow("barcode", barcode_part);
  cv::imshow("clahe2", clahe_image);
  cv::imshow("dog", dog_sharp_image);
  //cv::imshow("result", dst_image);

  return dst_image;
}

std::vector<cv::Point2f> DefaultBarcodeDetector::detect(const cv::Mat& image) const {
  std::vector<cv::Point2f> corners;
  cv::barcode::BarcodeDetector detector;
  detector.detect(image, corners);

  return corners;
}

std::vector<ArticleNumber> DefaultBarcodeDetector::decode(const cv::Mat& image, const std::vector<cv::Point2f>& corners) const {
  if (corners.size() < 4 || corners.size() % 4 != 0) {
    std::cout << "Invalid corner num" << std::endl;
    return std::vector<ArticleNumber>();
  }

  std::vector<ArticleNumber> article_numbers;
  cv::barcode::BarcodeDetector detector;
  //
  // �摜�̉��H�Ȃ��Ńf�R�[�h
  //
  std::vector<std::string> decoded_info;
  std::vector<cv::barcode::BarcodeType> decoded_type;
  detector.decode(image, corners, decoded_info, decoded_type);
  for (uint i = 0; i < decoded_info.size(); i++) {
    if (decoded_type.at(i) != cv::barcode::BarcodeType::NONE) {
      ArticleNumber article_number;
      article_number.article_number = decoded_info.at(i);
      article_number.type = decoded_type.at(i);
      article_number.method_type = 0;
      article_numbers.push_back(article_number);
    }
  }
  
  if (article_numbers.size() > 0) {
    return article_numbers;
  }

  //
  // �R���g���X�g�̕��R�������{��ɍēx�f�R�[�h�����݂�
  //
  cv::Mat flatten_image = image.clone();
  const int barcode_num = corners.size() / 4;
  for (int i = 0; i < barcode_num; i++) {
    std::vector<cv::Point2f> corner{
      corners.at(i * 4 + 0),
      corners.at(i * 4 + 1),
      corners.at(i * 4 + 2),
      corners.at(i * 4 + 3)
    };

    const cv::Mat tmp_image = flatten(flatten_image, corner);
    flatten_image = tmp_image.clone();
  }
  detector.decode(flatten_image, corners, decoded_info, decoded_type);
  for (uint i = 0; i < decoded_info.size(); i++) {
    if (decoded_type.at(i) != cv::barcode::BarcodeType::NONE) {
      ArticleNumber article_number;
      article_number.article_number = decoded_info.at(i);
      article_number.type = decoded_type.at(i);
      article_number.method_type = 1;
      article_numbers.push_back(article_number);
    }
  }
 
  return article_numbers;
}

