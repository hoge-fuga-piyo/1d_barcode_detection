#include <algorithm>

#include "BarcodeDetector.h"

const double BarcodeDetector::k_th = 2.34;

cv::Mat BarcodeDetector::preprocessing(const cv::Mat& image) const {
  // �K�E�V�A���t�B���^�Ńm�C�Y����
  cv::Mat gaussian_image;
  cv::GaussianBlur(image, gaussian_image, cv::Size(3, 3), 0, 0);
  cv::imshow("gaussian", gaussian_image);

  // ��Â̓�l���œ�l�摜�ɕϊ�
  // TODO: adaptiveThreshold�œ�l�����������ėp����������
  cv::Mat gray_image;
  cv::cvtColor(gaussian_image, gray_image, cv::COLOR_BGR2GRAY);
  cv::Mat binary_image;
  double threshold = cv::threshold(gray_image, binary_image, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  //double threshold = cv::threshold(gray_image, binary_image, 175, 255, cv::THRESH_BINARY_INV);
  //std::cout << "threshold: " << threshold << std::endl;

  return binary_image;
}

std::vector<std::vector<cv::Point>> BarcodeDetector::contoursDetection(const cv::Mat& binary_image) const {
  // �֊s���o
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
  //cv::findContours(binary_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  return contours;
}


bool BarcodeDetector::isBarcodePart(const std::vector<cv::Point>& contour) const {
  uint pc = contour.size();

  double d = getDiagonal(contour);

  //std::cout << (pc <= k_th * d) << ": " << pc << ", " << k_th * d << std::endl;
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

std::vector<cv::Point> BarcodeDetector::cutEdge(const std::vector<cv::Point>& contour) const {
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

  std::vector<cv::Point> cutted_contour;
  if ((max_x_point.x - min_x_point.x) > (max_y_point.y - min_y_point.y)) {
    int length_b = (max_x_point.x - min_x_point.x) / 8;
    int cut_min = min_x_point.x + length_b;
    int cut_max = max_x_point.x - length_b;

    for (const auto& point : contour) {
      if (point.x > cut_min && point.x < cut_max) {
        cutted_contour.push_back(point);
      }
    }
  } else {
    int length_b = (max_y_point.y - min_y_point.y) / 8;
    int cut_min = min_y_point.y + length_b;
    int cut_max = max_y_point.y - length_b;

    for (const auto& point : contour) {
      if (point.y > cut_min && point.y < cut_max) {
        cutted_contour.push_back(point);
      }
    }
  }

  return cutted_contour;
}

std::vector<std::vector<cv::Point>> BarcodeDetector::getLines(const std::vector<cv::Point> contour) const {
  if (contour.size() == 0) {
    return std::vector<std::vector<cv::Point>>();
  }

  auto get_key = [](const cv::Point& point) {
    return std::to_string(point.x) + "_" + std::to_string(point.y);
  };

  // �אڂ���8�ߖT�Ƀ��C�������݂��邩�`�F�b�N�B���݂���ꍇ�͂��̃��C����index��Ԃ�
  auto neighborhood_index = [&](const cv::Point& point, const std::unordered_map<std::string, std::tuple<cv::Point, int>>& map) -> int {
    for (int x_offset = -1; x_offset <= 1; x_offset++) {
      for (int y_offset = -1; y_offset <= 1; y_offset++) {
        if (x_offset == 0 && y_offset == 0) {
          continue;
        }
        std::string key = std::to_string(point.x + x_offset) + "_" + std::to_string(point.y + y_offset);
        if (map.count(key) > 0) {
          return std::get<1>(map.at(key));
        }
      }
    }
    return -1;
  };

  int line_index = 1;
  int new_line_point_num = 0;
  std::unordered_map<std::string, std::tuple<cv::Point, int>> line_map;
  line_map[get_key(contour.at(0))] = std::tuple(contour.at(0), line_index);
  while (true) {
    // �����̐����ɑ�����_���`�F�b�N
    for (const cv::Point& point : contour) {

      // ���ɐ����ɏ����ς݂̃s�N�Z��
      if (line_map.count(get_key(point)) > 0) {
        continue;
      }

      // �܂��������ĂȂ��s�N�Z��
      int point_line_index = neighborhood_index(point, line_map);
      if (point_line_index > 0) {
        line_map[get_key(point)] = std::tuple(point, point_line_index);
        new_line_point_num++;
      }
    }
    
    // �Ώۂ̐����ւ̒ǉ��Ώۂ̓_���������݂��Ȃ�
    if (new_line_point_num == 0) {
      // �S�_�����ɏ������Ă��珈���I��
      bool end_line_divid = true;
      for (const cv::Point& point : contour) {
        if (line_map.count(get_key(point)) == 0) {
          end_line_divid = false;
          break;
        }
      }
      if (end_line_divid) {
        break;
      }

      // �܂������ɏ������ĂȂ��_��V�����_��1�I��ŐV���������ɏ���������
      line_index++;
      for (const cv::Point& point : contour) {
        if (line_map.count(get_key(point)) == 0) {
          line_map[get_key(point)] = std::tuple(point, line_index);
          break;
        }
      }
    }
    new_line_point_num = 0;
  }

  std::vector<std::vector<cv::Point>> lines(line_index, std::vector<cv::Point>());
  for (auto itr = line_map.begin(); itr != line_map.end(); ++itr) {
    std::tuple<cv::Point, int> value = itr->second;
    lines[std::get<1>(value) - 1].push_back(std::get<0>(value));
  }

  return lines;
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

cv::Mat BarcodeDetector::drawLine(const cv::Mat& image, std::vector<cv::Point> line, cv::Scalar color) const {
  cv::Mat dst_image = image.clone();
  for (const auto& point : line) {
    dst_image.at<cv::Vec3b>(point.y, point.x)[0] = color[0];
    dst_image.at<cv::Vec3b>(point.y, point.x)[1] = color[1];
    dst_image.at<cv::Vec3b>(point.y, point.x)[2] = color[2];
  }

  return dst_image;
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
  cv::drawContours(draw_image2, barcode_contours, -1, cv::Scalar(0, 255, 0));
  cv::imshow("contours2", draw_image2);

  std::cout << contours.size() << ", " << barcode_contours.size() << std::endl;

  //
  // Outer contour to line transformation
  //
  std::vector<std::vector<cv::Point>> cutted_contours;
  for (const auto& contour : barcode_contours) {
    // �֊s�̒[���J�b�g����2�{�̐����ɂ���
    const auto cutted_contour = cutEdge(contour);
    cutted_contours.push_back(cutted_contour);
    std::vector<std::vector<cv::Point>> lines = getLines(cutted_contour);

    cv::Mat cutted_draw_image = drawLine(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), cutted_contour, cv::Scalar(255, 0, 255));
    cv::imshow("cutted_single_line", cutted_draw_image);

    for (const auto& line : lines) {
      cv::Mat tmp_draw_image = drawLine(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), line, cv::Scalar(0, 255, 255));
      cv::imshow("line", tmp_draw_image);
      cv::waitKey(0);
    }
  }


  cv::Mat draw_image3 = drawLines(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), cutted_contours, cv::Scalar(255, 255, 0));
  cv::imshow("contours3", draw_image3);
  cv::waitKey(0);
}
