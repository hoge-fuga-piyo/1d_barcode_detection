#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/barcode.hpp>
#include <chrono>
#include "BarcodeDetector2.h"

BarcodeDetector2::BarcodeDetector2() : detect_num(2), minimum_bar_num(5) {}

void BarcodeDetector2::computeContoursInfo(const std::vector<std::vector<cv::Point>>& contours) {
  auto get_direction = [](const std::array<cv::Point, 4>& min_max_point) {
    const cv::Point min_x_point = min_max_point.at(0);
    const cv::Point max_x_point = min_max_point.at(1);
    const cv::Point min_y_point = min_max_point.at(2);
    const cv::Point max_y_point = min_max_point.at(3);

    if (max_x_point.x - min_x_point.x > max_y_point.y - min_y_point.y) {
      return Direction::Horizontal;
    } else {
      return Direction::Vertical;
    }
  };

  auto get_center = [](const std::array<cv::Point, 4>& min_max_point) {
    const cv::Point min_x_point = min_max_point.at(0);
    const cv::Point max_x_point = min_max_point.at(1);
    const cv::Point min_y_point = min_max_point.at(2);
    const cv::Point max_y_point = min_max_point.at(3);

    const double x = (min_x_point.x + max_x_point.x) / 2.0;
    const double y = (min_y_point.y + max_y_point.y) / 2.0;

    return cv::Point2d(x, y);
  };

  min_max_point.clear();
  contour_directions.clear();
  contour_centers.clear();
  min_max_point.reserve(contours.size());
  contour_directions.reserve(contours.size());
  contour_centers.reserve(contours.size());
  for (const auto& contour : contours) {
    auto tmp_min_max_point = getMinMaxPoint(contour);
    min_max_point.push_back(tmp_min_max_point);
    contour_directions.push_back(get_direction(tmp_min_max_point));
    contour_centers.push_back(get_center(tmp_min_max_point));
  }
}

cv::Mat BarcodeDetector2::preprocess(const cv::Mat& gray_image) const {
  // DoG�t�B���^�ŃG�b�W���o
  cv::Mat gaussian_small, gaussian_large;
  cv::GaussianBlur(gray_image, gaussian_small, cv::Size(3, 3), 0, 0);
  cv::GaussianBlur(gray_image, gaussian_large, cv::Size(5, 5), 0, 0);
  cv::Mat dog_image = gaussian_small - gaussian_large;

  // ��l��
  cv::Mat binary_image;
  cv::threshold(dog_image, binary_image, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

  return binary_image;
}

std::vector<std::vector<cv::Point>> BarcodeDetector2::contoursDetection(const cv::Mat& binary_image) const {
  // �֊s���o
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

  return contours;
}

std::array<cv::Point, 4> BarcodeDetector2::getMinMaxPoint(const std::vector<cv::Point>& contour) const {
  // �ǐ��̊ϓ_����std::max_element, std::min_element�g���������A�v�Z�ʍ팸�̂���for loop�ɂ���
  cv::Point min_x_point = contour.at(0);
  cv::Point max_x_point = contour.at(0);
  cv::Point min_y_point = contour.at(0);
  cv::Point max_y_point = contour.at(0);
  for (uint i = 1; i < contour.size(); i++) {
    if (min_x_point.x > contour.at(i).x) {
      min_x_point = contour.at(i);
    }
    if (max_x_point.x < contour.at(i).x) {
      max_x_point = contour.at(i);
    }
    if (min_y_point.y > contour.at(i).y) {
      min_y_point = contour.at(i);
    }
    if (max_y_point.y < contour.at(i).y) {
      max_y_point = contour.at(i);
    }
  }

  return std::array<cv::Point, 4>{
    min_x_point, 
    max_x_point, 
    min_y_point, 
    max_y_point
  };
}

std::array<cv::Point, 4> BarcodeDetector2::getMinMaxPoint(const std::vector<std::vector<cv::Point>>& contours) const {
  const auto result_min_max_point = getMinMaxPoint(contours.at(0));
  cv::Point result_min_x_point = result_min_max_point.at(0);
  cv::Point result_max_x_point = result_min_max_point.at(1);
  cv::Point result_min_y_point = result_min_max_point.at(2);
  cv::Point result_max_y_point = result_min_max_point.at(3);

  for (uint i = 1; i < contours.size(); i++) {
    const auto min_max_point = getMinMaxPoint(contours.at(i));
    const cv::Point min_x_point = min_max_point.at(0);
    const cv::Point max_x_point = min_max_point.at(1);
    const cv::Point min_y_point = min_max_point.at(2);
    const cv::Point max_y_point = min_max_point.at(3);

    if (result_min_x_point.x > min_x_point.x) {
      result_min_x_point = min_x_point;
    }
    if (result_max_x_point.x < max_x_point.x) {
      result_max_x_point = max_x_point;
    }
    if (result_min_y_point.y > min_y_point.y) {
      result_min_y_point = min_y_point;
    }
    if (result_max_y_point.y < max_y_point.y) {
      result_max_y_point = max_y_point;
    }
  }

  return std::array<cv::Point, 4>{
    result_min_x_point,
    result_max_x_point,
    result_min_y_point,
    result_max_y_point
  };
}

std::vector<std::vector<cv::Point>> BarcodeDetector2::removeSmallContours(int image_large_length, const std::vector<std::vector<cv::Point>>& contours) const {
  const int threshold = image_large_length * 0.02;

  std::vector<std::vector<cv::Point>> dst_contours;
  for (const auto& contour : contours) {
    const auto min_max_point = getMinMaxPoint(contour);
    const cv::Point min_x_point = min_max_point.at(0);
    const cv::Point max_x_point = min_max_point.at(1);
    const cv::Point min_y_point = min_max_point.at(2);
    const cv::Point max_y_point = min_max_point.at(3);

    const int length = max_x_point.x - min_x_point.x > max_y_point.y - min_y_point.y ? (max_x_point.x - min_x_point.x) : (max_y_point.y - min_y_point.y);
    if (length > threshold) {
      dst_contours.push_back(contour);
    }
  }

  return dst_contours;
}

std::vector<std::vector<cv::Point>> BarcodeDetector2::removeLargeContours(int image_large_length, const std::vector<std::vector<cv::Point>>& contours) const {
  const int threshold = image_large_length * 0.9;

  std::vector<std::vector<cv::Point>> dst_contours;
  for (const auto& contour : contours) {
    const auto min_max_point = getMinMaxPoint(contour);
    const cv::Point min_x_point = min_max_point.at(0);
    const cv::Point max_x_point = min_max_point.at(1);
    const cv::Point min_y_point = min_max_point.at(2);
    const cv::Point max_y_point = min_max_point.at(3);

    const int length = max_x_point.x - min_x_point.x > max_y_point.y - min_y_point.y ? (max_x_point.x - min_x_point.x) : (max_y_point.y - min_y_point.y);
    if (length < threshold) {
      dst_contours.push_back(contour);
    }
  }

  return dst_contours;
}

std::vector<std::vector<cv::Point>> BarcodeDetector2::removeInvalidContours(int image_largest_length, const std::vector<std::vector<cv::Point>>& contours) const {
  std::vector<std::vector<cv::Point>> dst_contours = removeSmallContours(image_largest_length, contours);
  dst_contours = removeLargeContours(image_largest_length, dst_contours);

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

  // �o�[�R�[�h�̈ꕔ�Ȃ�قڒ����`�̂͂��Ȃ̂ŁA���̒l�͑Ίp���̋ߎ��l�ɂȂ�͂�
  const cv::Point min_point(min_x_point.x, min_y_point.y);
  const cv::Point max_point(max_x_point.x, max_y_point.y);

  return cv::norm(max_point - min_point);
}

std::vector<std::vector<std::vector<cv::Point>>> BarcodeDetector2::detectParallelContours(const std::vector<std::vector<cv::Point>>& contours) const {
  const double radian_threshold = 10.0 * (M_PI / 180.0);

  std::vector<std::vector<std::vector<cv::Point>>> all_parallel_contours;
  for (uint i = 0; i < contours.size() - 1; i++) {
    const cv::Point2d& base_center_point1 = contour_centers[i];
    const cv::Point2d& base_center_point2 = contour_centers[i + 1];
    cv::Vec2d base_vector(base_center_point2 - base_center_point1);
    const double base_vec_norm = cv::norm(base_vector);
    base_vector = base_vector / base_vec_norm;

    const BarcodeDetector2::Direction direction = contour_directions.at(i);
    std::vector<std::vector<cv::Point>> parallel_contours{ contours.at(i), contours.at(i + 1) };
    for (uint j = 0; j < contours.size(); j++) {
      if (i == j || i + 1 == j) {
        continue;
      }
      if (direction != contour_directions.at(j)) {
        continue;
      }

      const cv::Vec2d target_vector(contour_centers.at(j) - base_center_point1);
      const double cos_theta = base_vector.dot(target_vector) / cv::norm(target_vector);
      const double radian = std::acos(cos_theta);
      if (radian < radian_threshold || (M_PI - radian) < radian_threshold) {
        parallel_contours.push_back(contours.at(j));
      }
    }

    if (parallel_contours.size() >= minimum_bar_num) {
      all_parallel_contours.push_back(parallel_contours);
    }
  }

  return all_parallel_contours;
}

std::vector<std::vector<std::vector<cv::Point>>> BarcodeDetector2::detectParallelContours(const std::vector<std::vector<std::vector<cv::Point>>>& all_parallel_contours) const {
  std::vector<std::vector<std::vector<cv::Point>>> new_parallel_contours;
  for (const auto& parallel_contours : all_parallel_contours) {
    auto result_parallel_contours = detectParallelContours(parallel_contours);
    std::sort(result_parallel_contours.begin(), result_parallel_contours.end(), [](const std::vector<std::vector<cv::Point>>& contours1, const std::vector<std::vector<cv::Point>>& contours2) {
      return contours1.size() > contours2.size();
    });
    
    if (result_parallel_contours.size() > 0) {
      new_parallel_contours.push_back(result_parallel_contours.at(0));
    }
  }

  return new_parallel_contours;
}

std::vector<std::vector<std::vector<cv::Point>>> BarcodeDetector2::detectSameLengthContours(const std::vector<std::vector<std::vector<cv::Point>>>& all_parallel_contours) const {
  const double ratio_threshold = 0.3;
  //const double ratio_threshold = 0.7;

  std::vector<std::vector<std::vector<cv::Point>>> new_parallel_contours;
  for (uint i = 0; i < all_parallel_contours.size(); i++) {
    std::vector<std::tuple<double, uint>> tmp_bar_length_list(all_parallel_contours.at(i).size());
    for (uint j = 0; j < all_parallel_contours.at(i).size(); j++) {
      tmp_bar_length_list[j] = std::tuple<double, uint>(getBarLength(all_parallel_contours.at(i).at(j)), j);
    }

    // �����̑傫�����Ƀ\�[�g
    std::sort(tmp_bar_length_list.begin(), tmp_bar_length_list.end(), [](const auto& obj1, const auto& obj2) {
      return std::get<0>(obj1) > std::get<0>(obj2);
    });

    std::vector<std::vector<cv::Point>> max_contours;
    for (uint j = 0; j < tmp_bar_length_list.size(); j++) {
      double base_length = std::get<0>(tmp_bar_length_list.at(j));

      if (max_contours.size() > tmp_bar_length_list.size() - j) {
        break;
      }

      std::vector<std::vector<cv::Point>> contours{ all_parallel_contours.at(i).at(j) };
      for (int k = j + 1; k < tmp_bar_length_list.size(); k++) {
        double target_length = std::get<0>(tmp_bar_length_list.at(k));
        if (base_length - target_length < base_length * ratio_threshold) {
        //if (target_length / base_length >= ratio_threshold) {
          contours.push_back(all_parallel_contours.at(i).at(k));
        } else {
          break;
        }
      }

      if (max_contours.size() < contours.size()) {
        max_contours = contours;
      }
    }

    if (max_contours.size() > minimum_bar_num) {
      new_parallel_contours.push_back(max_contours);
    }

    //all_bar_length_list[i] = tmp_bar_length_list;
  }

  //std::vector<std::vector<std::vector<cv::Point>>> new_parallel_contours;
  //for (uint i = 0; i < all_parallel_contours.size(); i++) {
  //  const std::vector<std::vector<cv::Point>>& parallel_contours = all_parallel_contours[i];
  //  const std::vector<std::tuple<double, uint>>& bar_length_list = all_bar_length_list[i];

  //  std::vector<std::vector<cv::Point>> max_parallel_contours;
  //  for (uint j = 0; j < parallel_contours.size(); j++) {
  //    double base_length = std::get<0>(bar_length_list.at(j));
  //    std::vector<std::vector<cv::Point>> tmp_parallel_contours{ parallel_contours.at(j) };
  //    tmp_parallel_contours.reserve(parallel_contours.size());
  //    for (uint k = 0; k < parallel_contours.size(); k++) {
  //      if (j == k) {
  //        continue;
  //      }
  //      
  //      double target_length = std::get<0>(bar_length_list.at(k));
  //      if (std::abs(base_length - target_length) < base_length * ratio_threshold) {
  //        tmp_parallel_contours.push_back(parallel_contours.at(k));
  //      }
  //    }

  //    if (tmp_parallel_contours.size() > max_parallel_contours.size()) {
  //      max_parallel_contours = tmp_parallel_contours;
  //    }
  //  }

  //  if (max_parallel_contours.size() >= minimum_bar_num) {
  //    new_parallel_contours.push_back(max_parallel_contours);
  //  }
  //}

  return new_parallel_contours;
}

std::vector<std::vector<std::vector<cv::Point>>> BarcodeDetector2::detectNearContours(const std::vector<std::vector<std::vector<cv::Point>>>& all_parallel_contours) const {
  const double length_ratio_threshold = 0.5;
  const int near_bar_num_threshold = 2;

  std::vector<std::vector<double>> all_bar_length_list(all_parallel_contours.size());
  for (uint i = 0; i < all_parallel_contours.size(); i++) {
    std::vector<double> tmp_bar_length_list(all_parallel_contours.at(i).size());
    for (uint j = 0; j < all_parallel_contours.at(i).size(); j++) {
      tmp_bar_length_list[j] = getBarLength(all_parallel_contours.at(i).at(j));
    }
    all_bar_length_list[i] = tmp_bar_length_list;
  }

  std::vector<std::vector<cv::Point2d>> all_bar_center_list(all_parallel_contours.size());
  for (uint i = 0; i < all_parallel_contours.size(); i++) {
    std::vector<cv::Point2d> tmp_bar_center_list(all_parallel_contours.at(i).size());
    for (uint j = 0; j < all_parallel_contours.at(i).size(); j++) {
      tmp_bar_center_list[j] = getCenter(all_parallel_contours.at(i).at(j));
    }
    all_bar_center_list[i] = tmp_bar_center_list;
  }

  std::vector<std::vector<std::vector<cv::Point>>> new_all_parallel_contours;
  for (uint i = 0; i < all_parallel_contours.size(); i++) {
    const std::vector<std::vector<cv::Point>>& parallel_contours = all_parallel_contours[i];
    const std::vector<double>& bar_length_list = all_bar_length_list[i];
    const std::vector<cv::Point2d>& bar_center_list = all_bar_center_list[i];

    std::vector<std::vector<cv::Point>> new_parallel_contours;
    int near_bar_num = 0;
    for (uint j = 0; j < parallel_contours.size(); j++) {
      const cv::Point2d base_center = bar_center_list.at(j);
      const double base_length = bar_length_list.at(j);
      for (uint k = 0; k < parallel_contours.size(); k++) {
        if (j == k) {
          continue;
        }

        const cv::Point2d target_center = bar_center_list.at(k);
        if (cv::norm(base_center - target_center) < base_length * length_ratio_threshold) {
          near_bar_num++;
          if (near_bar_num >= near_bar_num_threshold) {
            new_parallel_contours.push_back(parallel_contours.at(j));
          }
          break;
        }
      }
    }

    if (new_parallel_contours.size() >= minimum_bar_num) {
      new_all_parallel_contours.push_back(new_parallel_contours);
    }
  }

  return new_all_parallel_contours;
}

std::vector<std::vector<std::vector<cv::Point>>> BarcodeDetector2::getResultContours(const std::vector<std::vector<std::vector<cv::Point>>>& all_parallel_contours) const {
  std::vector<std::vector<std::vector<cv::Point>>> sorted_parallel_contours = all_parallel_contours;
  std::sort(sorted_parallel_contours.begin(), sorted_parallel_contours.end(), [](const std::vector<std::vector<cv::Point>>& contours1, const std::vector<std::vector<cv::Point>>& contours2) {
    return contours1.size() > contours2.size();
  });

  auto overlap = [](const cv::Point& rect1_1, const cv::Point& rect1_2, const cv::Point& rect2_1, const cv::Point& rect2_2) {
    if (std::max(rect1_1.x, rect2_1.x) <= std::min(rect1_2.x, rect2_2.x)
      && std::max(rect1_1.y, rect2_1.y) <= std::min(rect1_2.y, rect2_2.y)) {
      return true;
    }

    return false;
  };

  // �̈�̏d�Ȃ�Ȃ����N�̗֊s�Q�����߂�
  std::vector<std::vector<std::vector<cv::Point>>> result_parallel_contours;
  for (const auto& parallel_contours : sorted_parallel_contours) {
    const auto target_min_max_point = getMinMaxPoint(parallel_contours);
    const cv::Point target_min_x_point = target_min_max_point.at(0);
    const cv::Point target_max_x_point = target_min_max_point.at(1);
    const cv::Point target_min_y_point = target_min_max_point.at(2);
    const cv::Point target_max_y_point = target_min_max_point.at(3);

    const cv::Point rect1_1 = cv::Point(target_min_x_point.x, target_min_y_point.y);
    const cv::Point rect1_2 = cv::Point(target_max_x_point.x, target_max_y_point.y);

    bool is_overlap = false;
    for (const auto& result_contours : result_parallel_contours) {
      const auto min_max_point = getMinMaxPoint(result_contours);
      const cv::Point min_x_point = min_max_point.at(0);
      const cv::Point max_x_point = min_max_point.at(1);
      const cv::Point min_y_point = min_max_point.at(2);
      const cv::Point max_y_point = min_max_point.at(3);

      const cv::Point rect2_1 = cv::Point(min_x_point.x, min_y_point.y);
      const cv::Point rect2_2 = cv::Point(max_x_point.x, max_y_point.y);

      is_overlap = overlap(rect1_1, rect1_2, rect2_1, rect2_2);
    }

    if (!is_overlap) {
      result_parallel_contours.push_back(parallel_contours);
      if (result_parallel_contours.size() == detect_num) {
        break;
      }
    }
  }

  return result_parallel_contours;
}

cv::Mat BarcodeDetector2::createBarcodeImage(const cv::Mat& gray_image, const std::vector<std::vector<std::vector<cv::Point>>>& parallel_contours) const {
  const double margin_ratio_threshold = 0.7;

  cv::Mat dst_image = cv::Mat(gray_image.rows, gray_image.cols, CV_8UC1, cv::Scalar(255));
  for (const auto& contours : parallel_contours) {
      const auto min_max_point = getMinMaxPoint(contours);
      const cv::Point min_x_point = min_max_point.at(0);
      const cv::Point max_x_point = min_max_point.at(1);
      const cv::Point min_y_point = min_max_point.at(2);
      const cv::Point max_y_point = min_max_point.at(3);
      const int x_length = max_x_point.x - min_x_point.x;
      const int y_length = max_y_point.y - min_y_point.y;
      const int x_margin = x_length * margin_ratio_threshold;
      const int y_margin = y_length * margin_ratio_threshold;
      const int x_start = std::max(0, min_x_point.x - x_margin);
      const int y_start = std::max(0, min_y_point.y - y_margin);
      const int x_end = std::min(max_x_point.x + x_margin, gray_image.cols);
      const int y_end = std::min(max_y_point.y + y_margin, gray_image.rows);

      cv::Mat image_part(gray_image, cv::Rect(x_start, y_start, x_end - x_start, y_end - y_start));
      cv::Mat dst_image_part(dst_image, cv::Rect(x_start, y_start, x_end - x_start, y_end - y_start));

      image_part.copyTo(dst_image_part);
  }

  return dst_image;
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

      cv::Point center = getCenter(contour);
      cv::circle(dst_image, center, 3, cv::Scalar(0, 255, 255), -1);
    }
    cv::rectangle(dst_image, cv::Point(min_x, min_y), cv::Point(max_x, max_y), cv::Scalar(0, 255, 0));
  }

  return dst_image;
}

std::vector<cv::Point2f> BarcodeDetector2::detect(const cv::Mat& image) {
  bool draw_image_flag = true;

  // �O����
  // �O���[�X�P�[���ϊ�
  auto start = std::chrono::system_clock::now();
  cv::Mat gray_image;
  cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
  const cv::Mat filtered_image = preprocess(gray_image);
  auto end = std::chrono::system_clock::now();
  std::cout << "preprocess : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  // �֊s���o
  start = std::chrono::system_clock::now();
  contours = contoursDetection(filtered_image);
  end = std::chrono::system_clock::now();
  std::cout << "find contours : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::Mat draw_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    cv::drawContours(draw_image, contours, -1, cv::Scalar(0, 0, 255));
    cv::imshow("draw1", draw_image);
  }

  // �s�v�ȗ֊s���폜
  start = std::chrono::system_clock::now();
  const int image_length = image.rows > image.cols ? image.rows : image.cols;
  contours = removeInvalidContours(image_length, contours);
  end = std::chrono::system_clock::now();
  std::cout << "remove short contours : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::Mat draw_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    cv::drawContours(draw_image, contours, -1, cv::Scalar(0, 0, 255));
    cv::imshow("draw2", draw_image);
  }

  // �֊s�̊e������v�Z
  start = std::chrono::system_clock::now();
  computeContoursInfo(contours);
  end = std::chrono::system_clock::now();
  std::cout << "contours info : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  // ���s�ȗ֊s���Ƃɕ�����
  start = std::chrono::system_clock::now();
  std::vector<std::vector<std::vector<cv::Point>>> all_parallel_contours = detectParallelContours(contours);
  end = std::chrono::system_clock::now();
  std::cout << "parallel contours : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    std::vector<std::vector<cv::Point>> tmp_contours;
    for (const auto& parallel_contours : all_parallel_contours) {
      for (const auto& contour : parallel_contours) {
        tmp_contours.push_back(contour);
      }
    }
    cv::Mat draw_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    cv::drawContours(draw_image, tmp_contours, -1, cv::Scalar(0, 0, 255));
    cv::imshow("draw3", draw_image);

    draw_image = drawContourGroup(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), all_parallel_contours);
    cv::imshow("draw3_2", draw_image);
  }

  // �������s�����ȗ֊s�͎̂Ă�
  start = std::chrono::system_clock::now();
  all_parallel_contours = detectSameLengthContours(all_parallel_contours);
  end = std::chrono::system_clock::now();
  std::cout << "same length : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    std::vector<std::vector<cv::Point>> tmp_contours;
    for (const auto& parallel_contours : all_parallel_contours) {
      for (const auto& contour : parallel_contours) {
        tmp_contours.push_back(contour);
      }
    }
    cv::Mat draw_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    cv::drawContours(draw_image, tmp_contours, -1, cv::Scalar(0, 0, 255));
    cv::imshow("draw4", draw_image);

    draw_image = drawContourGroup(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), all_parallel_contours);
    cv::imshow("draw4_2", draw_image);
  }

  // �ׂ荇���o�[�܂ł̋������󂢂Ă���֊s�͎̂Ă�
  start = std::chrono::system_clock::now();
  all_parallel_contours = detectNearContours(all_parallel_contours);
  end = std::chrono::system_clock::now();
  std::cout << "near contours : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::Mat draw_image = drawContourGroup(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), all_parallel_contours);
    cv::imshow("draw5", draw_image);
  }

  // �ēx���s�ɂȂ��Ă��Ȃ����̂��̂Ă�
  start = std::chrono::system_clock::now();
  all_parallel_contours = detectParallelContours(all_parallel_contours);
  end = std::chrono::system_clock::now();
  std::cout << "parallel contours2 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::Mat draw_image = drawContourGroup(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), all_parallel_contours);
    cv::imshow("draw6", draw_image);
  }

  // �ēx�ׂ荇���o�[�𒲂ׂċ������󂢂Ă�����͎̂̂Ă�
  start = std::chrono::system_clock::now();
  all_parallel_contours = detectNearContours(all_parallel_contours);
  end = std::chrono::system_clock::now();
  std::cout << "near contours2 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::Mat draw_image = drawContourGroup(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), all_parallel_contours);
    cv::imshow("draw7", draw_image);
  }

  // �ēx���s�ɂȂ��Ă��Ȃ����̂��̂Ă�
  start = std::chrono::system_clock::now();
  all_parallel_contours = detectParallelContours(all_parallel_contours);
  end = std::chrono::system_clock::now();
  std::cout << "parallel contours3 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::Mat draw_image = drawContourGroup(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), all_parallel_contours);
    cv::imshow("draw8", draw_image);
  }

  // �o�[�̐����������N�����擾
  start = std::chrono::system_clock::now();
  all_parallel_contours = getResultContours(all_parallel_contours);
  end = std::chrono::system_clock::now();
  std::cout << "result contours : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::Mat draw_image = drawContourGroup(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), all_parallel_contours);
    cv::imshow("draw9", draw_image);
  }

  // �o�[�R�[�h�̗̈�ȊO�𔒓h�肵���摜�����
  start = std::chrono::system_clock::now();
  cv::Mat result_image = createBarcodeImage(gray_image, all_parallel_contours);
  end = std::chrono::system_clock::now();
  std::cout << "white image : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::imshow("result", result_image);
  }

  // �o�[�R�[�h�̌��o
  std::vector<cv::Point2f> barcode_corners;
  cv::barcode::BarcodeDetector detector;
  detector.detect(result_image, barcode_corners);

  if (draw_image_flag) {
    cv::Mat draw_image = image.clone();
    for (const auto& corner : barcode_corners) {
      cv::circle(draw_image, corner, 3, cv::Scalar(0, 0, 255), -1);
    }
    cv::imshow("result_plot", draw_image);
  }

  return barcode_corners;
}

std::vector<ArticleNumber> BarcodeDetector2::decode(const cv::Mat& image, const std::vector<cv::Point2f>& corners) const {
  if (corners.size() < 4 || corners.size() % 4 != 0) {
    std::cout << "Invalid corner num" << std::endl;
    return std::vector<ArticleNumber>();
  }

  std::vector<ArticleNumber> article_numbers;
  cv::barcode::BarcodeDetector detector;

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
 
  return article_numbers;
}
