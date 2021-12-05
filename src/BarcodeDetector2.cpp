#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/barcode.hpp>
#include <chrono>
#include "BarcodeDetector2.h"

BarcodeDetector2::BarcodeDetector2() : detect_num(2), minimum_bar_num(5) {}

cv::Mat BarcodeDetector2::preprocess(const cv::Mat& gray_image) const {
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
  // 可読性の観点だとstd::max_element, std::min_element使いたいが、計算量削減のためfor loopにする
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
  const double radian_threshold = 10.0 * (M_PI / 180.0);

  auto get_direction = [&](const std::vector<cv::Point>& contour) {
    const auto min_max_point = getMinMaxPoint(contour);
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

  std::vector<std::vector<std::vector<cv::Point>>> all_parallel_contours;
  for (uint i = 0; i < contours.size() - 1; i++) {
    const cv::Point2d base_center_point1 = getCenter(contours.at(i));
    const cv::Point2d base_center_point2 = getCenter(contours.at(i + 1));
    const cv::Vec2d base_vector(base_center_point2 - base_center_point1);

    const BarcodeDetector2::Direction direction = get_direction(contours.at(i));
    //if (direction == Direction::Vertical) {
    //  const double cos_theta = base_vector.dot(cv::Vec2d(1.0, 0.0)) / (cv::norm(base_vector));
    //  const double radian = std::acos(cos_theta);
    //  if ((45.0 * (M_PI / 180.0) < radian && radian < 135.0 * (M_PI / 180.0))) {
    //    continue;
    //  }
    //} else {
    //  const double cos_theta = base_vector.dot(cv::Vec2d(0.0, 1.0)) / (cv::norm(base_vector));
    //  const double radian = std::acos(cos_theta);
    //  if ((45.0 * (M_PI / 180.0) < radian && radian < 135.0 * (M_PI / 180.0))) {
    //    continue;
    //  }
    //}

    std::vector<std::vector<cv::Point>> parallel_contours{ contours.at(i), contours.at(i + 1) };
    for (uint j = 0; j < contours.size(); j++) {
      if (i == j || i + 1 == j) {
        continue;
      }
      if (direction != get_direction(contours.at(j))) {
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
  const double ratio_threshold = 0.7;

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

std::vector<std::vector<std::vector<cv::Point>>> BarcodeDetector2::detectNearContours(const std::vector<std::vector<std::vector<cv::Point>>>& all_parallel_contours) const {
  const double length_ratio_threshold = 0.5;
  const int near_bar_num_threshold = 2;

  std::vector<std::vector<std::vector<cv::Point>>> new_all_parallel_contours;
  for (const auto& parallel_contours : all_parallel_contours) {
    std::vector<std::vector<cv::Point>> new_parallel_contours;
    int near_bar_num = 0;
    for (uint i = 0; i < parallel_contours.size(); i++) {
      const cv::Point2d base_center = getCenter(parallel_contours.at(i));
      const double base_length = getBarLength(parallel_contours.at(i));
      for (uint j = 0; j < parallel_contours.size(); j++) {
        if (i == j) {
          continue;
        }

        const cv::Point2d target_center = getCenter(parallel_contours.at(j));
        if (cv::norm(base_center - target_center) < base_length * length_ratio_threshold) {
          near_bar_num++;
          if (near_bar_num >= near_bar_num_threshold) {
            new_parallel_contours.push_back(parallel_contours.at(i));
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

  // 領域の重ならない上位N個の輪郭群を求める
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

std::vector<cv::Point2f> BarcodeDetector2::detect(const cv::Mat& image) const {
  bool draw_image_flag = false;

  // 前処理
  // グレースケール変換
  auto start = std::chrono::system_clock::now();
  cv::Mat gray_image;
  cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
  cv::Mat filtered_image = preprocess(gray_image);
  auto end = std::chrono::system_clock::now();
  std::cout << "preprocess : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

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

  // 不要な輪郭を削除
  start = std::chrono::system_clock::now();
  contours = removeInvalidContours(contours);
  end = std::chrono::system_clock::now();
  std::cout << "remove short contours : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::Mat draw_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    cv::drawContours(draw_image, contours, -1, cv::Scalar(0, 0, 255));
    cv::imshow("draw2", draw_image);
  }

  // 平行な輪郭ごとに分ける
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

  // 長さが不揃いな輪郭は捨てる
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

  // 隣り合うバーまでの距離が空いている輪郭は捨てる
  start = std::chrono::system_clock::now();
  all_parallel_contours = detectNearContours(all_parallel_contours);
  end = std::chrono::system_clock::now();
  std::cout << "near contours : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::Mat draw_image = drawContourGroup(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), all_parallel_contours);
    cv::imshow("draw5", draw_image);
  }

  // 再度平行になっていないものを捨てる
  start = std::chrono::system_clock::now();
  all_parallel_contours = detectParallelContours(all_parallel_contours);
  end = std::chrono::system_clock::now();
  std::cout << "parallel contours2 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::Mat draw_image = drawContourGroup(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), all_parallel_contours);
    cv::imshow("draw6", draw_image);
  }

  // 再度隣り合うバーを調べて距離が空いているものは捨てる
  start = std::chrono::system_clock::now();
  all_parallel_contours = detectNearContours(all_parallel_contours);
  end = std::chrono::system_clock::now();
  std::cout << "near contours2 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::Mat draw_image = drawContourGroup(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), all_parallel_contours);
    cv::imshow("draw7", draw_image);
  }

  // 再度平行になっていないものを捨てる
  start = std::chrono::system_clock::now();
  all_parallel_contours = detectParallelContours(all_parallel_contours);
  end = std::chrono::system_clock::now();
  std::cout << "parallel contours3 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::Mat draw_image = drawContourGroup(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), all_parallel_contours);
    cv::imshow("draw8", draw_image);
  }

  // バーの数が多い上位N件を取得
  start = std::chrono::system_clock::now();
  all_parallel_contours = getResultContours(all_parallel_contours);
  end = std::chrono::system_clock::now();
  std::cout << "result contours : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::Mat draw_image = drawContourGroup(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), all_parallel_contours);
    cv::imshow("draw9", draw_image);
  }

  // バーコードの領域以外を白塗りした画像を作る
  start = std::chrono::system_clock::now();
  cv::Mat result_image = createBarcodeImage(gray_image, all_parallel_contours);
  end = std::chrono::system_clock::now();
  std::cout << "white image : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " msec" << std::endl;

  if (draw_image_flag) {
    cv::imshow("result", result_image);
  }

  // バーコードの検出
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
