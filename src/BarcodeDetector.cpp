#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>
#include "BarcodeDetector.h"

int BarcodeDetector::pdf_interval_t = 4;
double BarcodeDetector::pdf_length_ratio = 0.12;

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

double BarcodeDetector::barcodeAngleDetermine(const std::vector<Bar>& bars) const {
  std::array<int, 180> angle_distribution{};
  std::fill(angle_distribution.begin(), angle_distribution.end(), 0);
  for (const Bar& bar : bars) {
    if (!bar.isValid()) {
      continue;
    }

    const int floored_degree = static_cast<int>(std::floor(bar.getDegree()));
    angle_distribution[floored_degree]++;
  }

  for (int i = 0; i < 180; i++) {
    std::cout << i << ": " << angle_distribution[i] << std::endl;
  }

  // 確率密度が最も大きなところを選択
  int dentist_degree = 0;
  int dentist_degree_count = 0;
  for (int i = 0; i <= 180 - pdf_interval_t; i++) {
    int degree_count = 0;
    for (int j = i; j < i + 4; j++) {
      degree_count += angle_distribution[j];
    }

    if (dentist_degree_count < degree_count) {
      dentist_degree_count = degree_count;
      dentist_degree = i;
    }
  }

  // バーコードの向き
  double m_lambda = 0.0;
  for (int i = dentist_degree; i < dentist_degree + pdf_interval_t; i++) {
    m_lambda += i * angle_distribution[i];
  }
  m_lambda /= (double)dentist_degree_count;

  std::cout << "index: " << dentist_degree << std::endl;
  std::cout << "count: " << dentist_degree_count << std::endl;

  return m_lambda;
}

void BarcodeDetector::updateValidityWithAngle(std::vector<Bar>& bars, double degree) const {
  double min_degree = degree - ((double)pdf_interval_t / 2.0);
  double max_degree = degree + ((double)pdf_interval_t / 2.0);


  for (Bar& bar : bars) {
    if (!bar.isValid()) {
      continue;
    }

    bool is_valid = false;
    if (bar.getDegree() >= min_degree && bar.getDegree() <= max_degree) {
      is_valid = true;
    }

    // 最小の許容角度が0度以下になった場合は+180度して、その角度より大きければ許容角度内とみなす
    if (min_degree < 0.0) {
      double min_degree2 = min_degree + 180.0;
      if (bar.getDegree() >= min_degree2) {
        is_valid = true;
      }
    }

    // 最大の許容角度が180度以上になった場合は-180度して、その角度より小さければ許容角度内とみなす
    if (max_degree > 180.0) {
      double max_degree2 = max_degree - 180.0;
      if (bar.getDegree() <= max_degree2) {
        is_valid = true;
      }
    }

    bar.setIsValid(is_valid);
  }
}

double BarcodeDetector::barcodeLengthDetermine(const std::vector<Bar>& bars) const {
  std::vector<double> length_list;
  for (const auto& bar : bars) {
    if(!bar.isValid()) {
      continue;
    }

    const double length = bar.getBarLength();
    length_list.push_back(length);

    std::cout << "length: " << length << std::endl;
  }

  int max_valid_num = 0;
  double max_valid_length = 0.0;
  for (uint i = 0; i < bars.size(); i++) {
    if (!bars.at(i).isValid()) {
      continue;
    }

    const double base_length = bars.at(i).getBarLength();
    const double offset = base_length * pdf_length_ratio;
    const double min_threshold = bars.at(i).getBarLength() - offset;
    const double max_threshold = bars.at(i).getBarLength() + offset;
    int valid_num = 0;
    for (uint j = 0; j < bars.size(); j++) {
      if (!bars.at(j).isValid()) {
        continue;
      }

      const double length = bars.at(j).getBarLength();
      if (length >= min_threshold && length <= max_threshold) {
        valid_num++;
      }
    }

    if (valid_num > max_valid_num) {
      max_valid_num = valid_num;
      max_valid_length = base_length;
    }
  }

  return max_valid_length;
}

void BarcodeDetector::updateValidityWithLength(std::vector<Bar>& bars, double length) const {
  const double offset = length * pdf_length_ratio;
  const double min_threshold = length - offset;
  const double max_threshold = length + offset;
  for (auto& bar : bars) {
    if (!bar.isValid()) {
      continue;
    }

    const double bar_length = bar.getBarLength();
    if (bar_length < min_threshold || bar_length > max_threshold) {
      bar.setIsValid(false);
    }
  }
}

void BarcodeDetector::removeFewBarDirection(std::vector<Bar>& bars, double degree) const {
  double radian = degree * (M_PI / 180.0);
  cv::Mat rotation_mat = (cv::Mat_<double>(2, 2) << std::cos(radian), -std::sin(radian), std::sin(radian), std::cos(radian));
  cv::Mat barcode_direction_vec = rotation_mat * (cv::Mat_<double>(2, 1) << 1.0, 0.0);
  std::cout << rotation_mat << std::endl;
  std::cout << barcode_direction_vec << std::endl;

  const double degree_diff_threshold = 10.0;
  int max_bar_num = 0;
  std::unordered_map<int, bool> max_bar_index;
  for (uint i = 0; i < bars.size(); i++) {
    if (!bars.at(i).isValid()) {
      continue;
    }

    const cv::Point2d bar_center = bars.at(i).getCenter();
    int bar_num = 0;
    std::unordered_map<int, bool> bar_index;
    for (uint j = 0; j < bars.size(); j++) {
      if (!bars.at(j).isValid()) {
        continue;
      }

      const cv::Point2d tmp_bar_center = bars.at(j).getCenter();
      const cv::Vec2d center_diff_vec = tmp_bar_center - bar_center;
      cv::Vec2d normalized_vec;
      cv::normalize(center_diff_vec, normalized_vec);

      double diff_radian = std::acos(barcode_direction_vec.dot(normalized_vec));
      double diff_degree = diff_radian * (180.0 / M_PI);

      if (diff_degree > 90.0 - degree_diff_threshold && diff_degree < 90.0 + degree_diff_threshold) {
        bar_num++;
        bar_index[j] = true;
      }

      std::cout << "diff radian: " << diff_degree << std::endl;
    }

    if (bar_num > max_bar_num) {
      max_bar_num = bar_num;
      max_bar_index = bar_index;
    }
  }

  for (uint i = 0; i < bars.size(); i++) {
    if (!bars.at(i).isValid()) {
      continue;
    }

    if (max_bar_index.count(i) == 0) {
      bars[i].setIsValid(false);
    }
  }
}

std::array<cv::Point, 4> BarcodeDetector::getBarcodeCorner(std::vector<Bar>& bars) const {
  std::vector<cv::Point> points;

  for (const auto& bar : bars) {
    if (!bar.isValid()) {
      continue;
    }

    std::array<cv::Point, 4> contour = bar.getCorner();
    points.insert(points.end(), contour.begin(), contour.end());
  }

  cv::Point max_x_point = *std::max_element(points.begin(), points.end(), [](const cv::Point& p1, const cv::Point& p2) {
    return p1.x < p2.x;
  });

  cv::Point min_x_point = *std::min_element(points.begin(), points.end(), [](const cv::Point& p1, const cv::Point& p2) {
    return p1.x < p2.x;
  });

  cv::Point max_y_point = *std::max_element(points.begin(), points.end(), [](const cv::Point& p1, const cv::Point& p2) {
    return p1.y < p2.y;
  });

  cv::Point min_y_point = *std::min_element(points.begin(), points.end(), [](const cv::Point& p1, const cv::Point& p2) {
    return p1.y < p2.y;
  });

  std::array<cv::Point, 4> corner;
  corner[0] = cv::Point(min_x_point.x, min_y_point.y);
  corner[1] = cv::Point(max_x_point.x, min_y_point.y);
  corner[2] = cv::Point(min_x_point.x, max_y_point.y);
  corner[3] = cv::Point(max_x_point.x, max_y_point.y);

  return corner;
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

cv::Mat BarcodeDetector::drawBars(const cv::Mat& image, const std::vector<Bar>& bars, cv::Scalar color) const {
  std::vector<std::vector<cv::Point>> contours;
  for (const Bar& bar : bars) {
    if (!bar.isValid()) {
      continue;
    }
    contours.push_back(bar.getContour());
  }

  return drawLines(image, contours, color);
}

std::array<cv::Point, 4> BarcodeDetector::detect(const cv::Mat& image) const {
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
  // 各バーを直線にフィッティングしてバーの角度を計算する
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

  // バーコード全体の向きを決定する
  double barcode_angle_degree = barcodeAngleDetermine(bars);
  std::cout << "barcode angle: " << barcode_angle_degree << std::endl;

  // バーコードの向きに合わないバーは無効にする
  updateValidityWithAngle(bars, barcode_angle_degree);

  cv::Mat draw_image5 = drawBars(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), bars, cv::Scalar(255, 0, 255));
  cv::imshow("angle", draw_image5);

  // バーの長さが外れ値なものは無効にする
  const double bar_length = barcodeLengthDetermine(bars);
  updateValidityWithLength(bars, bar_length);
  std::cout << "fixed length: " << bar_length << std::endl;

  cv::Mat draw_image6 = drawBars(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), bars, cv::Scalar(125, 255, 0));
  cv::imshow("length", draw_image6);

  // バーが平行に一定以上存在する部分のみをバーコードとみなす
  removeFewBarDirection(bars, barcode_angle_degree);

  cv::Mat draw_image7 = drawBars(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), bars, cv::Scalar(0, 255, 125));
  cv::imshow("direction", draw_image7);

  // バーコードの領域を示す端の4点を返す
  std::array<cv::Point, 4> corner = getBarcodeCorner(bars);

  cv::waitKey(0);

  return corner;
}
