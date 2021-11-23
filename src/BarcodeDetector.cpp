#include <algorithm>
#include "BarcodeDetector.h"

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

  return binary_image;
}

std::vector<std::vector<cv::Point>> BarcodeDetector::contoursDetection(const cv::Mat& binary_image) const {
  // 輪郭抽出
  std::vector<std::vector<cv::Point>> contours;
  //cv::findContours(binary_image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
  cv::findContours(binary_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  return contours;
}


bool BarcodeDetector::isBarcodePart(const std::vector<cv::Point>& contour) const {
  const double k_th = 2.34;
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

std::vector<std::vector<cv::Point>> BarcodeDetector::getLines(const std::vector<cv::Point>& contour) const {
  if (contour.size() == 0) {
    return std::vector<std::vector<cv::Point>>();
  }

  auto get_key = [](const cv::Point& point) {
    return std::to_string(point.x) + "_" + std::to_string(point.y);
  };

  // 隣接する8近傍にラインが存在するかチェック。存在する場合はそのラインのindexを返す
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
    // 既存の線分に属する点をチェック
    for (const cv::Point& point : contour) {

      // 既に線分に所属済みのピクセル
      if (line_map.count(get_key(point)) > 0) {
        continue;
      }

      // まだ所属してないピクセル
      int point_line_index = neighborhood_index(point, line_map);
      if (point_line_index > 0) {
        line_map[get_key(point)] = std::tuple(point, point_line_index);
        new_line_point_num++;
      }
    }
    
    // 対象の線分への追加対象の点がもう存在しない
    if (new_line_point_num == 0) {
      // 全点が線に所属してたら処理終了
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

      // まだ線分に所属してない点を新しい点を1つ選んで新しい線分に所属させる
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

std::vector<std::vector<cv::Point>> BarcodeDetector::getBarcodeCandidateLines(const std::vector<cv::Point>& contour) const {
  // カット済みの輪郭を線分に分割
  std::vector<std::vector<cv::Point>> lines = getLines(contour);

  // 線分2本以上を検知できなければバーコードの一部とみなさない
  if (lines.size() < 2) {
    return std::vector<std::vector<cv::Point>>();
  }

  // 線分を3本以上検知した場合は元も長い線分2本をバーコードの一部とみなす
  std::vector<std::vector<cv::Point>> largest_lines(2, std::vector<cv::Point>());
  if (lines.size() >= 3) {
    for (const auto& line : lines) {
      if (largest_lines.at(0).size() < line.size()) {
        largest_lines[1] = largest_lines[0];
        largest_lines[0] = line;
      } else if (largest_lines.at(1).size() < line.size()) {
        largest_lines[1] = line;
      }
    }

    return largest_lines;
  }

  return lines;
}

std::vector<cv::Point2d> BarcodeDetector::samplingLine(const std::vector<cv::Point>& line) const {
  const int sampling_interval = 10;

  // 線分の方向がx軸かy軸方向か確認
  std::tuple<Direction, int> direction_tuple = getDirection(line);
  Direction direction = std::get<0>(direction_tuple);
  int length = std::get<1>(direction_tuple);

  //std::cout << length << std::endl;

  // サンプリングできないくらい線分が短ければサンプリングをあきらめる
  if (length/sampling_interval <= 2) {
    return std::vector<cv::Point2d>();
  }

  // サンプリング実施
  std::vector<cv::Point2d> sampling_points;
  if (direction == Direction::Horizontal) {
    std::vector<cv::Point> sorted_line = line;
    std::sort(sorted_line.begin(), sorted_line.end(), [](const cv::Point& p1, const cv::Point& p2) {
      return p1.x < p2.x;
    });

    std::cout << line.size() << ", " << sorted_line.at(0).x << ", " << sorted_line.at(sorted_line.size() - 1).x << std::endl;

    int sampling_num = (sorted_line.at(sorted_line.size() - 1).x - sorted_line.at(0).x) / sampling_interval;
    for (int sampling_index = 0; sampling_index < sampling_num; sampling_index++) {
      std::vector<cv::Point> line_part;
      int min_x = sorted_line.at(0).x + sampling_interval * sampling_index;
      int max_x = min_x + sampling_interval;
      //std::cout << "min: " << min_x << ", max: " << max_x << std::endl;
      for (const auto& point : sorted_line) {
        if (point.x >= min_x && point.x < max_x) {
          //std::cout << point << std::endl;
          line_part.push_back(point);
        }
      }

      cv::Point2d sampling_point = sampling(line_part);
      sampling_points.push_back(sampling_point);
      //std::cout << "samplint index: " << sampling_index << ", line part point num: " << line_part.size() << std::endl;
    }
  } else {

  }

  return sampling_points;
}

cv::Point2d BarcodeDetector::sampling(const std::vector<cv::Point>& line_part) const {
  uint m00 = line_part.size();
  double m10 = 0;
  double m01 = 0;

  // 断面一次モーメント導出
  for (const auto& point : line_part) {
    m10 += 1.0 * 1.0 * (double)point.x;
    m01 += 1.0 * 1.0 * (double)point.y;
    //std::cout << point << std::endl;
  }

  // 図心を導出する
  return cv::Point2d(m10 / (double)m00, m01 / (double)m00);
}

std::tuple<BarcodeDetector::Direction, int> BarcodeDetector::getDirection(const std::vector<cv::Point>& line) const {
  cv::Point max_x_point = *std::max_element(line.begin(), line.end(), [](const cv::Point& p1, const cv::Point& p2) {
    return p1.x < p2.x;
  });

  cv::Point min_x_point = *std::min_element(line.begin(), line.end(), [](const cv::Point& p1, const cv::Point& p2) {
    return p1.x < p2.x;
  });

  cv::Point max_y_point = *std::max_element(line.begin(), line.end(), [](const cv::Point& p1, const cv::Point& p2) {
    return p1.y < p2.y;
  });

  cv::Point min_y_point = *std::min_element(line.begin(), line.end(), [](const cv::Point& p1, const cv::Point& p2) {
    return p1.y < p2.y;
  });

  int x_length = max_x_point.x - min_x_point.x;
  int y_length = max_y_point.y - min_y_point.y;

  Direction direction = x_length > y_length ? Direction::Horizontal : Direction::Vertical;
  int larger_length = x_length > y_length ? x_length : y_length;

  return std::tuple(direction, larger_length);
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

  //
  // Outer contour to line transformation
  //
  std::vector<std::vector<std::vector<cv::Point>>> lines;
  std::vector<std::vector<cv::Point>> cutted_contours;
  for (const auto& contour : barcode_contours) {
    // 輪郭の端をカットして2本の線分にする
    const auto cutted_contour = cutEdge(contour);
    cutted_contours.push_back(cutted_contour);
    std::vector<std::vector<cv::Point>> single_bar_lines = getBarcodeCandidateLines(cutted_contour);
    lines.push_back(single_bar_lines);

    // 2本の線分をそれぞれサンプリングする
    for (const auto& line : single_bar_lines) {
      std::vector<cv::Point2d> sampling_points = samplingLine(line);
      cv::Mat sampling_image = drawLine(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), sampling_points, cv::Scalar(0, 0, 255));
      cv::imshow("sampling", sampling_image);
      cv::waitKey(0);
    }

    //cv::Mat cutted_draw_image = drawLine(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), cutted_contour, cv::Scalar(255, 0, 255));
    //cv::imshow("cutted_single_line", cutted_draw_image);

    //for (const auto& line : single_bar_lines) {
    //  cv::Mat tmp_draw_image = drawLine(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), line, cv::Scalar(0, 255, 255));
    //  cv::imshow("line", tmp_draw_image);
    //  cv::waitKey(0);
    //}
  }


  cv::Mat draw_image3 = drawLines(cv::Mat::zeros(image.rows, image.cols, CV_8UC3), lines, cv::Scalar(255, 255, 0));
  cv::imshow("contours3", draw_image3);
  cv::waitKey(0);
}
