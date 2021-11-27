#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include "Bar.h"

Bar::Bar(const std::vector<cv::Point>& contour) {
  this->contour = contour;
  is_valid = isBarcodeElement();
  corner = detectCorner(contour);
  center = detectCenter(contour);
}

bool Bar::isBarcodeElement() {
  const double k_th = 2.34;
  uint pc = contour.size();

  double d = getDiagonalLength(contour);

  if (pc <= k_th * d) {
    return true;
  }

  return false;
}

std::array<cv::Point, 4> Bar::detectCorner(const std::vector<cv::Point>& contour) const {
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

  std::array<cv::Point, 4> corner;
  corner[0] = cv::Point(min_x_point.x, min_y_point.y);
  corner[1] = cv::Point(max_x_point.x, min_y_point.y);
  corner[2] = cv::Point(min_x_point.x, max_y_point.y);
  corner[3] = cv::Point(max_x_point.x, max_y_point.y);

  return corner;
}

cv::Point2d Bar::detectCenter(const std::vector<cv::Point>& contour) const {
  // バーは長方形のはずなので中点は重心と近似できるはず
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

  double x = (double)(min_x_point.x + max_x_point.x) / 2.0;
  double y = (double)(min_y_point.y + max_y_point.y) / 2.0;

  return cv::Point2d(x, y);
}

double Bar::getDiagonalLength(const std::vector<cv::Point>& contour) const {
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

  // バーコードの一部ならほぼ長方形のはずなので、この値は対角線の近似値になるはず
  double distance = std::sqrt(std::pow((double)(max_x_point.x - min_x_point.x), 2.0) + std::pow((double)(max_y_point.y - min_y_point.y), 2.0));

  return distance;
}

std::vector<cv::Point> Bar::cutEdge(const std::vector<cv::Point>& contour) const {
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

std::vector<std::vector<cv::Point>> Bar::detectAllLines(const std::vector<cv::Point>& contour) const {
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

std::array<std::vector<cv::Point>, 2> Bar::detectLines(const std::vector<cv::Point>& contour) const {
  // カット済みの輪郭を線分に分割
  std::vector<std::vector<cv::Point>> lines = detectAllLines(contour);

  // 線分2本以上を検知できなければバーコードの一部とみなさない
  if (lines.size() < 2) {
    std::array<std::vector<cv::Point>, 2> empty_lines;
    std::fill(empty_lines.begin(), empty_lines.end(), std::vector<cv::Point>());
    return empty_lines;
  }

  // 線分を3本以上検知した場合は元も長い線分2本をバーコードの一部とみなす
  std::array<std::vector<cv::Point>, 2> largest_lines;
  std::fill(largest_lines.begin(), largest_lines.end(), std::vector<cv::Point>());
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

std::tuple<Bar::Direction, int> Bar::getDirection(const std::vector<cv::Point>& line) const {
  if (line.size() < 2) {
    return std::tuple(Direction::Horizontal, 0);
  }

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

cv::Point2d Bar::samplingPoint(const std::vector<cv::Point>& line_part) const {
  uint m00 = line_part.size();
  double m10 = 0;
  double m01 = 0;

  // 断面一次モーメント導出
  for (const auto& point : line_part) {
    m10 += 1.0 * 1.0 * (double)point.x;
    m01 += 1.0 * 1.0 * (double)point.y;
  }

  // 図心を導出する
  return cv::Point2d(m10 / (double)m00, m01 / (double)m00);
}

std::vector<cv::Point2d> Bar::samplingLine(const std::vector<cv::Point>& line) const {
  const int sampling_interval = 10;

  // 線分の方向がx軸かy軸方向か確認
  std::tuple<Direction, int> direction_tuple = getDirection(line);
  Direction direction = std::get<0>(direction_tuple);
  int length = std::get<1>(direction_tuple);

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

    int sampling_num = (sorted_line.at(sorted_line.size() - 1).x - sorted_line.at(0).x) / sampling_interval;
    for (int sampling_index = 0; sampling_index < sampling_num; sampling_index++) {
      std::vector<cv::Point> line_part;
      int min_x = sorted_line.at(0).x + sampling_interval * sampling_index;
      int max_x = min_x + sampling_interval;
      for (const auto& point : sorted_line) {
        if (point.x >= min_x && point.x < max_x) {
          line_part.push_back(point);
        }
      }

      cv::Point2d sampling_point = samplingPoint(line_part);
      sampling_points.push_back(sampling_point);
    }
  } else {
    std::vector<cv::Point> sorted_line = line;
    std::sort(sorted_line.begin(), sorted_line.end(), [](const cv::Point& p1, const cv::Point& p2) {
      return p1.y < p2.y;
    });

    int sampling_num = (sorted_line.at(sorted_line.size() - 1).y - sorted_line.at(0).y) / sampling_interval;
    for (int sampling_index = 0; sampling_index < sampling_num; sampling_index++) {
      std::vector<cv::Point> line_part;
      int min_y = sorted_line.at(0).y + sampling_interval * sampling_index;
      int max_y = min_y + sampling_interval;
      for (const auto& point : sorted_line) {
        if (point.y >= min_y && point.y < max_y) {
          line_part.push_back(point);
        }
      }

      cv::Point2d sampling_point = samplingPoint(line_part);
      sampling_points.push_back(sampling_point);
    }
  }

  return sampling_points;
}

double Bar::lineDegree(const std::vector<cv::Point2d>& sampling_points) const {
  cv::Vec4f fitting_info;
  cv::fitLine(sampling_points, fitting_info, cv::DIST_L2, 0, 0.01, 0.01);

  cv::Point2d direction(fitting_info[0], fitting_info[1]);

  double radian = std::acos(cv::Point2d(1.0, 0.0).dot(direction));
  double degree = radian * 180.0 / M_PI;

  return degree;
}

bool Bar::isValid() const {
  return is_valid;
}

void Bar::setIsValid(bool is_valid) {
  this->is_valid = is_valid;
}

double Bar::getDegree() const {
  return degree;
}

double Bar::getBarLength() const {
  // 長編の方が圧倒的に長いバーコード成分なら、バーの長さを対角線の長さで近似できるはず
  return getDiagonalLength(contour);
}

std::array<cv::Point, 4> Bar::getCorner() const {
  return corner;
}

cv::Point2d Bar::getCenter() const {
  return center;
}

void Bar::lineFitting() {
  // 輪郭の端をカットして2本の線分にする
  std::vector<cv::Point> cutted_contour = cutEdge(contour);
  std::array<std::vector<cv::Point>, 2> lines = detectLines(cutted_contour);
  if (lines.at(0).size() == 0 || lines.at(1).size() == 0) {
    is_valid = false;
    return;
  }
  this->lines = lines;

  // 2本の線分をそれぞれサンプリングする
  const int min_point_num = 2;  // TODO 要調整
  std::array<std::vector<cv::Point2d>, 2> sampling_lines;
  std::fill(sampling_lines.begin(), sampling_lines.end(), std::vector<cv::Point2d>());
  for (int i = 0; i < 2; i++) {
    std::vector<cv::Point2d> sampling_points = samplingLine(lines.at(i));
    sampling_lines[i] = sampling_points;
    if (sampling_points.size() <= min_point_num) {
      is_valid = false;
      return;
    }
  }
  this->sampling_lines = sampling_lines;

  // サンプリングした点群を直線にフィッティングしてx軸との角度を求める
  std::array<double, 2> line_degrees{0.0, 0.0};
  for (int i = 0; i < 2; i++) {
    line_degrees[i] = lineDegree(sampling_lines.at(i));
  }

  // 2本の線分の角度の平均をバーの角度とする
  // 2本の線分の成す角が一定以上ならはじいてしまってもいいかも
  // 20度以上 or 160度未満ならはじくてきな感じで 
  degree = (line_degrees.at(0) + line_degrees.at(1)) / 2.0;
}

std::vector<cv::Point> Bar::getContour() const {
  return contour;
}

std::array<std::vector<cv::Point2d>, 2> Bar::getSamplingLines() const {
  return sampling_lines;
}

std::array<std::vector<cv::Point>, 2> Bar::getLines() const {
  return lines;
}
