#ifndef _ARTICLE_NUMBER_H_
#define _ARTICLE_NUMBER_H_

#include <opencv2/barcode.hpp>

class ArticleNumber {
public:
  std::string article_number;
  cv::barcode::BarcodeType type;
  int method_type; // for DEBUG
};

#endif
