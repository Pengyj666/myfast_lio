#include <opencv2/opencv.hpp>

int tmp_test() {
  // cv::Mat img = cv::Mat(400, 400, CV_8UC1, cv::Scalar(255));
  // cv::circle(img, cv::Point(200, 200), 10, cv::Scalar(0), -1);
  // cv::circle(img, cv::Point(200, 300), 10, cv::Scalar(0), -1);
  // cv::circle(img, cv::Point(300, 200), 10, cv::Scalar(0), -1);
  // cv::circle(img, cv::Point(300, 300), 10, cv::Scalar(0), -1);
  cv::Mat img = cv::imread("/home/edy/test.png", cv::IMREAD_GRAYSCALE);

  cv::Mat binary;
  double thresh = 127;      // 阈值
  double maxval = 255;      // 最大值
  cv::threshold(img, binary, thresh, maxval, cv::THRESH_BINARY);

  cv::Mat dist;
  cv::distanceTransform(binary, dist, cv::DIST_L2, cv::DIST_MASK_5);

  // 将距离转换为8位图像显示
  cv::Mat dist_8u;
  dist.convertTo(dist_8u, CV_8U);
  cv::imshow("img", img);
  cv::imshow("Distance Transform", dist_8u);
  cv::waitKey(0);
  return 0;
}