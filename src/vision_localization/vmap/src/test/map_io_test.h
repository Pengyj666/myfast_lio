#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

int map_io_test()
{
  Eigen::Quaterniond q(0.707, 0.0, 0.707, 0.0);
  Eigen::Vector3d t(1.0, 2.0, 3.0);
  int loop_index = 3;
  double timestamp = 12.0;
  Eigen::Matrix<double, 8, 1> loop_info;
  loop_info << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;

  std::string mower_dir = "/home/edy/Workspace/CY_ws/mower_ws/mower_localization/";
  std::string data_dir = mower_dir + "vision_localization/support_files/";
  std::string kp_file = data_dir + "test_keypoints.txt";
  std::string des_file = data_dir + "test_briefdes.dat";

  std::vector<cv::KeyPoint> keypoints;
  // std::vector<cv::
}