#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <ros/package.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <sensor_msgs/CompressedImage.h>
#include <geometry_msgs/PointStamped.h>
#include <cv_bridge/cv_bridge.h>

#include <atomic>
#include <queue>
#include <string>
#include <thread>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "common/timed_queue.h"
#include "common/sysutils.h"

using namespace utils;

// 相机参数
cv::Size img_size(640, 480);
cv::Mat camKL, camKR; // 投影内参
cv::Mat camDL, camDR; // 畸变系数
cv::Mat stereo_R, stereo_T;
cv::Mat stereo_R1, stereo_R2, stereo_P1, stereo_P2, stereo_Q;
std::atomic<bool> params_valid(false);

ros::Publisher pub_rgbd_pointcloud;

struct Pose {
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};

cv::Mat getImageFromMsg(const sensor_msgs::CompressedImage::ConstPtr &img_msg, bool to_gray = true);
bool calc_stereo_depth(const cv::Mat &imgL, const cv::Mat &imgR, cv::Mat &imgL_rect, cv::Mat &imgR_rect, cv::Mat &imgD);
void pub_rgbd_cloud(const cv::Mat &imgL, const cv::Mat &imgD, const Pose &Twc);
bool load_stereo_params(const std::string &config_file);
bool load_pose_graph(const std::string &pg_file, TimedQueue<Pose> &pose_q);

int main(int argc, char** argv) {
  ros::init(argc, argv, "vis_test");
  ros::NodeHandle nh("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

  pub_rgbd_pointcloud = nh.advertise<sensor_msgs::PointCloud2>("/as_vmap/rgbd_pointcloud", 1);

  if(argc != 3)
  {
    ROS_ERROR("VIO::main(), usage: rosrun vmap vis_test -d [config file]");
    return 1;
  }

  // 1. 读取地图位姿
  TimedQueue<Pose> pose_q;
  std::string pg_file ="/home/edy/.ros/tttt/pose_graph.txt";
  if (!load_pose_graph(pg_file, pose_q)) {
    return 1;
  }
  double pose_q_last_ts = pose_q(0);
  ROS_INFO("VIO::main(), pose_q size: %d, last_ts: %.3f", pose_q.size(), pose_q_last_ts);

  // 2. 读取相机参数
  std::string config_file = argv[2];
  if (!load_stereo_params(config_file)) {
    return 1;
  }

  // 3. 读取地图原始bag文件, 生成点云发布
  const std::string camL_topic = "/vio/left/image_raw/compressed";
  const std::string camR_topic = "/vio/right/image_raw/compressed";
  std::vector<std::string> topics;
  topics.push_back(camL_topic);
  topics.push_back(camR_topic);

  rosbag::Bag bag;
  // bag.open("/home/edy/datasets/vio_dev_bags/sn201_new-vio-rtk-fusion-dev1.bag", rosbag::bagmode::Read);
  bag.open("/home/edy/datasets/vmap_dev_bags/2025-03-23-18-22-37.bag", rosbag::bagmode::Read);
  
  rosbag::View view(bag, rosbag::TopicQuery(topics));
  
  std::queue<sensor_msgs::CompressedImage::ConstPtr> imgL_buf;
  std::queue<sensor_msgs::CompressedImage::ConstPtr> imgR_buf;
  double pre_ts = 1000.0;
  for (const rosbag::MessageInstance& msg : view) {
    Sleep(10);
    const auto& tpn = msg.getTopic();
    if (tpn == camL_topic) {
      sensor_msgs::CompressedImage::ConstPtr img_msg = msg.instantiate<sensor_msgs::CompressedImage>();
      if (img_msg != nullptr) {
        imgL_buf.push(img_msg);
      }
    } else if (tpn == camR_topic) {
      sensor_msgs::CompressedImage::ConstPtr img_msg = msg.instantiate<sensor_msgs::CompressedImage>();
      if (img_msg != nullptr) {
        imgR_buf.push(img_msg);
      }
    }

    cv::Mat imgL, imgR;
    double frame_ts = -1.0;
    if (!imgL_buf.empty() && !imgR_buf.empty()) {
      double tsL = imgL_buf.front()->header.stamp.toSec();
      double tsR = imgR_buf.front()->header.stamp.toSec();
      if (tsL < tsR - 0.003) {
        imgL_buf.pop();
        // std::printf("drop imgL: tsL: %.3f\n", tsL);
      } else if (tsL > tsR + 0.003) {
        imgR_buf.pop();
        // std::printf("drop imgR: tsR: %.3f\n", tsR);
      } else {
        imgL = getImageFromMsg(imgL_buf.front());
        imgL_buf.pop();          
        imgR = getImageFromMsg(imgR_buf.front());
        imgR_buf.pop();
        frame_ts = tsL;
        // std::printf("calc depth: tsL: %.3f, tsR: %.3f\n", tsL, tsR);
      }
    }

    if (!imgL.empty() && !imgR.empty()) {
      // 查找位姿
      int idx = pose_q.findAfter(frame_ts);
      bool find_pose = false;
      if (idx >= 0) {
        if (std::abs(frame_ts - pose_q(idx)) < 0.005) {
          std::printf("find pose: frame_ts: %.3f, pose_ts: %.3f, idx: %d\n", frame_ts, pose_q(idx), idx);
          find_pose = true;
        }
      } else {
        // std::printf("not find pose: frame_ts: %.3f\n", frame_ts);
      }

      if (!find_pose) {
        continue;
      }
      
      cv::Mat imgD, imgL_rect, imgR_rect;
      if (calc_stereo_depth(imgL, imgR, imgL_rect, imgR_rect, imgD)) {
        pub_rgbd_cloud(imgL_rect, imgD, pose_q[idx]);
      }
    }
  }
  
  // 3. 订阅重定位位姿，可视化
  return 0;
}

cv::Mat getImageFromMsg(const sensor_msgs::CompressedImage::ConstPtr &img_msg, bool to_gray)
{
  cv::Mat img;
  try {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg);
    cv::Mat image = cv_ptr->image;
    img = image.clone();
    if (to_gray)
    {
      cv::cvtColor(img, img, CV_RGB2GRAY);
    }
  } catch (cv_bridge::Exception& e) {
    std::printf("Could not convert from '%s' to 'bgr8'.\n", img_msg->format.c_str());
  }

  return img;
}

bool calc_stereo_depth(const cv::Mat &imgL, const cv::Mat &imgR, 
                       cv::Mat &imgL_rect, cv::Mat &imgR_rect, cv::Mat &imgD) {
  cv::Mat mapLx, mapLy, mapRx, mapRy;
  cv::initUndistortRectifyMap(camKL, camDL, stereo_R1, stereo_P1, img_size, CV_32FC1, mapLx, mapLy);
  cv::initUndistortRectifyMap(camKR, camDR, stereo_R2, stereo_P2, img_size, CV_32FC1, mapRx, mapRy);

  cv::remap(imgL, imgL_rect, mapLx, mapLy, cv::INTER_LINEAR);
  cv::remap(imgR, imgR_rect, mapRx, mapRy, cv::INTER_LINEAR);

  int minDisparity = 0;
  int numDisparities = 16 * 5;  // 必须为16的倍数，这里设置为80
  int blockSize = 4;           // 匹配块大小，奇数
  int disp12MaxDiff = 1;
  int preFilterCap = 63;
  int uniquenessRatio = 10;
  int speckleWindowSize = 100;
  int speckleRange = 2;
  int mode = cv::StereoSGBM::MODE_SGBM_3WAY;

  cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
      minDisparity, 
      numDisparities, 
      blockSize, 
      8 * blockSize * blockSize, 
      32 * blockSize * blockSize, 
      disp12MaxDiff, 
      preFilterCap, 
      uniquenessRatio, 
      speckleWindowSize, 
      speckleRange, 
      mode);

  cv::Mat disp, disp8;
  sgbm->compute(imgL_rect, imgR_rect, disp);
  cv::normalize(disp, disp8, 0, 255, cv::NORM_MINMAX, CV_8U);

  cv::Mat depth, depth_show;
  cv::reprojectImageTo3D(disp, depth, stereo_Q, true);

  std::vector<cv::Mat> depth_channels(3);
  cv::split(depth, depth_channels);
  cv::Mat z_depth = depth_channels[2];
  imgD = z_depth * 16.0;
  return true;
}

void pub_rgbd_cloud(const cv::Mat &imgL, const cv::Mat &imgD, const Pose &Twc)
{
  sensor_msgs::PointCloud2 rgbd_cloud;
  rgbd_cloud.header.frame_id = "world";
  rgbd_cloud.header.stamp = ros::Time::now();
  rgbd_cloud.fields.resize(4);
  rgbd_cloud.fields[0].name = "x";
  rgbd_cloud.fields[0].offset = 0;
  rgbd_cloud.fields[0].count = 1;
  rgbd_cloud.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
  rgbd_cloud.fields[1].name = "y";
  rgbd_cloud.fields[1].offset = 4;
  rgbd_cloud.fields[1].count = 1;
  rgbd_cloud.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
  rgbd_cloud.fields[2].name = "z";
  rgbd_cloud.fields[2].offset = 8;
  rgbd_cloud.fields[2].count = 1;
  rgbd_cloud.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
  rgbd_cloud.fields[3].name = "rgb";
  rgbd_cloud.fields[3].offset = 12;
  rgbd_cloud.fields[3].count = 1;
  rgbd_cloud.fields[3].datatype = sensor_msgs::PointField::UINT32;

  rgbd_cloud.is_bigendian = false;
  rgbd_cloud.point_step = 16;
  rgbd_cloud.height = 1;
  rgbd_cloud.is_dense = true;

  const int width = imgL.cols;
  const int height = imgL.rows;
  rgbd_cloud.width = width * height;
  rgbd_cloud.row_step = rgbd_cloud.point_step * rgbd_cloud.width;
  rgbd_cloud.data.resize(rgbd_cloud.row_step * rgbd_cloud.height);

  sensor_msgs::PointCloud2Iterator<float> iter_x(rgbd_cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(rgbd_cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(rgbd_cloud, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(rgbd_cloud, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(rgbd_cloud, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(rgbd_cloud, "b");

  const float fx = stereo_P1.at<double>(0, 0);
  const float fy = stereo_P1.at<double>(1, 1);
  const float cx = stereo_P1.at<double>(0, 2);
  const float cy = stereo_P1.at<double>(1, 2);
  
  int cnt = 0;
  for (int j = 0; j < height; j+=10) {
    for (int i = 0; i < width; i+=10) {
      float d = imgD.at<float>(j, i);
      if (d <= 0.2f || d >= 3.0f) {
        continue;
      }
      cnt++;

      float x = d;
      float y = -(i - cx) * d / fx;
      float z = -(j - cy) * d / fy;

      Eigen::Vector3d p(x, y, z);
      p = Twc.q * p + Twc.pos;

      *iter_x = p.x();
      *iter_y = p.y();
      *iter_z = p.z();
      
      *iter_r = imgL.at<cv::Vec3b>(j, i)[2];
      *iter_g = imgL.at<cv::Vec3b>(j, i)[1];
      *iter_b = imgL.at<cv::Vec3b>(j, i)[0];

      ++iter_x; ++iter_y; ++iter_z;
      ++iter_r; ++iter_g; ++iter_b;
    }
  }
  ROS_INFO("VIO::pub_rgbd(): rgbd pointcloud size: %d",  cnt);
  pub_rgbd_pointcloud.publish(rgbd_cloud);
}

bool load_stereo_params(const std::string &config_file)
{
  ROS_INFO("VIO::main() config_file: %s", config_file.c_str());

  if (!IsFileExisting(config_file.c_str())) {
    ROS_ERROR("VIO::main() config_file: %s not exist", config_file.c_str());
    return false;
  }

  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if(!fsSettings.isOpened())
  {
    ROS_ERROR("VIO::main(): ERROR: Wrong path to settings");
    return false;
  }
  std::string cam0Path;
  fsSettings["cam0_calib"] >> cam0Path;
  std::string cam1Path;
  fsSettings["cam1_calib"] >> cam1Path;
  std::string stereoPath;
  fsSettings["stereo_calib"] >> stereoPath;
  
  // 读取双目内参
  cv::FileStorage cam0FS(cam0Path, cv::FileStorage::READ);
  if(!cam0FS.isOpened())
  {
    ROS_ERROR("VIO::main(): Wrong path to camera0 calibration file: %s", cam0Path.c_str());
    return false;
  }
  {
    cv::FileNode dist_Node = cam0FS["distortion_parameters"];
    double k1 = static_cast< double > (dist_Node["k1"]);
    double k2 = static_cast< double > (dist_Node["k2"]);
    double p1 = static_cast< double > (dist_Node["p1"]);
    double p2 = static_cast< double > (dist_Node["p2"]);
    double k3 = static_cast< double > (dist_Node["k3"]);
    cv::FileNode intri_Node = cam0FS["projection_parameters"];
    double fx = static_cast< double > (intri_Node["fx"]);
    double fy = static_cast< double > (intri_Node["fy"]);
    double cx = static_cast< double > (intri_Node["cx"]);
    double cy = static_cast< double > (intri_Node["cy"]);
    ROS_INFO("VIO::main(): camera0 intrinsic param: fx=%.8f, fy=%.8f, cx=%.8f, cy=%.8f", fx, fy, cx, cy);
    ROS_INFO("VIO::main(): camera0 distortion param: k1=%.8f, k2=%.8f, p1=%.8f, p2=%.8f, k3=%.8f", k1, k2, p1, p2, k3);
    camKL = (cv::Mat_<double>(3, 3) << fx, 0.0, cx,  0.0, fy, cy,  0.0, 0.0, 1.0);
    camDL = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
  }
  
  cv::FileStorage cam1FS(cam1Path, cv::FileStorage::READ);
  if(!cam1FS.isOpened())
  {
    ROS_ERROR("VIO::main(): Wrong path to camera1 calibration file: %s", cam1Path.c_str());
    return false;
  }
  {
    cv::FileNode dist_Node = cam1FS["distortion_parameters"];
    double k1 = static_cast< double > (dist_Node["k1"]);
    double k2 = static_cast< double > (dist_Node["k2"]);
    double p1 = static_cast< double > (dist_Node["p1"]);
    double p2 = static_cast< double > (dist_Node["p2"]);
    double k3 = static_cast< double > (dist_Node["k3"]);
    cv::FileNode intri_Node = cam1FS["projection_parameters"];
    double fx = static_cast< double > (intri_Node["fx"]);
    double fy = static_cast< double > (intri_Node["fy"]);
    double cx = static_cast< double > (intri_Node["cx"]);
    double cy = static_cast< double > (intri_Node["cy"]);
    ROS_INFO("VIO::main(): camera1 intrinsic param: fx=%.8f, fy=%.8f, cx=%.8f, cy=%.8f", fx, fy, cx, cy);
    ROS_INFO("VIO::main(): camera1 distortion param: k1=%.8f, k2=%.8f, p1=%.8f, p2=%.8f, k3=%.8f", k1, k2, p1, p2, k3);
    camKR = (cv::Mat_<double>(3, 3) << fx, 0.0, cx,  0.0, fy, cy,  0.0, 0.0, 1.0);
    camDR = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
  }

  // 读取双目外参
  cv::FileStorage stereoFS(stereoPath, cv::FileStorage::READ);
  if(!stereoFS.isOpened())
  {
    ROS_ERROR("VIO::main(): Wrong path to stereo calibration file: %s", stereoPath.c_str());
    return false;
  }
  cv::FileNode stereoNode = stereoFS["stereo_params"];
  double roll = static_cast< double > (stereoNode["Rx"]);
  double pitch = static_cast< double > (stereoNode["Ry"]);
  double yaw = static_cast< double > (stereoNode["Rz"]);
  double tx = static_cast< double > (stereoNode["Tx"]) * 0.001;
  double ty = static_cast< double > (stereoNode["Ty"]) * 0.001;
  double tz = static_cast< double > (stereoNode["Tz"]) * 0.001;
  ROS_INFO("VIO::main(): stereo extrinsic param: rpy=%.8f,%.8f,%.8f, xyz=%.8f,%.8f,%.8f", roll, pitch, yaw, tx, ty, tz);
  Eigen::Matrix3d Rx, Ry, Rz;
  Rx << 1.0, 0.0, 0.0,
      0.0, cos(roll), -sin(roll),
      0.0, sin(roll), cos(roll);
  Ry << cos(pitch), 0.0, sin(pitch),
      0.0, 1.0, 0.0,
      -sin(pitch), 0.0, cos(pitch);
  Rz << cos(yaw), -sin(yaw), 0.0,
      sin(yaw), cos(yaw), 0.0,
      0.0, 0.0, 1.0;
  Eigen::Matrix3d R_rl = Rz * Ry * Rx;
  Eigen::Vector3d t_rl(tx, ty, tz);

  stereo_R = (cv::Mat_<double>(3, 3) << R_rl(0, 0), R_rl(0, 1), R_rl(0, 2),
                                        R_rl(1, 0), R_rl(1, 1), R_rl(1, 2),
                                        R_rl(2, 0), R_rl(2, 1), R_rl(2, 2));
  stereo_T = (cv::Mat_<double>(3, 1) << t_rl(0), t_rl(1), t_rl(2));

  cv::stereoRectify(camKL, camDL, camKR, camDR, img_size, 
      stereo_R, stereo_T, stereo_R1, stereo_R2, stereo_P1, stereo_P2, stereo_Q, 
      cv::CALIB_ZERO_DISPARITY, 0, img_size);

  float base_line = -stereo_P2.at<double>(0, 3) / stereo_P2.at<double>(0, 0);

  std::cout << "\nbase_line: \n" << base_line << std::endl;
  std::cout << "\nP1:\n" << stereo_P1 << std::endl;
  std::cout << "\nP2:\n" << stereo_P2 << std::endl;
  std::cout << "\nQ:\n" << stereo_Q << std::endl;

  params_valid.store(true);

  ROS_INFO("VIO::main(): read stereo calibration successfully");

  return true;
}

bool load_pose_graph(const std::string &pg_file, TimedQueue<Pose> &pose_q)
{
  pose_q.reset(9192);

  FILE * pFile;
  if (!IsFileExisting(pg_file.c_str())) {
    ROS_ERROR("pose file not exist\n");
    return false;
  }
  pFile = fopen (pg_file.c_str(),"r");

  int index;
  double time_stamp;
  double VIO_Tx, VIO_Ty, VIO_Tz;
  double PG_Tx, PG_Ty, PG_Tz;
  double VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz;
  double PG_Qw, PG_Qx, PG_Qy, PG_Qz;
  double loop_info_0, loop_info_1, loop_info_2, loop_info_3;
  double loop_info_4, loop_info_5, loop_info_6, loop_info_7;
  int loop_index;
  int keypoints_num;
  Eigen::Matrix<double, 8, 1 > loop_info;
  int cnt = 0;
  while (fscanf(pFile,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d", &index, &time_stamp, 
                &VIO_Tx, &VIO_Ty, &VIO_Tz, 
                &PG_Tx, &PG_Ty, &PG_Tz, 
                &VIO_Qw, &VIO_Qx, &VIO_Qy, &VIO_Qz, 
                &PG_Qw, &PG_Qx, &PG_Qy, &PG_Qz, 
                &loop_index,
                &loop_info_0, &loop_info_1, &loop_info_2, &loop_info_3, 
                &loop_info_4, &loop_info_5, &loop_info_6, &loop_info_7,
                &keypoints_num) != EOF) 
  {
    Pose pose;
    pose.pos << VIO_Tx, VIO_Ty, VIO_Tz;
    pose.q.w() = VIO_Qw;
    pose.q.x() = VIO_Qx;
    pose.q.y() = VIO_Qy;
    pose.q.z() = VIO_Qz;

    pose_q.emplace_back(pose, time_stamp);
    cnt++;
  }
  fclose (pFile);
  ROS_INFO("read pose graph done, cnt: %d", cnt);
  return true;
}