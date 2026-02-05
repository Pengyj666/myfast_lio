#include <cstdio>
#include <queue>
#include <mutex>
#include <string>
#include <thread>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "common/log_filters.h"
#include "common/sysutils.h"

using namespace utils;

std::queue<sensor_msgs::CompressedImage::ConstPtr> img0_buf;
std::queue<sensor_msgs::CompressedImage::ConstPtr> img1_buf;
std::mutex m_buf;

std::vector<cv::Vec3b> colorBar2;

// 相机参数
cv::Size img_size(640, 480);
cv::Mat camKL, camKR; // 投影内参
cv::Mat camDL, camDR; // 畸变系数
cv::Mat stereo_R, stereo_T;
cv::Mat stereo_R1, stereo_R2, stereo_P1, stereo_P2, stereo_Q;
std::atomic<bool> params_valid(false);

ros::Publisher pub_color_depth;
ros::Publisher pub_rect_stereo;
ros::Publisher pub_rgbd_image;
ros::Publisher pub_rgbd_pointcloud;

void procImgD(const cv::Mat &imgD, cv::Mat &colorD, 
              const std::vector<cv::Vec3b> &colorMap) {
  int Len = colorMap.size()-1;
  if (Len <= 1)
    return;

  const float factor = 1.0;
  float min_dist = 0.2 * factor;
  float max_dist = 3.0 * factor;

  colorD = cv::Mat(imgD.rows, imgD.cols, CV_8UC3, cv::Scalar::all(0));
  for (int r=0; r<imgD.rows; ++r)
  for (int c=0; c<imgD.cols; ++c) {
    float dist = imgD.at<float>(r, c);
    if (dist <= min_dist || dist >= max_dist)
      continue;
    int idx = std::min(Len, int(Len*(dist / max_dist)));
    if (dist > max_dist)
      colorD.at<cv::Vec3b>(r, c) = cv::Vec3b(255,255,255);
    else
      colorD.at<cv::Vec3b>(r, c) = colorMap[idx];
  }

  std::string d1 = std::to_string(int(imgD.at<float>(300, 200) * 1000));
  std::string d2 = std::to_string(int(imgD.at<float>(400, 200) * 1000));
  std::string d3 = std::to_string(int(imgD.at<float>(300, 400) * 1000));
  std::string d4 = std::to_string(int(imgD.at<float>(400, 400) * 1000));

  cv::circle(colorD, cv::Point(200, 300), 2, cv::Scalar(255, 255, 255), -1);
  cv::circle(colorD, cv::Point(200, 400), 2, cv::Scalar(255, 255, 255), -1);
  cv::circle(colorD, cv::Point(400, 300), 2, cv::Scalar(255, 255, 255), -1);
  cv::circle(colorD, cv::Point(400, 400), 2, cv::Scalar(255, 255, 255), -1);

  cv::putText(colorD, d1, cv::Point(200, 300), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  cv::putText(colorD, d2, cv::Point(200, 400), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  cv::putText(colorD, d3, cv::Point(400, 300), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  cv::putText(colorD, d4, cv::Point(400, 400), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

cv::Mat getImageFromMsg(const sensor_msgs::CompressedImage::ConstPtr &img_msg, bool to_gray = true)
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

void img0_callback(const sensor_msgs::CompressedImage::ConstPtr &img_msg)
{
  static double pre_ts = 0.0;
  double cur_ts = img_msg->header.stamp.toSec();
  if (pre_ts > 0.0 && cur_ts > pre_ts + 0.2) {
    ROS_WARN("VIO::img0_callback(): img0 timestamp jump too large, dts=%.3f, cur_ts=%.3f", cur_ts - pre_ts, cur_ts);
  }
  static SimpleLogFilter log_filter(1000);
  if (log_filter.Output(GetNow_Steady())) {
    ROS_INFO("VIO::img0_callback(): cur img0 timestamp: %.3f\n", cur_ts);
  }
  pre_ts = cur_ts;
  m_buf.lock();
  img0_buf.push(img_msg);
  m_buf.unlock();
}

void img1_callback(const sensor_msgs::CompressedImage::ConstPtr &img_msg)
{
  static double pre_ts = 0.0;
  double cur_ts = img_msg->header.stamp.toSec();
  if (pre_ts > 0.0 && cur_ts > pre_ts + 0.2) {
    ROS_WARN("VIO::img1_callback(): img1 timestamp jump too large, dts=%.3f, cur_ts=%.3f", cur_ts - pre_ts, cur_ts);
  }
  static SimpleLogFilter log_filter(1000);
  if (log_filter.Output(GetNow_Steady())) {
    ROS_INFO("VIO::img1_callback(): cur img1 timestamp: %.3f", cur_ts);
  }
  pre_ts = cur_ts;
  m_buf.lock();
  img1_buf.push(img_msg);
  m_buf.unlock();
}

void pub_rgbd(const cv::Mat &imgL, const cv::Mat &imgR, const cv::Mat &imgD)
{
  // 对齐矫正图像
  cv::Mat rect_img;
  cv::hconcat(imgL, imgR, rect_img);
  cv::line(rect_img, cv::Point(0, 200), cv::Point(rect_img.cols, 200), cv::Scalar(0, 0, 255), 1);
  {
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rect_img).toImageMsg();
    msg->header.stamp = ros::Time::now();
    pub_rect_stereo.publish(msg);
  }

  // 着色深度图
  cv::Mat depth_show;
  procImgD(imgD, depth_show, colorBar2);
  {
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", depth_show).toImageMsg();
    msg->header.stamp = ros::Time::now();
    pub_color_depth.publish(msg);
  }

  // rgbd图像
  cv::Mat rgbd_l;
  cv::addWeighted(imgL, 0.7, depth_show, 0.3, 0.0, rgbd_l);
  {
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rgbd_l).toImageMsg();
    msg->header.stamp = ros::Time::now();
    pub_rgbd_image.publish(msg);
  }

  // 彩色点云
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
  for (int j = 0; j < height; j+=5) {
    for (int i = 0; i < width; i+=5) {
      float d = imgD.at<float>(j, i);
      if (d <= 0.2f || d >= 5.0f) {  
        continue;
      }
      cnt++;

      *iter_x = d;
      *iter_y = -(i - cx) * d / fx;
      *iter_z = -(j - cy) * d / fy;
      
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

void sync_process()
{
  while(1)
  {
    cv::Mat imgL, imgR;
    std_msgs::Header header;
    double time = 0;
    m_buf.lock();
    if (!img0_buf.empty() && !img1_buf.empty())
    {
      double time0 = img0_buf.front()->header.stamp.toSec();
      double time1 = img1_buf.front()->header.stamp.toSec();
      // 0.003s sync tolerance
      if(time0 < time1 - 0.003)
      {
        img0_buf.pop();
        printf("throw img0\n");
      }
      else if(time0 > time1 + 0.003)
      {
        img1_buf.pop();
        printf("throw img1\n");
      }
      else
      {
        time = img0_buf.front()->header.stamp.toSec();
        header = img0_buf.front()->header;
        imgL = getImageFromMsg(img0_buf.front(), false);
        img0_buf.pop();
        imgR = getImageFromMsg(img1_buf.front(), false);
        img1_buf.pop();
      }
    }
    m_buf.unlock();

    static SimpleLogFilter fps_filter(200);
    if (!imgL.empty() && !imgR.empty() && fps_filter.Output(GetNow_Steady())) {
      // 计算并应用校正映射
      cv::Mat mapLx, mapLy, mapRx, mapRy;
      cv::initUndistortRectifyMap(camKL, camDL, stereo_R1, stereo_P1, img_size, CV_32FC1, mapLx, mapLy);
      cv::initUndistortRectifyMap(camKR, camDR, stereo_R2, stereo_P2, img_size, CV_32FC1, mapRx, mapRy);

      cv::Mat imgL_rect, imgR_rect;
      cv::remap(imgL, imgL_rect, mapLx, mapLy, cv::INTER_LINEAR);
      cv::remap(imgR, imgR_rect, mapRx, mapRy, cv::INTER_LINEAR);
      // 3. 计算视差
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

      // 4. 计算深度图        
      cv::Mat depth, depth_show;
      cv::reprojectImageTo3D(disp, depth, stereo_Q, true);

      std::vector<cv::Mat> depth_channels(3);  
      cv::split(depth, depth_channels);
      cv::Mat z_depth = depth_channels[2];
      z_depth = z_depth * 16.0;  // ？？？
      
      pub_rgbd(imgL_rect, imgR_rect, z_depth);
    }
    
    std::chrono::milliseconds dura(5);
    std::this_thread::sleep_for(dura);
  }
}

int main(int argc, char **argv) 
{
  // color map
  // 2. |1----R---->0|0<----B----1|
  //    |0<--------1 G 1-------->0|
  const int Len = 400;
  int colorR=0, colorG=0, colorB=0;
  for (int i=0; i<Len; ++i) {
    colorR = std::max(0, std::min(255, 255 - 255*i*2/Len));
    colorG = std::max(0, std::min(255, 255 - 255*std::abs(Len/2-i)*2/Len));
    colorB = std::max(0, std::min(255, 255 - 255*(Len-i)*2/Len));
    colorBar2.push_back(cv::Vec3b(colorB, colorG, colorR));
  }
  
  ros::init(argc, argv, "as_stereo_rgbd");
  ros::NodeHandle nh("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

  if(argc != 3)
  {
    ROS_ERROR("VIO::main(), usage: rosrun vmap as_stereo_rgbd -d [config file]");
    return 1;
  }

  // 读取相机参数
  std::string config_file = argv[2];
  ROS_INFO("VIO::main() config_file: %s", config_file.c_str());

  if (!IsFileExisting(config_file.c_str())) {
    ROS_ERROR("VIO::main() config_file: %s not exist", config_file.c_str());
    return 1;
  }

  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if(!fsSettings.isOpened())
  {
    ROS_ERROR("VIO::main(): ERROR: Wrong path to settings");
    return 1;
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
    return 1;
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
    return 1;
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
    return 1;
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

  ros::Subscriber sub_img0 = nh.subscribe("/vio/left/image_raw/compressed", 1, img0_callback);
  ros::Subscriber sub_img1 = nh.subscribe("/vio/right/image_raw/compressed", 1, img1_callback);

  pub_color_depth = nh.advertise<sensor_msgs::Image>("/as_stereo/color_depth", 1);
  pub_rect_stereo = nh.advertise<sensor_msgs::Image>("/as_stereo/rect_stereo", 1);
  pub_rgbd_image = nh.advertise<sensor_msgs::Image>("/as_stereo/rgbd_img", 1);
  pub_rgbd_pointcloud = nh.advertise<sensor_msgs::PointCloud2>("/as_stereo/rgbd_pointcloud", 1);

  std::thread sync_thread{sync_process};
  ros::spin();

  return 0;
}
