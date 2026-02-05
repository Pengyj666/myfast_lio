#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <cv_bridge/cv_bridge.h>

#include <queue>
#include <mutex>
#include <opencv2/opencv.hpp>

#include <sensor_msgs/Image.h>

#include "common/sysutils.h"

void procImgD(const cv::Mat &imgD, cv::Mat &colorD, 
              const std::vector<cv::Vec3b> &colorMap) {
  int Len = colorMap.size()-1;
  if (Len <= 1)
    return;

  const float factor = 1.0 / 16.0;
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

  std::string d1 = std::to_string(int(imgD.at<float>(300, 200) * 16.0 * 1000));
  std::string d2 = std::to_string(int(imgD.at<float>(400, 200) * 16.0 * 1000));
  std::string d3 = std::to_string(int(imgD.at<float>(300, 400) * 16.0 * 1000));
  std::string d4 = std::to_string(int(imgD.at<float>(400, 400) * 16.0 * 1000));

  std::printf("d1: %s, d2: %s, d3: %s, d4: %s\n", d1.c_str(), d2.c_str(), d3.c_str(), d4.c_str());

  cv::circle(colorD, cv::Point(200, 300), 2, cv::Scalar(255, 255, 255), -1);
  cv::circle(colorD, cv::Point(200, 400), 2, cv::Scalar(255, 255, 255), -1);
  cv::circle(colorD, cv::Point(400, 300), 2, cv::Scalar(255, 255, 255), -1);
  cv::circle(colorD, cv::Point(400, 400), 2, cv::Scalar(255, 255, 255), -1);

  cv::putText(colorD, d1, cv::Point(200, 300), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  cv::putText(colorD, d2, cv::Point(200, 400), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  cv::putText(colorD, d3, cv::Point(400, 300), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  cv::putText(colorD, d4, cv::Point(400, 400), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

cv::Mat getImageFromMsg(const sensor_msgs::CompressedImage::ConstPtr &img_msg) {
  cv::Mat img;
  try {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg);
    cv::Mat image = cv_ptr->image;
    img = image.clone();
    cv::cvtColor(img, img, CV_RGB2GRAY);
  } catch (cv_bridge::Exception& e) {
    std::printf("Could not convert from '%s' to 'bgr8'.\n", img_msg->format.c_str());
  }

  return img;
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv::Mat img;
    try {
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat image = cv_ptr->image;
      img = image.clone();
      // cv::cvtColor(img, img, CV_RGB2GRAY);
    } catch (cv_bridge::Exception& e) {
      std::printf("Could not convert from '%s' to 'bgr8'.\n", img_msg->encoding.c_str());
    }

    return img;
}

int bag_to_image() {
  const std::string camL_topic = "/vio/left/image_raw";
  const std::string camR_topic = "/vio/right/image_raw";

  std::queue<sensor_msgs::ImageConstPtr> imgL_buf;
  std::queue<sensor_msgs::ImageConstPtr> imgR_buf;

  rosbag::Bag bag;
  // bag.open("/home/edy/datasets/vio_dev_bags/sn201_new-vio-rtk-fusion-dev1.bag", rosbag::bagmode::Read);
  bag.open("/home/edy/datasets/depth_data/sn1478/2025-07-22-17-08-55.bag", rosbag::bagmode::Read);
  std::string img_dir = "/home/edy/datasets/depth_data/sn1478/imgs/";
  utils::CreateDir(img_dir.c_str());

  std::vector<std::string> topics;
  topics.push_back(camL_topic);
  topics.push_back(camR_topic);

  rosbag::View view(bag, rosbag::TopicQuery(topics));

  double pre_ts = 1000.0;
  int img_idx = 1;
  for (const rosbag::MessageInstance& msg : view) {
    const auto& tpn = msg.getTopic();
    // std::printf("topic: %s\n", tpn.c_str());
    if (tpn == camL_topic) {
      sensor_msgs::ImageConstPtr img_msg = msg.instantiate<sensor_msgs::Image>();
      if (img_msg != nullptr) {
        imgL_buf.push(img_msg);
      }
    } else if (tpn == camR_topic) {
      sensor_msgs::ImageConstPtr img_msg = msg.instantiate<sensor_msgs::Image>();
      if (img_msg != nullptr) {
        imgR_buf.push(img_msg);
      }
    }

    cv::Mat imgL, imgR;
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
        // std::printf("calc depth: tsL: %.3f, tsR: %.3f\n", tsL, tsR);

        static double pre_ts2 = 0.0;
        if (tsL - pre_ts2 > 1.51) {
          pre_ts2 = tsL;
          cv::Mat simg;
          cv::hconcat(imgL, imgR, simg);
          cv::imwrite(img_dir + std::to_string(img_idx) + ".png", simg);
          img_idx++;
        }
      }
    }
  }
  return 0;
}

int stereo_test() {
  // color map
  // 2. |1----R---->0|0<----B----1|
  //    |0<--------1 G 1-------->0|
  const int Len = 400;
  int colorR=0, colorG=0, colorB=0;
  std::vector<cv::Vec3b> colorBar2;
  for (int i=0; i<Len; ++i) {
    colorR = std::max(0, std::min(255, 255 - 255*i*2/Len));
    colorG = std::max(0, std::min(255, 255 - 255*std::abs(Len/2-i)*2/Len));
    colorB = std::max(0, std::min(255, 255 - 255*(Len-i)*2/Len));
    colorBar2.push_back(cv::Vec3b(colorB, colorG, colorR));
  }

  const std::string camL_topic = "/vio/left/image_raw/compressed";
  const std::string camR_topic = "/vio/right/image_raw/compressed";

  std::queue<sensor_msgs::CompressedImage::ConstPtr> imgL_buf;
  std::queue<sensor_msgs::CompressedImage::ConstPtr> imgR_buf;

  rosbag::Bag bag;
  // bag.open("/home/edy/datasets/vio_dev_bags/sn201_new-vio-rtk-fusion-dev1.bag", rosbag::bagmode::Read);
  bag.open("/home/edy/datasets/vmap_dev_bags/2025-03-23-18-22-37.bag", rosbag::bagmode::Read);

  cv::Mat camKL, camKR; // 投影内参
  cv::Mat camDL, camDR; // 畸变系数
  cv::Mat R, T;
  camKL = (cv::Mat_<double>(3, 3) << 449.6796875, 0, 317.9763489, 0, 449.5430908, 241.8325958, 0, 0, 1);
  camKR = (cv::Mat_<double>(3, 3) << 448.7043762, 0, 322.4730529, 0, 448.4299926, 239.1013336, 0, 0, 1);
  camDL = (cv::Mat_<double>(5, 1) << -4.1786822676658630e-01, 2.2426512837409973e-01, 1.1921262921532616e-04, -3.5174135700799525e-04, -7.0185758173465729e-02);
  camDR = (cv::Mat_<double>(5, 1) << -4.2113566398620605e-01, 2.3245093226432800e-01, -4.1001121280714869e-04, 1.2631643585336860e-05, -7.6264567673206329e-02);
  R = (cv::Mat_<double>(3, 3) << 0.9999158410685516, -0.00949436530587951, -0.008841255997295523,
                                 0.009531031326767268, 0.9999461145704209, 0.004114291842689601,
                                 0.00880171699268862, -0.004198211876162598, 0.9999524512670708);
  T = (cv::Mat_<double>(3, 1) << -0.06008021163940429, -0.0003623995780944825, -0.0002935472726821899);
  // T = (cv::Mat_<double>(3, 1) << -60.08021163940429, -0.3623995780944825, -0.2935472726821899);

  cv::Size img_size(640, 480);
  cv::Mat R1, R2, P1, P2, Q;
  cv::stereoRectify(camKL, camDL, camKR, camDR, img_size, R, T, R1, R2, P1, P2, Q, 
                    cv::CALIB_ZERO_DISPARITY, 0, img_size);

  float base_line = -P2.at<double>(0, 3) / P2.at<double>(0, 0);

  std::cout << "\nbase_line: \n" << base_line << std::endl;
  std::cout << "\nP1:\n" << P1 << std::endl;
  std::cout << "\nP2:\n" << P2 << std::endl;
  std::cout << "\nQ:\n" << Q << std::endl;

  std::vector<std::string> topics;
  topics.push_back(camL_topic);
  topics.push_back(camR_topic);

  rosbag::View view(bag, rosbag::TopicQuery(topics));

  double pre_ts = 1000.0;
  for (const rosbag::MessageInstance& msg : view) {
    const auto& tpn = msg.getTopic();
    // std::printf("topic: %s\n", tpn.c_str());
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
        // std::printf("calc depth: tsL: %.3f, tsR: %.3f\n", tsL, tsR);
      }
    }

    static int cnt = 0;
    if (!imgL.empty() && !imgR.empty() && cnt++ % 15 == 0) {
      // 计算并应用校正映射
      cv::Mat mapLx, mapLy, mapRx, mapRy;
      cv::initUndistortRectifyMap(camKL, camDL, R1, P1, img_size, CV_32FC1, mapLx, mapLy);
      cv::initUndistortRectifyMap(camKR, camDR, R2, P2, img_size, CV_32FC1, mapRx, mapRy);

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
      cv::reprojectImageTo3D(disp, depth, Q, true);

      std::vector<cv::Mat> depth_channels(3);
      cv::split(depth, depth_channels);
      cv::Mat z_depth = depth_channels[2];

      // z_depth.setTo(0, z_depth < 0.1);
      // z_depth.setTo(0, z_depth > 10.0);

      // cv::normalize(z_depth, depth_show, 0, 255, cv::NORM_MINMAX, CV_8U);
      // cv::applyColorMap(depth_show, depth_show, cv::COLORMAP_JET);
      procImgD(z_depth, depth_show, colorBar2);

      cv::Mat rect_img;
      cv::hconcat(imgL_rect, imgR_rect, rect_img);
      cv::line(rect_img, cv::Point(0, 200), cv::Point(rect_img.cols, 200), cv::Scalar(0, 0, 255), 1);

      // 显示结果
      cv::imshow("rect_img", rect_img);
      cv::imshow("Disparity", disp8);
      cv::imshow("Depth", depth_show);
      cv::waitKey(1000);        
    }
  }
  return 0;
}