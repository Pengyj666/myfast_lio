#include "geo_utils/tf_helper.h"
#include "droslog/log.h"

namespace utils {

// JOHN_NOTE 先只管x,y,yaw
void TFHelper::SetParams_Gps2Base(double x, double y, double z, double roll, double pitch, double yaw) {
  droslog(LogLevel::INFO, "TFHelper::SetParams_Gps2Base(): xyz: %.3f %.3f %.3f, rpy: %.3f %.3f %.3f",
      x, y, z, roll, pitch, yaw);

  Gps2Base_t_ << x, y, z;
  Eigen::Matrix3d R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Gps2Base_q_ = Eigen::Quaterniond(R);
}

void TFHelper::SetParams_Lidar2Base(double x, double y, double z, double roll, double pitch, double yaw) {
  droslog(LogLevel::INFO, "TFHelper::SetParams_Lidar2Base(): xyz: %.3f %.3f %.3f, rpy: %.3f %.3f %.3f",
      x, y, z, roll, pitch, yaw);

  Lidar2Base_t_ << x, y, z;
  Eigen::Matrix3d R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Lidar2Base_q_ = Eigen::Quaterniond(R);
}

void TFHelper::SetParams_Vio2Gps(double x, double y, double z, double roll, double pitch, double yaw) {
  droslog(LogLevel::INFO, "TFHelper::SetParams_Vio2Gps(): xyz: %.3f %.3f %.3f, rpy: %.3f %.3f %.3f",
      x, y, z, roll, pitch, yaw);

  Vio2Gps_t_ << x, y, z;
  Eigen::Matrix3d R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Vio2Gps_q_ = Eigen::Quaterniond(R);
}

void TFHelper::SetParams_Lidar2Gps(double x, double y, double z, double roll, double pitch, double yaw) {
  droslog(LogLevel::INFO, "TFHelper::SetParams_Lidar2Gps(): xyz: %.3f %.3f %.3f, rpy: %.3f %.3f %.3f",
      x, y, z, roll, pitch, yaw);

  Lidar2Gps_t_ << x, y, z;
  Eigen::Matrix3d R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Lidar2Gps_q_ = Eigen::Quaterniond(R);
}

void TFHelper::SetParams_Imu2Gps(double x, double y, double z, double roll, double pitch, double yaw) {
  droslog(LogLevel::INFO, "TFHelper::SetParams_Imu2Gps(): xyz: %.3f %.3f %.3f, rpy: %.3f %.3f %.3f",
      x, y, z, roll, pitch, yaw);

  Imu2Gps_t_ << x, y, z;
  Eigen::Matrix3d R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Imu2Gps_q_ = Eigen::Quaterniond(R);
}

void TFHelper::SetParams_VioImu2Gps(double x, double y, double z, double roll, double pitch, double yaw) {
  droslog(LogLevel::INFO, "TFHelper::SetParams_VioImu2Gps(): xyz: %.3f %.3f %.3f, rpy: %.3f %.3f %.3f",
      x, y, z, roll, pitch, yaw);

  VioImu2Gps_t_ << x, y, z;
  Eigen::Matrix3d R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  VioImu2Gps_q_ = Eigen::Quaterniond(R);
}

common::Pose TFHelper::TF_Vio2Gps(const Eigen::Vector3d& t, const Eigen::Quaterniond& q) {
  common::Pose res;
  res.quat = Vio2Gps_q_.inverse() * q;
  res.pos = t - res.quat * Vio2Gps_t_;
  return res;
}

common::Pose TFHelper::TF_Lidar2Gps(const Eigen::Vector3d& t, const Eigen::Quaterniond& q) {
  common::Pose res;
  res.quat = Lidar2Gps_q_.inverse() * q;
  res.pos = t - res.quat * Lidar2Gps_t_;
  return res;
}

common::Pose TFHelper::TF_Lidar2Base(const Eigen::Vector3d& t, const Eigen::Quaterniond& q) {
  common::Pose res;
  res.quat = Lidar2Base_q_.inverse() * q;
  res.pos = t - res.quat * Lidar2Base_t_;
  return res;
}

common::Pose TFHelper::TF_Gps2Vio(const Eigen::Vector3d& t, const Eigen::Quaterniond& q) {
  common::Pose res;
  res.quat = Vio2Gps_q_ * q;
  res.pos = t + q * Vio2Gps_t_;
  return res;
}

common::Pose TFHelper::TF_Imu2Gps(const Eigen::Vector3d& t, const Eigen::Quaterniond& q) {
  common::Pose res;
  res.quat = Imu2Gps_q_.inverse() * q;
  res.pos = t - res.quat * Imu2Gps_t_;
  return res;
}

common::Pose TFHelper::TF_Gps2Base(const Eigen::Vector3d& t, const Eigen::Quaterniond& q) {
  common::Pose res;
  res.quat = Gps2Base_q_.inverse() * q;
  res.pos = t - res.quat * Gps2Base_t_;
  return res;
}

} // namespace utils