#ifndef UTILS_COMMON_DATA_TYPE_BASIC_H
#define UTILS_COMMON_DATA_TYPE_BASIC_H

#include <Eigen/Core>
#include <Eigen/Dense>

#include "common/common_def.h"

namespace common {

struct Pose {
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();            // x, y, z; m
  Eigen::Quaterniond quat = Eigen::Quaterniond::Identity(); // Quaternion
};

struct ProbPose {
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();          // x, y, z; meter
  Eigen::Matrix3d pos_cov = Eigen::Matrix3d::Zero();

  Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
  Eigen::Matrix3d quat_cov = Eigen::Matrix3d::Zero();
};

struct ImuData {
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();    // m/s^2
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();   // rad/s

  Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
};

struct Vel3D {
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();      // m/s
  Eigen::Vector3d ang_vel = Eigen::Vector3d::Zero();  // rad/s
};

struct GnssData {
  std::string rtk_type = common::RTK_UNKNOWN;
  Eigen::Vector3d lla = Eigen::Vector3d::Zero();      // Latitude in degree, longitude in degree, and altitude in meter.
  Eigen::Vector3d enu = Eigen::Vector3d::Zero();      // meter
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();      // Covariance matrix of the lla, in meter
};

struct WheelVel {
  Eigen::Vector3d vel = Eigen::Vector3d::Zero(); // lv_x, lv_y, av_z; m/s, m/s, rad/s
};

struct Odometry {
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();         // x, y, z; m
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity(); // Quaternion
  Eigen::Vector3d linear = Eigen::Vector3d::Zero();      // vx, vy, vz; m/s
  Eigen::Vector3d angular = Eigen::Vector3d::Zero();  // wx, wy, wz; rad/s
};

struct OdomWithCov {
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();         // x, y, z; m
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity(); // Quaternion
  Eigen::Matrix3d pos_cov = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d q_cov = Eigen::Matrix3d::Zero();

  Eigen::Vector3d linear = Eigen::Vector3d::Zero();      // vx, vy, vz; m/s
  Eigen::Vector3d angular = Eigen::Vector3d::Zero();  // wx, wy, wz; rad/s
  Eigen::Matrix3d linear_cov = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d angular_cov = Eigen::Matrix3d::Zero();
};

struct VioResult {
  int vio_confidence = -1;    // -1: invalid, 0: failure, 1: low, 2: mid, 3: good
  Odometry vio;               // transformed to motion-centor
};

struct PoseWithGnss {
  double timestamp = 0.0; // sec
  bool vreloc_valid = false;
  ProbPose ppose;           // origin_vio_pose
  GnssData gnss;            // converted to ENU
  ProbPose vreloc;
  ProbPose aligned_ppose;   // 对齐后的vio_pose, 主要是当gnss无效时使用, 简单的边缘化
};

struct VioResultWithGnss {
  double timestamp = 0.0;     // sec
  int vio_confidence = -1;    // -1: invalid, 0: failure, 1: low, 2: mid, 3: good
  Odometry vio;               // transformed to motion-centor
  GnssData gnss;              // converted to ENU
};

struct PositionMeasurement {
  double timestamp = 0.0; // sec

  int type = 0;          // 0: invalid, 1: gnss, 2: vio
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();      // x, y, z; meter
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();      // Covariance matrix of pos, in meter
};

struct PoseMeasurement {
  double timestamp = 0.0; // sec

  int type = 0;          // 0: invalid, 1: gnss, 2: vio
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();          // x, y, z; meter
  Eigen::Matrix3d pos_cov = Eigen::Matrix3d::Zero();

  Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
  Eigen::Matrix3d quat_cov = Eigen::Matrix3d::Zero();
};

struct RtkGnss {
  double timestamp = 0.0; // sec
  std::string position_type = common::RTK_UNKNOWN;
  Eigen::Vector3d lla = Eigen::Vector3d::Zero();        // Latitude in degree, longitude in degree, and altitude in meter.
  Eigen::Vector3d lla_sigma = Eigen::Vector3d::Zero();  // meter

  std::string ref_station_id;
  float diff_age = 0.f;         // 差分龄期, sec
  float solution_age = 0.f;     // 解算龄期, sec
  int num_sats_tracked = 0;     // 跟踪卫星数
  int num_sats_used = 0;        // 解算使用的卫星数

  std::string ref_station_status;     // 参考站状态, "FINE" = OK, "FAIL" = 移动站没有收到参考站的数据
  int lora_rssi_dbm = -128;           // LoRa信号强度, dBm 
};

struct MotionState {
  double timestamp = 0.0;
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();              // m/s^2, 载体坐标系
  Eigen::Vector3d ang_vel = Eigen::Vector3d::Zero();          // rad/s, 载体坐标系
};

struct NavState {
  double timestamp = 0.0;
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();              // m/s^2, 载体坐标系
  Eigen::Vector3d ang_vel = Eigen::Vector3d::Zero();          // rad/s, 载体坐标系
  double off_rtk_dist = 0.0;    // 掉rtk后的移动总里程(含角程)
  double off_reloc_dist = 0.0;  // 掉vio/lio reloc后的移动总里程(含角程)
  double only_iw_dist = 0.0;    // 到当前纯imu-wheel(掉vio/lio)推算里程(含角程)

  int type = 0;  // 0: iw, 1: rtk ,2: lio/vio-reloc, 3: lio/vio,
};

struct EskfState {
  double timestamp = 0.0;
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
  Eigen::Vector3d vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();

  ImuData imu_data;
};

} // namespace common

#endif//UTILS_COMMON_DATA_TYPE_BASIC_H
