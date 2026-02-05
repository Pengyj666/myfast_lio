#ifndef COMMON_HEADING_ESTIMATOR_H
#define COMMON_HEADING_ESTIMATOR_H

#include <atomic>
#include <mutex>
#include <queue>
#include <thread>
#include <Eigen/Core>

#include "common/data_type.h"
#include "common/data_utils.h"
#include "common/timed_queue.h"

// 用来纠rtk-imu-eskf
// 1. rtk成线点, 按一定间距取点, 作为关键帧
// 2. 关键帧: ts, enu, imu(rpy,acc,gyro), wheel_vel
// 3. 

class HeadingEstimator {
 public:
  struct Config {
    double line_heading_KF_dist = 0.05;          // 直线段估计航向角: rtk关键帧间隔, m
    int line_heading_KF_window = 10;          // 直线段估计航向角: rtk关键帧窗口大小, 个
    double line_heading_dist_min = 0.3;          // 直线段估计航向角: rtk首尾间隔最小距离, m
    double line_heading_dist_max = 1.5;          // 直线段估计航向角: rtk首尾间隔最大距离, m
    double line_heading_imu_diff_yaw = 0.17;     // 直线段估计航向角: imu航向角离散极差, 偏离均值的最大值, rad 
    double line_heading_imu_sigma_yaw = 0.08;    // 直线段估计航向角: imu航向角离散标准差, rad
    double line_heading_imu_sum_yaw = 0.08;      // 直线段估计航向角: imu航向角头尾差, rad
    bool debug_log = false;
  };

  struct StateTic {
    double timestamp = 0.0;
    Eigen::Vector3d enu = Eigen::Vector3d::Zero();      // 局部enu坐标
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();      // 轮速, lv_x, lv_y, av_z
    Eigen::Vector3d rpy = Eigen::Vector3d::Zero();      // imu自融合姿态, 欧拉角
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();  // IMU自融合姿态, 取相对姿态用
  };

  struct MoveState {
    double timestamp = 0.0;
    double r = 1000.0;
  };

  static HeadingEstimator* Instance() {
    static HeadingEstimator ins;
    return &ins;
  }
  ~HeadingEstimator();

  void SetConfig(const Config &config);

  void FeedData(const common::Data_Gnss& gnss_data);
  void FeedData(const common::Data_Imu& imu_data);
  void FeedData(const common::Data_WheelVel& wheel_vel);
  // eskf 的位姿状态, 用于校验是否航向角偏移, 已经转到局部enu
  void FeedData(const common::Data_Pose& pose);

  // result: pos 为局部ENU, quat 为局部ENU下的相对东向的姿态
  common::Data_Pose GetEnuHeading();

  const Config& config() const { return config_; }

 private:
  HeadingEstimator();
  HeadingEstimator(const HeadingEstimator&) = delete;
  HeadingEstimator& operator=(const HeadingEstimator&) = delete;

  void Run();
  std::atomic_bool to_stop_;
  std::thread run_thread_;

  Config config_;

  std::mutex eq_mutex_;
  common::Data_Pose enu_quat_;  // 估计的enu下的位姿

  utils::TimedQueue<StateTic> st_q_;

  std::mutex gnss_mutex_;
  utils::TimedQueue<common::Data_Gnss> gnss_q_;
  std::mutex imu_mutex_;
  utils::TimedQueue<common::Data_Imu> imu_q_;
  std::mutex wheel_vel_mutex_;
  utils::TimedQueue<common::Data_WheelVel> wheel_vel_q_;
  std::mutex pose_mutex_;
  utils::TimedQueue<common::Data_Pose> pose_q_;   // eskf
};
  
#endif //COMMON_HEADING_ESTIMATOR_H