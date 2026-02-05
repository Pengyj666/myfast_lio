#ifndef COMMON_DIST_ODOMETER_H
#define COMMON_DIST_ODOMETER_H

#include <mutex>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "common/timed_queue.h"

// 基于速度的里程累计
class DistOdometer {
 public:
  struct DistOdom {
    double timestamp = 0.0;
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d ang_vel = Eigen::Vector3d::Zero();

    double dist = 0.0;
    double ang_dist = 0.0;
  };

  DistOdometer();
  ~DistOdometer();

  void Update(double ts, const Eigen::Vector3d &vel, const Eigen::Vector3d &ang_vel);

  DistOdom dist_odom(double ts = 0.0);

 private:
  std::mutex mtx_;
  utils::TimedQueue<DistOdom> dist_;
};

#endif// COMMON_DIST_ODOMETER_H