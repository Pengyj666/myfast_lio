#ifndef UTILS_ODOM_SUMMER_H
#define UTILS_ODOM_SUMMER_H

#include <deque>
#include <atomic>
#include <Eigen/Core>

#include "droslog/log.h"
#include "common/common_def.h"
#include "common/data_type.h"
#include "common/sysutils.h"
#include "common/timed_queue.h"
#include "geo_utils/geo_utils.h"

namespace utils {

// 一个简易的Gnss数据状态监控器，只用于跟踪RTK固定解情况
class OdomSummer {
 public:
  OdomSummer(double max_dd = 10.0) : last_pose_(common::Data_Pose()), max_dd_(max_dd), dist_sum_(0.0), dyaw_sum_(0.0) {}

  void Reset() {
    last_pose_ = common::Data_Pose();
    dist_sum_.store(0.0);
    dyaw_sum_.store(0.0);
  }

  void Update(const double &ts, const Eigen::Vector3d &pos, const Eigen::Quaterniond &q) {
    if (last_pose_.timestamp <= 0.0) {
      last_pose_.timestamp = ts;
      last_pose_.pose.pos = pos;
      last_pose_.pose.quat = q;
      return;
    } else {
      double ddist = (pos - last_pose_.pose.pos).norm();
      Eigen::Quaterniond dq = q.inverse() * last_pose_.pose.quat;
      double dyaw = GetEulerRPY(dq).norm();

      dist_sum_.store(dist_sum_.load() + ddist);
      dyaw_sum_.store(dyaw_sum_.load() + dyaw);      
    }
  }

  double GetDistSum() { return dist_sum_.load(); }
  double GetYawSum() { return dyaw_sum_.load(); }

  double GetRemainDist() const { return max_dd_ - dist_sum_.load() - dyaw_sum_.load() * 0.5; }

 private:
  common::Data_Pose last_pose_;
  double max_dd_ = 10.0;
  std::atomic<double> dist_sum_;
  std::atomic<double> dyaw_sum_;
};

}  // namespace utils

#endif  // UTILS_ODOM_SUMMER_H
