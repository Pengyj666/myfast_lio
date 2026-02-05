#include "common/dist_odometer.h"

#include "droslog/log.h"

using namespace utils;

DistOdometer::DistOdometer() {
  dist_.reset(8192);
}

DistOdometer::~DistOdometer() {}

void DistOdometer::Update(double ts, const Eigen::Vector3d &vel, const Eigen::Vector3d &ang_vel) {
  if (ts <= 0.0) {
    return;
  }

  std::lock_guard<std::mutex> lock(mtx_);
  if (dist_.size() == 0) {
    DistOdom odom;
    odom.timestamp = ts;
    dist_.emplace_back(odom, ts);
  } else {
    double pre_ts = dist_(0);
    if (ts > pre_ts) {
      auto pre_odom = dist_[0];
      double dist = 0.5 * (pre_odom.vel + vel).norm() * (ts - pre_ts);
      double ang_dist = 0.5 * (pre_odom.ang_vel + ang_vel).norm() * (ts - pre_ts);
      DistOdom odom;
      odom.timestamp = ts;
      odom.vel = vel;
      odom.ang_vel = ang_vel;
      odom.dist = pre_odom.dist + dist;
      odom.ang_dist = pre_odom.ang_dist + ang_dist;
      dist_.emplace_back(odom, ts);

      static double log_ts = 0.0;
      if (ts > log_ts + 5.0) {
        droslog(LogLevel::INFO, "DistOdometer::Update() dist_odom: %.3f", odom.dist);
        log_ts = ts;
      }
    }
  }
}

DistOdometer::DistOdom DistOdometer::dist_odom(double ts) {
  std::lock_guard<std::mutex> lock(mtx_);
  int idx = dist_.findAfter(ts);
  if (idx > 0) {
    return dist_[idx - 1];
  } else if (idx == 0) {
    return dist_[0];
  } 
  
  return DistOdom();
}

