#include "common/heading_estimator.h"

#include "common/log_filters.h"
#include "common/math_utils.h"
#include "common/sysutils.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"

using namespace utils;

HeadingEstimator::HeadingEstimator() : to_stop_(true) {
  droslog(LogLevel::INFO, "HeadingEstimator::ctor() ++++++");
  imu_q_.reset(2048);
  wheel_vel_q_.reset(2048);
  gnss_q_.reset(256);
  pose_q_.reset(256);
  st_q_.reset(64);

  run_thread_ = std::thread(&HeadingEstimator::Run, this);

  droslog(LogLevel::INFO, "HeadingEstimator::ctor() ------");
}

HeadingEstimator::~HeadingEstimator() {
  droslog(LogLevel::INFO, "HeadingEstimator::dtor() ++++++");
  to_stop_.store(true);
  if (run_thread_.joinable()) {
    droslog(LogLevel::INFO, "HeadingEstimator::dtor() wait run-thread to stop");
    run_thread_.join();
    droslog(LogLevel::INFO, "HeadingEstimator::dtor() run-thread stopped");
  }
  droslog(LogLevel::INFO, "HeadingEstimator::dtor() ------");
}

void HeadingEstimator::SetConfig(const Config &config) {
  config_ = config;
}

void HeadingEstimator::FeedData(const common::Data_Gnss& gnss_data) {
  if (gnss_data.gnss.rtk_type != common::RTK_NARROW_INT) 
    return;
  static double pre_ts = 0.0;
  if (gnss_data.timestamp <= pre_ts) {
    return;
  }
  pre_ts = gnss_data.timestamp;

  std::lock_guard<std::mutex> lock(gnss_mutex_);
  gnss_q_.emplace_back(gnss_data, gnss_data.timestamp);
}

void HeadingEstimator::FeedData(const common::Data_Imu& imu_data) {
  std::lock_guard<std::mutex> lock(imu_mutex_);
  imu_q_.emplace_back(imu_data, imu_data.timestamp);
}

void HeadingEstimator::FeedData(const common::Data_WheelVel& wheel_vel) {
  std::lock_guard<std::mutex> lock(wheel_vel_mutex_);
  wheel_vel_q_.emplace_back(wheel_vel, wheel_vel.timestamp);
}

void HeadingEstimator::FeedData(const common::Data_Pose& pose) {
  std::lock_guard<std::mutex> lock(pose_mutex_);
  pose_q_.emplace_back(pose, pose.timestamp);
}

common::Data_Pose HeadingEstimator::GetEnuHeading() {
  std::lock_guard<std::mutex> lock(eq_mutex_);
  return enu_quat_;
}

void HeadingEstimator::Run() {
  droslog(LogLevel::INFO, "HeadingEstimator::Run() ++++++");
  utils::TimedQueue<StateTic> st_q;
  st_q.reset(1024);

  common::Data_Gnss pre_proc_gnss;
  to_stop_.store(false);
  while (!to_stop_.load()) {
    Sleep(50);

    common::Data_Gnss gnss;
    {
      std::lock_guard<std::mutex> lock(gnss_mutex_);
      if (gnss_q_.size() > 0 && gnss_q_(0) > pre_proc_gnss.timestamp) {
        double dx = gnss_q_[0].gnss.enu[0] - pre_proc_gnss.gnss.enu[0];
        double dy = gnss_q_[0].gnss.enu[1] - pre_proc_gnss.gnss.enu[1];
        double dist = std::sqrt(dx * dx + dy * dy);
        if (dist > config_.line_heading_KF_dist) {
          gnss = gnss_q_[0];
          pre_proc_gnss = gnss;
        }
      }
    }

    if (gnss.timestamp <= 0.0)
      continue;
    
    // 生成关键帧
    StateTic st;
    st.timestamp = gnss.timestamp;
    st.enu = gnss.gnss.enu;
    bool st_ok = true;
    {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      int idx = imu_q_.findAfter(st.timestamp);
      if (idx >= 0 && std::abs(st.timestamp - imu_q_(idx)) < 0.1) {
        st.quat = imu_q_[idx].imu.quat;
      } else {
        st_ok = false;
        static SimpleLogFilter log_filter(2000);
        if (log_filter.Output(GetNow_Steady())) {
          droslog(LogLevel::WARN, "HeadingEstimator::Run() 没有查到对应的imu数据, gnss.ts=%.3f, idx=%d", st.timestamp, idx);
        }
      }
    }
    {
      std::lock_guard<std::mutex> lock(wheel_vel_mutex_);
      int idx = wheel_vel_q_.findAfter(st.timestamp);
      if (idx >= 0 && std::abs(st.timestamp - wheel_vel_q_(idx)) < 0.1) {
        st.vel = wheel_vel_q_[idx].vel.vel;
      } else {
        st_ok = false;
        static SimpleLogFilter log_filter(2000);
        if (log_filter.Output(GetNow_Steady())) {
          droslog(LogLevel::WARN, "HeadingEstimator::Run() 没有查到对应的wheel_vel数据, gnss.ts=%.3f, idx=%d", st.timestamp, idx);
        }
      }
    }

    if (!st_ok) 
      continue;
    
    st.rpy = GetEulerRPY(st.quat);
    st_q_.emplace_back(st, st.timestamp);

    if (st_q_.size() < config_.line_heading_KF_window) {
      if (config_.debug_log) {
        droslog(LogLevel::INFO, "HeadingEstimator::Run() st_q_.size() < KF_window, st_q_.size()=%d", st_q_.size());
      }
      continue;
    }

    // 有新的关键帧进来, 估计一次
    // 直线度检查
    // 1. 距离检查
    auto tail_st = st_q_[config_.line_heading_KF_window - 1];
    double dx = tail_st.enu[0] - st.enu[0];
    double dy = tail_st.enu[1] - st.enu[1];
    double dist = std::sqrt(dx * dx + dy * dy);
    if (dist < config_.line_heading_dist_min || dist > config_.line_heading_dist_max) {
      if (config_.debug_log) {
        droslog(LogLevel::INFO, "HeadingEstimator::Run() dist=%.3f, too long or too short", dist);
      }
      continue;
    }
    // 2. 检查速度，均值 > 0.1, 最小值 > -0.1
    bool vel_ok = true;
    double vel_sum = 0.0;
    double max_dyaw = 0.0;
    double min_dyaw = 0.0;
    for (int i = 0; i < config_.line_heading_KF_window; i++) {
      auto &st_i = st_q_[i];
      if (st_i.vel[0] < -0.1) {
        vel_ok = false;
        break;
      }
      vel_sum += st_i.vel[0];

      double tdyaw = KeepAngleInPI(st_i.rpy[2] - st.rpy[2]);
      max_dyaw = std::max(max_dyaw, tdyaw);
      min_dyaw = std::min(min_dyaw, tdyaw);
    }

    if (vel_ok) {
      vel_sum /= config_.line_heading_KF_window;
    } else {
      continue;
    }

    if (vel_sum < 0.1) {
      continue;
    }
    // 3. 检查角度，极差, 首尾差
    double dyaw = KeepAngleInPI(st.rpy[2] - tail_st.rpy[2]);
    if (dyaw > config_.line_heading_imu_sum_yaw) {
      if (config_.debug_log) {
        droslog(LogLevel::INFO, "HeadingEstimator::Run() 本段首尾航向角偏差较大 dyaw=%.3f", dyaw);
      }
      continue;
    }
    double dyaw_maxmin = max_dyaw - min_dyaw;
    if (dyaw_maxmin > config_.line_heading_imu_diff_yaw) {
      if (config_.debug_log) {
        droslog(LogLevel::INFO, "HeadingEstimator::Run() 本段航向角极差较大 max_dyaw=%.3f min_dyaw=%.3f", max_dyaw, min_dyaw);
      }
      continue;
    }
    // 4. 直线度检查通过，计算航向角
    double yaw = get_yaw(st.enu[0], st.enu[1], tail_st.enu[0], tail_st.enu[1]);

    // {
    //   static SimpleLogFilter log_filter(2000);
    //   if (log_filter.Output(GetNow_Steady())) {
    //     droslog(LogLevel::INFO, "HeadingEstimator::Run() 直线段航向角: ts=%.3f, yaw=%.3f, dist=%.3f, dyaw_maxmin=%.3f, dyaw_headtail=%.3f, enu0=%.3f,%.3f,%.3f -> enu1=%.3f,%.3f,%.3f", 
    //         st.timestamp, yaw, dist, dyaw_maxmin, dyaw, tail_st.enu[0], tail_st.enu[1], tail_st.enu[2], st.enu[0], st.enu[1], st.enu[2]);
    //   }
    // }
    
    std::lock_guard<std::mutex> lock(eq_mutex_);
    enu_quat_.timestamp = st.timestamp;
    enu_quat_.pose.quat = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    enu_quat_.pose.pos = st.enu;
  }
  droslog(LogLevel::INFO, "HeadingEstimator::Run() ------");
}