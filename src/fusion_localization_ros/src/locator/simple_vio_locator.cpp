#include "locator/simple_vio_locator.h"

#include "common/sysutils.h"
#include "common/log_filters.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"
#include "geo_utils/tf_helper.h"

#include "common/debug_client.h"
#include "common/gnss_converter.h"
#include "common/sensor_monitor.h"
#include "common/vio_tracker.h"
#include "common/vio_gnss_initor.h"
#include "common/vio_checker.h"

#include <ros/ros.h>

using namespace utils;

SimpleVioLocator::SimpleVioLocator() {
  droslog(LogLevel::INFO, "SimpleVioLocator::ctor() ++++++");
  imu_q_.reset(8192);
  wheel_vel_q_.reset(8192);
  vio_q_.reset(1024);
  vmap_odom_q_.reset(1024);
  state_tracker_valid_.store(false);

  work_mode_.store(-1);
  work_state_.store(-1);
  error_code_.store(0);

  droslog(LogLevel::INFO, "SimpleVioLocator::ctor() ------");
}

SimpleVioLocator::~SimpleVioLocator() {
  droslog(LogLevel::INFO, "SimpleVioLocator::dtor() ++++++");
  droslog(LogLevel::INFO, "SimpleVioLocator::dtor() ------");
}

void SimpleVioLocator::Reset() {
  droslog(LogLevel::INFO, "SimpleVioLocator::Reset() ++++++");
  state_tracker_valid_.store(false);

  work_mode_.store(-1);
  work_state_.store(-1);
  error_code_.store(0);

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_tracker_.Reset();
  }

  droslog(LogLevel::INFO, "SimpleVioLocator::Reset() ------");
}

void SimpleVioLocator::SetConfig(const Config& config) {
  droslog(LogLevel::INFO, "SimpleVioLocator::SetConfig() ++++++");
  config_ = config;
  droslog(LogLevel::INFO, "SimpleVioLocator::SetConfig() ------");
}

const SimpleVioLocator::Config& SimpleVioLocator::GetConfig() const {
  return config_;
}

bool SimpleVioLocator::IsValid() {
  if (state_tracker_valid_.load()) {
    auto ns = GetNavState();
    if (ns.off_rtk_dist < config_.max_off_rtk_dist && ns.off_reloc_dist < config_.max_off_reloc_dist && ns.only_iw_dist < config_.max_only_iw_dist) {
      return true;
    }
  }
  return false;
}

bool SimpleVioLocator::SetWorkMode(int mode) {
  droslog(LogLevel::INFO, "SimpleVioLocator::SetWorkMode() set mode: %d -> %d, state %d -> 0", 
      work_mode_.load(), mode, work_state_.load());
  work_mode_.store(mode);
  return false;
}

bool SimpleVioLocator::SetWorkState(int state, int err_code) {
  droslog(LogLevel::WARN, "SimpleVioLocator::SetWorkState() state: %d -> %d, err_code=%d", work_state_.load(), state, err_code);
  work_state_.store(state);
  return true;
}

common::NavState SimpleVioLocator::GetNavState(double timestamp) {
  common::NavState nav_state;
  if (state_tracker_valid_.load()) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    nav_state = state_tracker_.GetState(timestamp);
  }
  return nav_state; 
}

bool SimpleVioLocator::SetInitNavState(const common::NavState &state) {
  if (state.timestamp <= 0) {
    droslog(LogLevel::ERROR, "SimpleVioLocator::SetInitNavState() timestamp is invalid: %.3f", state.timestamp);
    return false;
  }
  auto rpy = GetEulerRPY(state.quat);
  droslog(LogLevel::INFO, "SimpleVioLocator::SetInitNavState() 定位器设置初始化状态 ts=%.3f, pos=%.3f, %.3f, %.3f, rpy=%.3f, %.3f, %.3f", 
      state.timestamp, state.pos.x(), state.pos.y(), state.pos.z(), rpy[0], rpy[1], rpy[2]);
  
  // TODO 这里先把缓存数据提取出来, 处理缓存的所有imu-wheel数据
  std::vector<common::Data_Imu> imu_cache_;    
  {
    double next_ts = state.timestamp - 0.03;
    std::lock_guard<std::mutex> lock(imu_q_mutex_);
    int ind = imu_q_.findAfter(next_ts);
    if (ind > 0) {
      for (; ind >= 0; --ind) {
        imu_cache_.push_back(imu_q_[ind]);
      }
    }
  }
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_tracker_.InitOrigin(state);
  }
  droslog(LogLevel::INFO, "SimpleVioLocator::SetInitNavState() 处理缓存数据, imu_cache_.size()=%d", imu_cache_.size());
  for (int i = 0; i < imu_cache_.size(); ++i) {
    auto imu = imu_cache_[i];
    auto wvel = EstimateWheelVel(imu.timestamp);
    if (wvel.timestamp > 0.0) {
      std::lock_guard<std::mutex> lock(state_mutex_);
      state_tracker_.Predict(imu.timestamp, imu.imu, wvel.vel);
    } else {
      droslog(LogLevel::ERROR, "SimpleVioLocator::SetInitNavState() 处理缓存: wheel_vel invalid: imu_ts=%.3f, wheel.last_ts=%.3f, wheel.size=%d",
          imu.timestamp, wheel_vel_q_.last_timstamp(), (int)wheel_vel_q_.size());
    }
  }
  state_tracker_valid_.store(true);
  SetWorkState(3);
  return true;
}

bool SimpleVioLocator::ProcessImuData(const common::Data_Imu &imu) {
  static double pre_ts = 0.0;
  if (imu.timestamp > pre_ts) {
    pre_ts = imu.timestamp;
    {
      std::lock_guard<std::mutex> lock(imu_q_mutex_);
      imu_q_.emplace_back(imu, imu.timestamp);
    }
    if (work_state_.load() >= 0) {
      // ns_tracker跟踪
      auto wvel = EstimateWheelVel(imu.timestamp);
      if (wvel.timestamp > 0.0) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_tracker_.Predict(imu.timestamp, imu.imu, wvel.vel);
      } else {
        static SimpleLogFilter log_filter(1000);
        if (log_filter.Output(GetNow_Steady())) {
          droslog(LogLevel::ERROR, "SimpleVioLocator::ProcessImuData: wheel_vel invalid: imu_ts=%.3f, wheel.last_ts=%.3f, wheel.size=%d",
              imu.timestamp, wheel_vel_q_.last_timstamp(), (int)wheel_vel_q_.size());
        }
      }
    }
    return true;
  }
  return false;
}

bool SimpleVioLocator::ProcessGpsData(const common::Data_Gnss &gnss) {
  static double pre_ts = 0.0;
  if (gnss.timestamp <= pre_ts) {
    return false;
  }
  pre_ts = gnss.timestamp;

  if (!state_tracker_valid_.load()) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    int ret = state_tracker_.Measurement(gnss.timestamp, gnss.gnss.enu, 1);
    return true;
  }

  return true;
}

bool SimpleVioLocator::ProcessEstHeading(double ts, const Eigen::Quaterniond &quat) {
  static double pre_ts = 0.0;
  if (ts <= pre_ts) {
    return false;
  }
  pre_ts = ts;

  if (!state_tracker_valid_.load()) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    int ret = state_tracker_.Measurement(ts, quat, 1);
    return true;
  }

  return true;
}

bool SimpleVioLocator::ProcessWheelData(const common::Data_WheelVel &wheel_vel) {
  static double pre_ts = 0.0;
  if (wheel_vel.timestamp > pre_ts) {
    pre_ts = wheel_vel.timestamp;
    {
      std::lock_guard<std::mutex> lock(wheel_vel_q_mutex_);
      wheel_vel_q_.emplace_back(wheel_vel, wheel_vel.timestamp);
    }

    return true;
  }
  return false;
}

bool SimpleVioLocator::ProcessVioData(const common::Data_VioResult& vio, int type) {
  if (!state_tracker_valid_.load()) {
    return false;
  }

  common::NavState cns;

  auto rpy = GetEulerRPY(vio.vio.q);
  
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    cns = state_tracker_.GetState();
    if (cns.off_rtk_dist < 1.5) {
      auto nsrpy = GetEulerRPY(cns.quat);
      static SimpleLogFilter log_filter(2000);
      if (log_filter.Output(GetNow_Steady())) {
        droslog(LogLevel::INFO, "SimpleVioLocator::ProcessVioData(): rtk跟踪时 vio姿态修正: (%.3f, %.3f, %.3f) -> (%.3f, %.3f, %.3f)", 
            nsrpy[0], nsrpy[1], nsrpy[2], rpy[0], rpy[1], rpy[2]);
      }
      int ret = state_tracker_.Measurement(vio.timestamp, vio.vio.q, 3);
      return false;
    }
  }

  int vio_type = 0;
  if (type == 1) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    int ret = state_tracker_.Measurement(vio.timestamp, vio.vio.pos, vio.vio.q, 4);
    vio_type = 1;
  } else if (type == 0) {
    if (cns.off_reloc_dist > 2.0) {
      std::lock_guard<std::mutex> lock(state_mutex_);
      int ret = state_tracker_.Measurement(vio.timestamp, vio.vio.pos, vio.vio.q, 2);
      vio_type = 2;
    }
  }
  
  {
    static SimpleLogFilter log_filter(2000);
    if (log_filter.Output(GetNow_Steady())) {
      if (vio_type == 0) {
        droslog(LogLevel::INFO, "SimpleVioLocator::ProcessVioData(): 掉了RTK修正, 但未使用VIO修正, type=%d, ts=%.3f, off_reloc_dist=%.3f", type, vio.timestamp, cns.off_reloc_dist);
      } else if (vio_type == 1) {
        droslog(LogLevel::INFO, "SimpleVioLocator::ProcessVioData(): 掉了RTK修正, 切到VIO重定位修正, ts=%.3f", vio.timestamp);
      } else if (vio_type == 2) {
        droslog(LogLevel::INFO, "SimpleVioLocator::ProcessVioData(): 掉了RTK修正, 切到VIO修正, ts=%.3f, off_reloc_dist=%.3f, pos=%.3f, %.3f, %.3f, rpy=%.3f, %.3f, %.3f", 
            vio.timestamp, cns.off_reloc_dist, vio.vio.pos[0], vio.vio.pos[1], vio.vio.pos[2], rpy[0], rpy[1], rpy[2]);
      }
    }
  }
  return vio_type > 0;
}

bool SimpleVioLocator::ProcessLioData(const common::Data_ProbPose& lio, int type) {
  if (!state_tracker_valid_.load()) {
    return false;
  }
  // if (type == 1) {
  //   // 暂时只处理vmap_odom
  //   std::lock_guard<std::mutex> lock(state_mutex_);
  //   int ret = state_tracker_.Measurement(lio.timestamp, lio.ppose.pos, lio.ppose.quat, 5);
  //   return true;
  // } else if (type == 0) {
  //   if (work_mode_.load() == 0) {
  //     std::lock_guard<std::mutex> lock(state_mutex_);
  //     int ret = state_tracker_.Measurement(lio.timestamp, lio.ppose.pos, lio.ppose.quat, 5);
  //   }
  //   return true;
  // }
  // return false;

  common::NavState cns;

  auto rpy = GetEulerRPY(lio.ppose.quat);
  
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    cns = state_tracker_.GetState();
    if (cns.off_rtk_dist < 1.5) {
      auto nsrpy = GetEulerRPY(cns.quat);
      static SimpleLogFilter log_filter(2000);
      if (log_filter.Output(GetNow_Steady())) {
        droslog(LogLevel::INFO, "SimpleVioLocator::ProcessLioData(): rtk跟踪时 Lio姿态修正: (%.3f, %.3f, %.3f) -> (%.3f, %.3f, %.3f)", 
            nsrpy[0], nsrpy[1], nsrpy[2], rpy[0], rpy[1], rpy[2]);
      }
      int ret = state_tracker_.Measurement(lio.timestamp, lio.ppose.quat, 3);
      return false;
    }
  }

  int lio_type = 0;
  if (type == 1) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    int ret = state_tracker_.Measurement(lio.timestamp, lio.ppose.pos, lio.ppose.quat, 4);
    lio_type = 1;
  } else if (type == 0) {
    if (cns.off_reloc_dist > 2.0) {
      std::lock_guard<std::mutex> lock(state_mutex_);
      int ret = state_tracker_.Measurement(lio.timestamp, lio.ppose.pos, lio.ppose.quat, 2);
      lio_type = 2;
    }
  }
  
  {
    static SimpleLogFilter log_filter(2000);
    if (log_filter.Output(GetNow_Steady())) {
      if (lio_type == 0) {
        droslog(LogLevel::INFO, "SimpleVioLocator::ProcessLioData(): 掉了RTK修正, 但未使用LIO修正, type=%d, ts=%.3f, off_reloc_dist=%.3f", type, lio.timestamp, cns.off_reloc_dist);
      } else if (lio_type == 1) {
        droslog(LogLevel::INFO, "SimpleVioLocator::ProcessLioData(): 掉了RTK修正, 切到LIO重定位修正, ts=%.3f", lio.timestamp);
      } else if (lio_type == 2) {
        droslog(LogLevel::INFO, "SimpleVioLocator::ProcessLioData(): 掉了RTK修正, 切到LIO修正, ts=%.3f, off_reloc_dist=%.3f, pos=%.3f, %.3f, %.3f, rpy=%.3f, %.3f, %.3f", 
            lio.timestamp, cns.off_reloc_dist, lio.ppose.pos[0], lio.ppose.pos[1], lio.ppose.pos[2], rpy[0], rpy[1], rpy[2]);
      }
    }
  }
  return lio_type > 0;
}

bool SimpleVioLocator::SetComputeHeadingState(int state) {
  return true;
}

common::Data_Gnss SimpleVioLocator::GetGnssInit() {
  std::lock_guard<std::mutex> lock(gnss_init_mutex_);
  return gnss_init_;
}

common::Data_WheelVel SimpleVioLocator::EstimateWheelVel(double ts) {
  std::lock_guard<std::mutex> lock(wheel_vel_q_mutex_);
  if (wheel_vel_q_.size() < 2) {
    return common::Data_WheelVel();
  }
  if (ts <= 0.0) {
    return wheel_vel_q_[0];
  } else {
    int idx = wheel_vel_q_.findAfter(ts);
    if (idx > 0) {
      common::Data_WheelVel ret = wheel_vel_q_[idx - 1];
      ret.timestamp = ts;
      ret.vel.vel = (ret.vel.vel + wheel_vel_q_[idx].vel.vel) * 0.5;
      return ret;
    } else if (idx == 0) {
      return wheel_vel_q_[0];
    }
  }

  return common::Data_WheelVel();
}

void SimpleVioLocator::CorrectNavTracker(int type) {
}
