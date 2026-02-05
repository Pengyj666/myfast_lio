#include "common/simple_tracker.h"

#include "common/log_filters.h"
#include "common/sysutils.h"
#include "droslog/log.h"
#include "geo_utils/tf_helper.h"

using namespace common;
using namespace utils;

namespace 
{
  const int k_capacity = 4096;
} // namespace 

void SimpleTracker::Reset() {
  // states_.reset(0);
}

double SimpleTracker::GetLastTimestamp() const {
  return (states_.size() > 0) ? states_(0) : 0.0;
}

common::NavState SimpleTracker::GetState(double ts) const {
  if (states_.size() > 0) {
    if (ts <= 0.0) {
      return states_[0];
    } else {
      NavState result;
      int idx = EstimateState(ts, &result, nullptr);
      if (idx == 0 || idx == 1) {
        return result;
      }
    }
  }
  return NavState();
}

bool SimpleTracker::InitOrigin(const NavState &state) {
  if (state.timestamp <= 0.0) {
    droslog(LogLevel::ERROR, "SimpleTracker::InitOrigin() Invalid timestamp: %.3ff", state.timestamp);
    return false;
  }

  states_.reset(k_capacity);
  states_.emplace_back(state, state.timestamp);
  return true;
}

bool SimpleTracker::Predict(double ts, const common::ImuData &imu, const common::WheelVel &wvel) {
  if (states_.size() > 0 && ts > states_(0) + 0.001) {
    if (ts > states_(0) + 0.05) {
      droslog(LogLevel::WARN, "SimpleTracker::Predict() Large time gap: %.3f, pre_ts=%.3f, cur_ts=%.3f", ts - states_(0), states_(0), ts);
    }

    Eigen::Vector3d vel = wvel.vel;
    Eigen::Vector3d ang_vel = imu.gyro;

    dist_odometer_.Update(ts, vel, ang_vel);

    auto res = UpdateByVel(states_[0], vel, ang_vel, ts);
    if (res.timestamp > 0.0) {
      return states_.emplace_back(res, ts);
    } else {
      droslog(LogLevel::WARN, "SimpleTracker::Predict(): UpdateByVel failed");
    }
  }
  return false;
}

namespace {
  const double k_ratio_ref = 0.2;
  const double k_ratio_vio = 0.1;

  double type_to_ratio(int type) {
    if (type == 1) {
      return k_ratio_ref;
    } else if (type == 2 || type == 3) {
      // vio-gnss
      return k_ratio_vio;
    } 
    return 0.1;
  }

  double type_to_ratio_quat(int type) {
    return 0.05;
  }

  Eigen::Vector3d SmothPos(const Eigen::Vector3d &p_o, const Eigen::Vector3d &p_c, int type) {
    double rr = type_to_ratio(type);
    return p_o + rr * (p_c - p_o);
  }

  Eigen::Quaterniond SmothQuat(const Eigen::Quaterniond &q_o, const Eigen::Quaterniond &q_c, int type) {
    double rr = type_to_ratio_quat(type);
    Eigen::Quaterniond q = q_o.slerp(rr, q_c);
    q.normalize();
    return q;
  }
} // namespace 

int  SimpleTracker::Measurement(double ts, const Eigen::Vector3d &org_pos, int type) {
  Eigen::Vector3d pos = org_pos;
  if (pos.hasNaN()) {
    droslog(LogLevel::WARN, "SimpleTracker::Measurement(p): 操 马德, 你传进来的数据有毒");
    return -1;
  }
  
  NavState org_ns;
  int idx = -1;
  int ret = EstimateState(ts, &org_ns, &idx);
  {
    static SimpleLogFilter log_filter(2000);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::WARN, "SimpleTracker::Measurement(p): type=%d, ts=%.3f, ret=%d, idx=%d", type, ts, ret, idx);
    }
  }

  if (ret == 0) {
    auto org_dist_odom = dist_odometer_.dist_odom(ts);

    // 如果是gps, 转一下pos
    if (type == 1) {
      auto base_pose = TFHelper::Instance()->TF_Gps2Base(org_pos, org_ns.quat);
      pos = base_pose.pos;
    }

    // 计算位姿变换
    // 将后续的位姿均更新
    NavState cns = org_ns;
    cns.pos = SmothPos(org_ns.pos, pos, type);
    cns.quat = org_ns.quat;
    for (int i = idx -1; i >= 0; --i) {
      // 计算相对位姿
      auto new_ns = UpdateByVel(cns, states_[i].vel, states_[i].ang_vel, states_[i].timestamp);
      if ((new_ns.timestamp - states_[i].timestamp) > 1e-6) {
        droslog(LogLevel::WARN, "SimpleTracker::Measurement(p): 修正出错, 时间戳不对, new_ts=%.3f, ts=%.3f", new_ns.timestamp, states_[i].timestamp);
        return -1;
      }

      states_[i] = new_ns;
      cns = states_[i];

      // 更新其他状态
      auto dist_odom = dist_odometer_.dist_odom(states_(i));
      if (type == 1) {  // rtk
        states_[i].off_rtk_dist = 0.0;
        states_[i].off_reloc_dist = 0.0;
        states_[i].only_iw_dist = 0.0;
      } else if (type == 4 || type == 5) { // vio/lio reloc
        states_[i].off_reloc_dist = 0.0;
        states_[i].only_iw_dist = 0.0;
      } else if (type == 2 || type == 3) { // vio/lio
        states_[i].only_iw_dist = 0.0;
      }
    }
  } else if (ret == 1) {
    // 如果是gps, 转一下pos
    if (type == 1) {
      auto base_pose = TFHelper::Instance()->TF_Gps2Base(org_pos, states_[0].quat);
      pos = base_pose.pos;
    }

    if (ts - states_(0) < 0.002) {
      states_[0].pos = SmothPos(states_[0].pos, pos, type);
      states_[0].only_iw_dist = 0;
      if (type == 1) {
        states_[0].off_rtk_dist = 0;
        states_[0].off_reloc_dist = 0;
      } else if(type == 4 || type == 5) {
        states_[0].off_reloc_dist = 0;
      }
      return 0;
    } else if (ts - states_(0) < 0.05) {
      NavState ns = states_[0];
      ns.pos = SmothPos(ns.pos, pos, type);
      ns.timestamp = ts;
      states_[0].only_iw_dist = 0;
      if (type == 1) {
        states_[0].off_rtk_dist = 0;
        states_[0].off_reloc_dist = 0;
      } else if(type == 4 || type == 5) {
        states_[0].off_reloc_dist = 0;
      }
      states_.emplace_back(ns, ts);
      return 0;
    }
  } else {
    static SimpleLogFilter log_filter(1000);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::WARN, "SimpleTracker::Measurement(p): EstimateState error: ts=%.3f, ret=%d, idx=%d", ts, ret);
    }
  }

  return ret;
}

int  SimpleTracker::Measurement(double ts, const Eigen::Quaterniond &quat, int type) {
  if (quat.matrix().hasNaN()) {
    droslog(LogLevel::WARN, "SimpleTracker::Measurement(q): 操 马德, 你传进来的数据有毒");
    return -1;
  }
  
  NavState org_ns;
  int idx;
  int ret = EstimateState(ts, &org_ns, &idx);
  if (ret == 0) {
    // 计算位姿变换
    // 将后续的位姿均更新
    NavState cns = org_ns;
    cns.pos = org_ns.pos;
    cns.quat = SmothQuat(org_ns.quat, quat, type);
    for (int i = idx -1; i >= 0; --i) {
      // 计算相对位姿
      auto new_ns = UpdateByVel(cns, states_[i].vel, states_[i].ang_vel, states_[i].timestamp);
      if ((new_ns.timestamp - states_[i].timestamp) > 1e-6) {
        droslog(LogLevel::WARN, "SimpleTracker::Measurement(q): 修正出错, 时间戳不对, new_ts=%.3f, ts=%.3f", new_ns.timestamp, states_[i].timestamp);
        return -1;
      }

      states_[i].quat = new_ns.quat;  // 仅仅更新姿态
      cns = states_[i];      
    }
  } else if (ret == 1) {
    if (ts - states_(0) < 0.002) {
      states_[0].quat = SmothQuat(states_[0].quat, quat, type);
      return 0;
    } else if (ts - states_(0) < 0.05) {
      NavState ns = states_[0];
      ns.quat = SmothQuat(ns.quat, quat, type);
      ns.timestamp = ts;
      states_.emplace_back(ns, ts);
      return 0;
    }
  }

  return ret;
}

int  SimpleTracker::Measurement(double ts, const Eigen::Vector3d &pos, const Eigen::Quaterniond &quat, int type) {
  if (pos.hasNaN() || quat.matrix().hasNaN()) {
    droslog(LogLevel::WARN, "SimpleTracker::Measurement(p&q): 操 马德, 你传进来的数据有毒");
    return -1;
  }
  
  NavState org_ns;
  int idx;
  int ret = EstimateState(ts, &org_ns, &idx);
  if (ret == 0) {
    auto org_dist_odom = dist_odometer_.dist_odom(ts);

    // 计算位姿变换
    // 将后续的位姿均更新
    NavState cns = org_ns;
    cns.pos = SmothPos(org_ns.pos, pos, type);
    cns.quat = SmothQuat(org_ns.quat, quat, type);
    for (int i = idx -1; i >= 0; --i) {
      // 计算相对位姿
      auto new_ns = UpdateByVel(cns, states_[i].vel, states_[i].ang_vel, states_[i].timestamp);
      if ((new_ns.timestamp - states_[i].timestamp) > 1e-6) {
        droslog(LogLevel::WARN, "SimpleTracker::Measurement(): 修正出错, 时间戳不对, new_ts=%.3f, ts=%.3f", new_ns.timestamp, states_[i].timestamp);
        return -1;
      }

      states_[i] = new_ns;
      cns = states_[i];

      // 更新其他状态
      auto dist_odom = dist_odometer_.dist_odom(states_(i));
      if (type == 1) {  // rtk
        states_[i].off_rtk_dist = dist_odom.dist - org_dist_odom.dist;
        states_[i].off_reloc_dist = states_[i].off_rtk_dist;
        states_[i].only_iw_dist = states_[i].off_rtk_dist;
      } else if (type == 4 || type == 5) { // vio/lio reloc
        states_[i].off_reloc_dist = dist_odom.dist - org_dist_odom.dist;
        states_[i].only_iw_dist = states_[i].off_reloc_dist;
      } else if (type == 2 || type == 3) { // vio/lio
        states_[i].only_iw_dist = dist_odom.dist - org_dist_odom.dist;
      }
    }
  } else if (ret == 1) {
    if (ts - states_(0) < 0.002) {
      states_[0].pos = SmothPos(states_[0].pos, pos, type);
      states_[0].quat = SmothQuat(states_[0].quat, quat, type);
      states_[0].only_iw_dist = 0;
      if (type == 1) {
        states_[0].off_rtk_dist = 0;
        states_[0].off_reloc_dist = 0;
      } else if(type == 4 || type == 5) {
        states_[0].off_reloc_dist = 0;
      }
      return 0;
    } else if (ts - states_(0) < 0.05) {
      NavState ns = states_[0];
      ns.pos = SmothPos(ns.pos, pos, type);
      ns.quat = SmothQuat(ns.quat, quat, type);
      ns.timestamp = ts;
      states_[0].only_iw_dist = 0;
      if (type == 1) {
        states_[0].off_rtk_dist = 0;
        states_[0].off_reloc_dist = 0;
      } else if(type == 4 || type == 5) {
        states_[0].off_reloc_dist = 0;
      }
      states_.emplace_back(ns, ts);
      return 0;
    }
  }

  return ret;
}

int  SimpleTracker::FindTs(double ts) const {
  int idx = states_.findAfter(ts);
  if (0 == idx) {
    return 1;
  } else if (idx > 0) {
    return 0;
  } else {
    return 2;
  }
  return -1;
}

int  SimpleTracker::EstimateState(double ts, NavState *state, int *ind) const {
  int idx = states_.findAfter(ts);
  if (idx == 0) {           // too new
    if (ind) *ind = idx;
    auto res = UpdateByVel(states_[idx], states_[idx].vel, states_[idx].ang_vel, ts);
    if (res.timestamp > 0 && state) {
      *state = res;
      return 1;
    }
  } else if (idx > 0) {     // in between
    if (ind) *ind = idx;

    if (state) {
      NavState new_ns = states_[idx-1];
      NavState old_ns = states_[idx];
      
      double rr = double(ts - old_ns.timestamp) / (new_ns.timestamp - old_ns.timestamp);
      // 位置插值
      state->pos = old_ns.pos + rr * (new_ns.pos - old_ns.pos);
      // 姿态插值
      state->quat = old_ns.quat.slerp(rr, new_ns.quat);
      state->quat.normalize();      
      // 速度插值
      state->vel = old_ns.vel + rr * (new_ns.vel - old_ns.vel);
      state->ang_vel = old_ns.ang_vel + rr * (new_ns.ang_vel - old_ns.ang_vel);

      state->off_rtk_dist = old_ns.off_rtk_dist + rr * (new_ns.off_rtk_dist - old_ns.off_rtk_dist);
      state->off_reloc_dist = old_ns.off_reloc_dist + rr * (new_ns.off_reloc_dist - old_ns.off_reloc_dist);
      state->only_iw_dist = old_ns.only_iw_dist + rr * (new_ns.only_iw_dist - old_ns.only_iw_dist);

      state->timestamp = ts;
    }
    return 0;
  } else {            // too old
    return 2;
  }
  return -1;          // error (UNEXPECTED)
}

// @param vel: 载体坐标系, 需要转换到导航坐标系
NavState SimpleTracker::UpdateByVel(const NavState &pre_state, 
    const Eigen::Vector3d &vel, const Eigen::Vector3d &ang_vel, double new_ts) const {
  NavState ret;
  double dt = new_ts - pre_state.timestamp;
  if (dt < 1e-3) {
    ret = pre_state;
    return ret;
  }

  if (dt < 0.0) {
    droslog(LogLevel::ERROR, "SimpleTracker::UpdateByVel() imu时间异常, cur_ts: %.3f, pre_ts: %.3f", new_ts, pre_state.timestamp);
    return ret;
  }
  if (dt > 0.2) {
    droslog(LogLevel::WARN, "SimpleTracker::UpdateByVel() imu时间间隔过大, cur_ts: %.3f, pre_ts: %.3f", new_ts, pre_state.timestamp);
  }

  // 更新速度
  ret.vel = vel;
  ret.ang_vel = ang_vel;

  // 更新姿态
  Eigen::Vector3d delta_angle_axis = 0.5 * (pre_state.ang_vel + ang_vel) * dt;
  if (delta_angle_axis.norm() > 1e-12) {
    ret.quat = pre_state.quat * Eigen::AngleAxisd(delta_angle_axis.norm(), delta_angle_axis.normalized());
    ret.quat.normalize();
  } else {
    ret.quat = pre_state.quat;
  }

  // 更新位置
  ret.pos = pre_state.pos + 0.5 * (pre_state.quat * pre_state.vel + ret.quat * ret.vel) * dt;

  // 里程状态
  double dist = (ret.pos - pre_state.pos).norm();
  ret.off_rtk_dist = pre_state.off_rtk_dist + dist;
  ret.off_reloc_dist = pre_state.off_reloc_dist + dist;
  ret.only_iw_dist = pre_state.only_iw_dist + dist;

  ret.timestamp = new_ts;
  return ret;
}
