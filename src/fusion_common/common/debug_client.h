#ifndef UTILS_COMMON_DEBUG_CLIENT_H
#define UTILS_COMMON_DEBUG_CLIENT_H

#include <atomic>
#include "common/sysutils.h"

namespace utils {

// 一个简易的VIO结果检查器, 主要检测VIO结果跳变问题
class DebugClient {
 public:
  static DebugClient* Instance() {
    static DebugClient ins;
    return &ins;
  }
  ~DebugClient() {}

  // 0-unknown, 1-rtk-vio, 2-vio-only, 3-rtk-only, 4-lidar-only
  void SetLocSensorType(int type) { loc_sensor_type_.store(type); }
  // 0-unknown, 1-rtk-vio, 2-vio-only, 3-rtk-only, 4-lidar-only
  int GetLocSensorType() { return loc_sensor_type_.load(); }

  // type: 0-维持原状, 1-RTK强制单点解, 2-RTK强制固定解, 3-RTK不输入
  void SetRtkState(int state) { rtk_state_.store(state); }
  // return: 0-维持原状, 1-RTK强制单点解, 2-RTK强制固定解, 3-RTK不输入
  int GetRtkState() { return rtk_state_.load(); }

  void UpdateRtkRefChange(const long long &ts) {
    rtk_ref_change_ts_.store(ts);
  }

  // 1.5秒内有效
  bool IsNeedRtkRefChange(const long long &ts) {
    return ts < rtk_ref_change_ts_.load() + 2000;
  }

  void SetDockingState(int state) { 
    docking_start_ts_.store(GetNow_Steady());
    docking_state_.store(state); 
  }
  bool GetDockingState() {
    return docking_state_.load() == 1 && GetNow_Steady() - docking_start_ts_.load() < 10000;
  }

  void SetVioOnOff(bool used) { used_vio_.store(used); }
  bool GetVioOnOff() { return used_vio_.load(); }

  void SetUseWheelVel(bool use) { use_wheel_vel_.store(use); }
  bool GetUseWheelVel() { return use_wheel_vel_.load(); }

  void SetLocAlwaysValid(bool valid) { loc_always_valid_.store(valid); }
  bool GetLocAlwaysValid() { return loc_always_valid_.load(); }

 private:
  DebugClient() {
    used_vio_.store(true);
    loc_sensor_type_.store(0);
    rtk_state_.store(0);
    docking_state_.store(0);
    docking_start_ts_.store(0);
    use_wheel_vel_.store(true);
    loc_always_valid_.store(false);
    rtk_ref_change_ts_.store(0);
  }
  DebugClient(const DebugClient&) = delete;
  DebugClient& operator=(const DebugClient&) = delete;

  std::atomic_bool used_vio_;
  std::atomic_int loc_sensor_type_;  // 0-unknown, 1-rtk-vio, 2-vio-only, 3-rtk-only, 4-lidar-only
  std::atomic_int rtk_state_;
  std::atomic_int docking_state_;   // 0-unknown, 1-docking, 2-not docking
  std::atomic<long long> docking_start_ts_;

  std::atomic_bool use_wheel_vel_;
  std::atomic_bool loc_always_valid_;

  std::atomic<long long> rtk_ref_change_ts_;
};

}  // namespace utils

#endif  // UTILS_COMMON_DEBUG_CLIENT_H
