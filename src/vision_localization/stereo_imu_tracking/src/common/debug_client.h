#ifndef UTILS_DEBUG_DEBUG_CLIENT_H
#define UTILS_DEBUG_DEBUG_CLIENT_H

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

 private:
  DebugClient() {
    rtk_state_.store(0);
    docking_state_.store(0);
    docking_start_ts_.store(0);
    rtk_ref_change_ts_.store(0);
  }
  DebugClient(const DebugClient&) = delete;
  DebugClient& operator=(const DebugClient&) = delete;

  std::atomic_int rtk_state_;
  std::atomic_int docking_state_;   // 0-unknown, 1-docking, 2-not docking
  std::atomic<long long> docking_start_ts_;

  std::atomic<long long> rtk_ref_change_ts_;
};

}  // namespace utils

#endif  // UTILS_DEBUG_DEBUG_CLIENT_H
