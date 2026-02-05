#ifndef COMMON_VMAP_MONITOR_H
#define COMMON_VMAP_MONITOR_H

#include <atomic>
#include "droslog/log.h"
#include "common/sysutils.h"

namespace utils {

// vmap状态监控器
class VmapMonitor {
 public:
  static VmapMonitor* Instance() {
    static VmapMonitor ins;
    return &ins;
  }
  ~VmapMonitor() {}

  // 0: idle, 1: mapping, 2: localization
  void FeedVmapState(long long ts, int state) {
    if (last_state_.load() != state) {
      droslog(LogLevel::INFO, "VmapMonitor::FeedVmapState(): Vmap状态变化 %d -> %d", last_state_.load(), state);
    }
    last_state_.store(state);

    if (0 == state) {
      last_idle_ts_.store(ts);
    } else if (2 == state) {
      last_loc_ts_.store(ts);
    }
  }

  void SetLoadMapTs(long long ts) {
    droslog(LogLevel::INFO, "VmapMonitor::SetLoadMapTs(): 刷新Vmap加载地图时间戳: %lld", ts);
    last_load_map_ts_.store(ts);
  }

  bool NeedLoadMap() {
    auto cur_ts = GetNow_Steady();
    bool ret = cur_ts > last_load_map_ts_.load() + 10000 && cur_ts > last_loc_ts_.load() + 10000 && cur_ts < last_idle_ts_.load() + 3000;
    if (ret) {
      droslog(LogLevel::INFO, "VmapMonitor::NeedLoadMap() 检测到Vmap需要加载地图: cur_ts=%lld, last_load_map_ts_=%lld, last_loc_ts_=%lld, last_idle_ts_=%lld",
          cur_ts, last_load_map_ts_.load(), last_loc_ts_.load(), last_idle_ts_.load());
    }

    return ret;
  }

 private:
  VmapMonitor() {
    last_state_.store(-1);
    last_load_map_ts_.store(0);
    last_idle_ts_.store(0);
    last_loc_ts_.store(0);
  }
  VmapMonitor(const VmapMonitor&) = delete;
  VmapMonitor& operator=(const VmapMonitor&) = delete;

  std::atomic<int> last_state_;
  std::atomic<long long> last_load_map_ts_;
  std::atomic<long long> last_idle_ts_;
  std::atomic<long long> last_loc_ts_;
};

}  // namespace utils

#endif  // COMMON_VMAP_MONITOR_H
