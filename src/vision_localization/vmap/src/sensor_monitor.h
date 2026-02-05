#ifndef VMAP_SENSOR_MONITOR_H
#define VMAP_SENSOR_MONITOR_H

#include <mutex>
#include "common/timed_queue.h"

namespace utils {

// 一个简易的传感器状态监控器
class SensorMonitor {  
 public:
  static SensorMonitor* Instance() {
    static SensorMonitor ins;
    return &ins;
  }
  ~SensorMonitor() {}

  void FeedCSState(double ts, const int &state) {
    if (ts > 0.0)
      return;

    std::lock_guard<std::mutex> lock(charging_station_state_q_mutex_);
    charging_station_state_q_.emplace_back(state, ts);
  }
  void FeedMovingState(double ts, const int &state) {
    if (ts > 0.0)
      return;

    std::lock_guard<std::mutex> lock(charging_station_state_q_mutex_);
    moving_state_q_.emplace_back(state, ts);
  }

  // 返回时间戳为ts的机器在桩状态, ts: 秒
  // return: -1: 未知, 0: 非在桩, 1: 在桩, 2: 在桩充电中
  int GetChargingStationState(double ts, double dts = 0.5) {
    std::lock_guard<std::mutex> lock(charging_station_state_q_mutex_);
    int cs_state = -1;
    {
      int idx = charging_station_state_q_.findAfter(ts);
      if (idx > 0) {
        cs_state = charging_station_state_q_[idx];
      }
      if (0 == idx) {
        if (charging_station_state_q_(0) + dts > ts) {
          cs_state = charging_station_state_q_[0];
        }
      }
    }

    int moving_state = -1;
    {
      int idx = moving_state_q_.findAfter(ts);
      if (idx > 0) {
        moving_state = moving_state_q_[idx];
      }
      for (int i = idx; i >= 0; i--) {
        if (moving_state_q_[i] == 1) {
          moving_state = 1;
        }
      }
      if (0 == idx) {
        if (moving_state_q_(0) + dts > ts) {
          moving_state = moving_state_q_[0];
        }
      }
    }

    if (cs_state >= 1) {
      if (moving_state == 1) {
        return 0;
      } else {
        return 1;
      }
    } else if (cs_state == 0) {
      return 0;
    }

    return -1;
  }

 private:
  SensorMonitor() {
    charging_station_state_q_.reset(128);   // 2hz, about 1min
    moving_state_q_.reset(512);
  }
  SensorMonitor(const SensorMonitor&) = delete;
  SensorMonitor& operator=(const SensorMonitor&) = delete;

  std::mutex charging_station_state_q_mutex_;
  TimedQueue<int> charging_station_state_q_;    // -1: 未知, 0: 非在桩, 1: 在桩, 2: 在桩充电中
  TimedQueue<int> moving_state_q_;              // -1: 未知, 0: 非移动, 1: 移动
};

}  // namespace utils

#endif  // VMAP_SENSOR_MONITOR_H
