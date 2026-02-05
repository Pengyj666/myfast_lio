#ifndef COMMON_SENSOR_MONITOR_H
#define COMMON_SENSOR_MONITOR_H

#include <mutex>
#include "common/data_type.h"
#include "common/timed_queue.h"

namespace utils {

// 一个简易的传感器状态监控器
class SensorMonitor {
 public:
  struct Config {
    // double on_charging_station_fts = 3.0;  // 在桩状态的过滤时间窗(ts0 ~ ts0+fts 均在桩, 方可认为ts0时刻在桩), 秒
  };

 public:
  static SensorMonitor* Instance() {
    static SensorMonitor ins;
    return &ins;
  }
  ~SensorMonitor() {}

  void SetConfig(const Config &config) {
    config_ = config;
  }

  void FeedData(const std::shared_ptr<common::DataBase> &sp);

  // 返回时间戳为ts的机器在桩状态, ts: 秒
  // return: -1: 未知, 0: 非在桩, 1: 在桩, 2: 在桩充电中
  int GetChargingStationState(double ts, double dts = 2.0);

 private:
  SensorMonitor();
  SensorMonitor(const SensorMonitor&) = delete;
  SensorMonitor& operator=(const SensorMonitor&) = delete;

  Config config_;
  
  std::mutex charging_station_state_q_mutex_;
  TimedQueue<int> charging_station_state_q_;    // -1: 未知, 0: 非在桩, 1: 在桩, 2: 在桩充电中
  TimedQueue<int> moving_state_q_;              // -1: 未知, 0: 静止, 1: 运动
};

}  // namespace utils

#endif  // COMMON_SENSOR_MONITOR_H
