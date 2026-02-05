#ifndef COMMON_OFFSET_TIMER_H
#define COMMON_OFFSET_TIMER_H

#include <atomic>
#include <mutex>
#include <thread>
#include "common/sysutils.h"
#include "common/timed_queue.h"

namespace utils {

// 一个简易时钟偏移估计器
// 每秒估计一次
class OffsetTimer {
 public:
  OffsetTimer(const std::string &info = "holdplace");
  ~OffsetTimer();

  void Hello();

  void FeedEmb_ts(double sys_ts, double emb_ts);
  // return: sys_ts - emb_ts, < 0.0 means invalid
  double GetEmb_dt() { return emb_dt_.load(); }

 private:
  void Run();
  std::atomic_bool to_stop_;
  std::thread run_thread_;

  std::atomic<double> emb_dt_;  // sec

  std::string info_;

  std::mutex emb_dts_mutex_;
  TimedQueue<double> emb_dts_q_;
};

}  // namespace utils

#endif  // COMMON_OFFSET_TIMER_H
