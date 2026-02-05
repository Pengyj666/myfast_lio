#ifndef COMMON_VIO_CHECKER_H
#define COMMON_VIO_CHECKER_H

#include <atomic>
#include <mutex>
#include "droslog/log.h"
#include "common/common_def.h"
#include "common/data_type.h"
#include "common/sysutils.h"
#include "common/timed_queue.h"

namespace utils {

// 一个简易的VIO结果检查器, 主要检测VIO结果跳变问题
class VioChecker {
 public:
  static VioChecker* Instance() {
    static VioChecker ins;
    return &ins;
  }
  ~VioChecker() {}

  // return 0: 平稳, 1: 小跳变, 2: 大跳变
  int Check(const common::Data_VioResult &vio) {
    int ret = 0;
    // 初步以非常简单的相邻帧位置差值进行检查, 机器以0.5m/s速度行走, 0.2s内位置差值超过0.4m则认为异常, 或者 1.0s 内所有相邻帧累计位置差值超过2m则认为异常
    double max_dts = 0.0, max_dist = 0.0;
    if (vio_q_.size() > 12) {
      for (int i = 1; i < vio_q_.size(); i+=1) {
        double dts = vio.timestamp - vio_q_(i);
        double dist_th1 = 0.5 * dts * 2;
        double dist_th2 = 0.5 * dts * 3;
        if (dts > 0.2) {
          if (dts < 0.4) {
            dist_th2 = 0.5 * dts * 4;   // > 0.4m ~ 0.8m
          }
          
          double dist = (vio.vio.pos - vio_q_[i].vio.pos).norm();
          if (dist > dist_th1) {
            ret = 1;
          }
          if (dist > dist_th2) {
            ret = 2;
          }
          if (dist > max_dist) {
            max_dist = dist;
            max_dts = dts;
          }
        }
      }
    }
    if (ret > 0) {
      static long long pre_ts = 0;
      if (GetNow_Steady() > pre_ts + 500) {
        droslog(LogLevel::WARN, "VioChecker::Check(), VIO结果跳变异常, 跳变类型: %d, cur_vio.ts: %.3f, 最大异常差:dts=%.3f, dist=%.3f", ret, vio.timestamp, max_dts, max_dist);
        pre_ts = GetNow_Steady();
      }
    }
    vio_q_.emplace_back(vio, vio.timestamp);
    return ret;
  }

 private:
  VioChecker() {
    vio_q_.reset(16);
  }
  VioChecker(const VioChecker&) = delete;
  VioChecker& operator=(const VioChecker&) = delete;

  std::mutex vio_q_mutex_;
  TimedQueue<common::Data_VioResult> vio_q_;
};

}  // namespace utils

#endif  // COMMON_VIO_CHECKER_H
