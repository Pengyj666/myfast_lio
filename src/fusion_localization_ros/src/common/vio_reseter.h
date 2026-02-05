#ifndef COMMON_VIO_RESETER_H
#define COMMON_VIO_RESETER_H

#include <atomic>
#include "droslog/log.h"
#include "common/common_def.h"
#include "common/data_type.h"
#include "common/sysutils.h"
#include "common/timed_queue.h"

namespace utils {

// 一个简易的VIO状态检测器, 用于检测是否需要重置VIO
// 这里的检测基于能正常收到vio数据, 无vio数据输出或者帧率极低的情况由上层逻辑对节点进行重启
class VioReseter {
 public:
  static VioReseter* Instance() {
    static VioReseter ins;
    return &ins;
  }
  ~VioReseter() {}

  void FeedData(const common::Data_VioResult &vio) {
    // 简单实现, 连续20帧出现vio剧烈跳动
    if (vio.timestamp - last_vio_.timestamp > 5.0 && last_vio_.timestamp > 0.0) {
      droslog(LogLevel::WARN, "VioReseter::FeedData() vio数据长时间未更新, cur_ts=%.3f, last_ts=%.3f", vio.timestamp, last_vio_.timestamp);
      vio_good_cnt_ = 0;
      vio_bad_cnt_ = 0;
    } 
    if (vio.timestamp - last_vio_.timestamp < 0.5) {
      double dist = (vio.vio.pos - last_vio_.vio.pos).norm();
      if (dist > 3.0) {
        if (vio_bad_cnt_++ > 15) {
          if (!is_vio_need_reset_.load()) {
            droslog(LogLevel::WARN, "VioReseter::FeedData() vio持续剧烈跳动, 检测到vio需要重置");
          }
          is_vio_need_reset_.store(true);
          vio_good_cnt_ = 0;
          vio_bad_cnt_ = 0;
        }
      } else if (dist < 0.5) {
        if (vio_good_cnt_++ > 10) {
          if (is_vio_need_reset_.load()) {
            droslog(LogLevel::WARN, "VioReseter::FeedData() vio恢复平稳");
          }
          is_vio_need_reset_.store(false);
          vio_bad_cnt_ = 0;
        }
      }
    }
    last_vio_ = vio;
  }

  bool IsVioNeedReset() {
    return is_vio_need_reset_.load() && GetNow_Steady() - last_reset_time_ > 10.0;
  }

  // 上层调用完重置vio后调用
  void UpdateResetTime() {
    last_reset_time_.store(GetNow_Steady());
    last_vio_ = common::Data_VioResult();
    is_vio_need_reset_.store(false);
    vio_good_cnt_ = 0;
    vio_bad_cnt_ = 0;
    droslog(LogLevel::WARN, "VioReseter::UpdateResetTime() time: %lld", last_reset_time_.load());
  }

 private:
  VioReseter() {
    is_vio_need_reset_.store(false);
    last_reset_time_.store(GetNow_Steady());
    last_failed_vio_time_.store(0.0);
    last_valid_vio_time_.store(0.0);
  }
  VioReseter(const VioReseter&) = delete;
  VioReseter& operator=(const VioReseter&) = delete;

  std::atomic_bool is_vio_need_reset_;
  std::atomic<long long> last_reset_time_;

  std::atomic<double> last_failed_vio_time_;
  std::atomic<double> last_valid_vio_time_;

  common::Data_VioResult last_vio_;
  int vio_good_cnt_ = 0;
  int vio_bad_cnt_ = 0;
};

}  // namespace utils

#endif  // COMMON_VIO_RESETER_H
