#ifndef COMMON_GNSS_MONITOR_H
#define COMMON_GNSS_MONITOR_H

#include <deque>
#include <atomic>
#include "common/data_type.h"
#include "common/common_def.h"
#include "common/log_filters.h"
#include "common/sysutils.h"
#include "common/timed_queue.h"
#include "droslog/log.h"

namespace utils {

// 一个简易的Gnss数据状态监控器，只用于跟踪RTK固定解情况
class GnssMonitor {
 public:
  static GnssMonitor* Instance() {
    static GnssMonitor ins;
    return &ins;
  }
  ~GnssMonitor() {}

  void Update(const common::Data_Gnss& gnss) {
    long long cur_ll_ts = GetNow_Steady();
    {
      static long long pre_ts = 0;
      if (cur_ll_ts > pre_ts + 5000) {
        droslog(LogLevel::INFO, "GnssMonitor::Update(), is_valid=%d, ts=%.3f, type: %s, last_fix_ts=%.3f, last_unfix_ts=%.3f", 
            is_valid_.load(), gnss.timestamp, gnss.gnss.rtk_type.c_str(), last_fix_ts_, last_unfix_ts_);
        pre_ts = cur_ll_ts;
      }
    }

    if (gnss.timestamp <= 10.0) {
      droslog(LogLevel::WARN, "GnssMonitor::Update(), 异常gnsss数据, 时间戳小于10s, ts=%.3f", gnss.timestamp);
      return;
    }
    last_data_ts_.store(cur_ll_ts);

    if (gnss.timestamp <= pre_gnss_.timestamp) {
      droslog(LogLevel::WARN, "GnssMonitor::Update(), 异常gnss数据, 时间戳不递增, new_ts=%.3f, but last_ts=%.3f", 
          gnss.timestamp, pre_gnss_.timestamp);
      return;
    }
    if (gnss.timestamp > pre_gnss_.timestamp + 0.3) {
      droslog(LogLevel::WARN, "GnssMonitor::Update(), 异常gnss数据, 时间戳跳跃大于0.3s, 丢帧严重, new_ts=%.3f, but last_ts=%.3f", 
          gnss.timestamp, pre_gnss_.timestamp);
    }
    if (gnss.gnss.rtk_type != pre_gnss_.gnss.rtk_type) {
      droslog(LogLevel::WARN, "GnssMonitor::Update(), gnss数据状态变化, [%s] -> [%s], new_ts=%.3f, last_ts=%.3f, new_enu=(%.6f, %.6f, %.6f), last_enu=(%.6f, %.6f, %.6f)", 
          pre_gnss_.gnss.rtk_type.c_str(), gnss.gnss.rtk_type.c_str(), gnss.timestamp, pre_gnss_.timestamp,
          gnss.gnss.enu[0], gnss.gnss.enu[1], gnss.gnss.enu[2],
          pre_gnss_.gnss.enu[0], pre_gnss_.gnss.enu[1], pre_gnss_.gnss.enu[2]);
    }

    if (is_valid_.load()) {
      // 当前为可靠状态

      // 新的数据为非固定解, 检查持续时间
      if (gnss.gnss.rtk_type != common::RTK_NARROW_INT && gnss.timestamp - last_fix_ts_ > dtime2_) {
        is_valid_.store(false);
        droslog(LogLevel::WARN, "GnssMonitor::Update(), RTK丢失固定解超过%.3f秒, 位置从可靠->不可靠, new_ts=%.3f, rtk_type=%s", dtime2_, gnss.timestamp, gnss.gnss.rtk_type.c_str());
      }
    } else {
      // 当前为不可靠状态

      // 新的数据为固定解, 检查持续时间
      if (gnss.gnss.rtk_type == common::RTK_NARROW_INT && gnss.timestamp - last_unfix_ts_ > dtime1_) {
        is_valid_.store(true);
        droslog(LogLevel::INFO, "GnssMonitor::Update(), RTK固定解持续大于%.3f秒, 位置从不可靠->可靠, new_ts=%.3f, rtk_type=%s", dtime1_, gnss.timestamp, gnss.gnss.rtk_type.c_str());
      }

      if (gnss.gnss.rtk_type != common::RTK_NARROW_INT) {
        static double pre_log_ts = 0.0;
        if (gnss.timestamp - pre_log_ts > 10.0) {
          pre_log_ts = gnss.timestamp;
          droslog(LogLevel::WARN, "GnssMonitor::Update(), RTK丢失固定解已持续 %.3f秒, new_ts=%.3f", gnss.timestamp - last_fix_ts_, gnss.timestamp);
        }
      }
    }
    pre_gnss_ = gnss;

    if (gnss.gnss.rtk_type == common::RTK_NARROW_INT) {
      last_fix_ts_ = gnss.timestamp;
      last_is_fixed_.store(true);
    } else {
      last_unfix_ts_ = gnss.timestamp;
      last_is_fixed_.store(false);
    }
  }

  bool IsGnssValid(int dts = 5000) {
    bool valid = is_valid_.load() && GetNow_Steady() < last_data_ts_.load() + dts;
    if (GetNow_Steady() >= last_data_ts_.load() + dts) {
      static long long log_ts = 0;
      if (GetNow_Steady() > log_ts + 5000) {
        log_ts = GetNow_Steady();
        droslog(LogLevel::WARN, "GnssMonitor::IsGnssValid(), GNSS数据已丢失 %lld ms, valid=%d", GetNow_Steady() - last_data_ts_.load(), valid);
      } 
    }
    return valid;
  }

  bool IsLastGnssFixed() {
    return last_is_fixed_.load() && GetNow_Steady() < last_data_ts_.load() + 1500;
  }

  // return 0: 未发生基站参考更新, 1: 发生基站参考更新
  int UpdateRtkRef(const double &ts, const double &lat, const double &lon, const double &alt, const bool &is_valid = true) {
    last_rtk_ref_ts_ = ts;
    last_rtk_ref_is_valid_ = is_valid;

    static SimpleLogFilter log_filter(2000);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::INFO, "GnssMonitor::UpdateRtkRef(), rtk_ref: is_valid=%d, ts=%.3f, lla=%.8f,%.8f,%.3f", is_valid, ts, lat, lon, alt);
    }

    if (!is_valid) {
      last_rtk_ref_invalid_ts_ = ts;
      return 0;
    }

    // 与当前使用参考坐标一致, 不做处理, 新坐标检验状态置false
    if (rtk_ref_lla_[0] == lat && rtk_ref_lla_[1] == lon && rtk_ref_lla_[2] == alt) {
      if (is_checking_new_ref_rtk_) {
        droslog(LogLevel::WARN, "GnssMonitor::UpdateRtkRef(), RTK基站参考坐标恢复至当前参考坐标, 应该是出现了短暂数据跳变");
      }
      is_checking_new_ref_rtk_ = false;
      return 0;
    }

    // 是否第一个新参考坐标
    if (!is_checking_new_ref_rtk_) {
      droslog(LogLevel::WARN, "GnssMonitor::UpdateRtkRef(), 收到第一个新RTK基站参考坐标 rtk_ref: ts=%.3f, lla=%.8f,%.8f,%.3f, 当前参考坐标: ts=%.3f, lla=%.8f,%.8f,%.3f", 
          ts, lat, lon, alt, rtk_ref_ts_, rtk_ref_lla_[0], rtk_ref_lla_[1], rtk_ref_lla_[2]);
      is_checking_new_ref_rtk_ = true;
      new_ref_rtk_lla_ << lat, lon, alt;
      new_ref_rtk_ts_ = ts;
    } else {
      // 新参考坐标与当前检验新参考坐标不一致, 更新当前检验新参考坐标
      if (new_ref_rtk_lla_[0] != lat || new_ref_rtk_lla_[1] != lon || new_ref_rtk_lla_[2] != alt) {
        droslog(LogLevel::WARN, "GnssMonitor::UpdateRtkRef(), 新RTK基站参考坐标二次变化 rtk_ref: ts=%.3f, lla=%.8f,%.8f,%.3f, 当前参考坐标: ts=%.3f, lla=%.8f,%.8f,%.3f", 
            ts, lat, lon, alt, rtk_ref_ts_, rtk_ref_lla_[0], rtk_ref_lla_[1], rtk_ref_lla_[2]);
        is_checking_new_ref_rtk_ = true;
        new_ref_rtk_lla_ << lat, lon, alt;
        new_ref_rtk_ts_ = ts;
      } else {
        droslog(LogLevel::WARN, "GnssMonitor::UpdateRtkRef(), 新RTK基站参考坐标持续中(%.1f秒), rtk_ref: ts=%.3f, lla=%.8f,%.8f,%.3f, 当前参考坐标: ts=%.3f, lla=%.8f,%.8f,%.3f", 
            ts - new_ref_rtk_ts_, ts, lat, lon, alt, rtk_ref_ts_, rtk_ref_lla_[0], rtk_ref_lla_[1], rtk_ref_lla_[2]);
        // 新参考坐标与当前检验新参考坐标一致, 检验持续时长，超过4秒则更新参考坐标
        if (ts - new_ref_rtk_ts_ > 4.0) {
          droslog(LogLevel::WARN, "GnssMonitor::UpdateRtkRef(), 新RTK基站参考坐标持续超过4秒, 更新RTK基站参考坐标及cur_ref_geo转换器, rtk_ref: ts=%.3f, lla=%.8f,%.8f,%.3f, 原参考坐标: ts=%.3f, lla=%.8f,%.8f,%.3f", 
              ts, lat, lon, alt, rtk_ref_ts_, rtk_ref_lla_[0], rtk_ref_lla_[1], rtk_ref_lla_[2]);
          rtk_ref_ts_ = ts;
          rtk_ref_lla_ << lat, lon, alt;
          return 1;
        }
      }
    }

    return 0;
  }

  // RTK基站参考有效: 上一个rtk_ref有效, 当前时间距离上一个rtk_ref有效时间较短, 当前时间距离上一个rtk_ref无效时间较长
  bool CheckRtkRef(const double &ts, const double &max_dts = 20.0) {
    if (last_rtk_ref_is_valid_ && ts - last_rtk_ref_ts_ < max_dts && ts - last_rtk_ref_invalid_ts_ > 3.0 && rtk_ref_ts_ > 0.0) {
      return true;
    }
    return false;
  }

 private:
  GnssMonitor(double dtime1 = 1.0, double dtime2 = 1.5) : dtime1_(dtime1), dtime2_(dtime2) {
    is_valid_.store(false);
    last_is_fixed_.store(false);
    last_data_ts_.store(0);
    droslog(LogLevel::INFO, "GnssMonitor::GnssMonitor(), dtime1=%.2f, dtime2=%.2f, last_fix_ts=%.3f, last_unfix_ts=%.3f", 
        dtime1_, dtime2_, last_fix_ts_, last_unfix_ts_);
    last_fix_ts_ = 0.0;
    last_unfix_ts_ = 0.0;
  }
  GnssMonitor(const GnssMonitor&) = delete;
  GnssMonitor& operator=(const GnssMonitor&) = delete;

  double dtime1_;   // RTK固定解可用时间阈值，位置不可靠状态下连续收到RTK固定解超过此时间，则认为位置进入可靠状态
  double dtime2_;   // RTK固定解不可用时间阈值，超过此时间还未收到RTK固定解，则认为位置进入不可靠状态

  std::atomic_bool is_valid_;
  std::atomic_bool last_is_fixed_;
  std::atomic<long long> last_data_ts_;

  double last_fix_ts_;
  double last_unfix_ts_;
  common::Data_Gnss pre_gnss_;

  double rtk_ref_ts_ = 0.0;
  Eigen::Vector3d rtk_ref_lla_ = Eigen::Vector3d::Zero();

  bool is_checking_new_ref_rtk_ = false;
  Eigen::Vector3d new_ref_rtk_lla_ = Eigen::Vector3d::Zero();
  double new_ref_rtk_ts_ = 0.0;

  double last_rtk_ref_ts_ = 0.0;
  bool last_rtk_ref_is_valid_ = false;
  double last_rtk_ref_invalid_ts_ = 0.0;
};

}  // namespace utils

#endif  // COMMON_GNSS_MONITOR_H
