#include "common/offset_timer.h"

#include <vector>

#include "common/log_filters.h"
#include "common/sysutils.h"
#include "droslog/log.h"

using namespace utils;

OffsetTimer::OffsetTimer(const std::string &info) : to_stop_(true), info_(info) {
  droslog(LogLevel::INFO, "OffsetTimer::ctor(%s) ++++++", info_.c_str());
  
  emb_dt_.store(-1e6);

  emb_dts_q_.reset(128);
  run_thread_ = std::thread(&OffsetTimer::Run, this);

  droslog(LogLevel::INFO, "OffsetTimer::ctor(%s) ------", info_.c_str());
}

OffsetTimer::~OffsetTimer() {
  droslog(LogLevel::INFO, "OffsetTimer::dtor(%s) ++++++", info_.c_str());
  to_stop_.store(true);
  if (run_thread_.joinable()) {
    droslog(LogLevel::INFO, "OffsetTimer::dtor(%s) wait run-thread to stop", info_.c_str());
    run_thread_.join();
    droslog(LogLevel::INFO, "OffsetTimer::dtor(%s) run-thread stopped", info_.c_str());
  }
  droslog(LogLevel::INFO, "OffsetTimer::dtor(%s) ------", info_.c_str());
}

void OffsetTimer::Hello() {
  droslog(LogLevel::INFO, "OffsetTimer::Hello(%s)", info_.c_str());
}

void OffsetTimer::FeedEmb_ts(double sys_ts, double emb_ts) {
  std::lock_guard<std::mutex> lock(emb_dts_mutex_);
  emb_dts_q_.emplace_back(sys_ts-emb_ts, sys_ts);
}

void OffsetTimer::Run() {
  droslog(LogLevel::INFO, "OffsetTimer::Run(%s) ++++++", info_.c_str());

  to_stop_.store(false);
  bool dts_init = false;
  while (!to_stop_.load()) {
    Sleep(100);

    long long cur_ts = GetNow_Steady();
    static SimpleLogFilter fps_filter(1000);
    if (fps_filter.Output(cur_ts)) {
      std::vector<double> vdts;
      int calc_num = (dts_init ? 15 : 100);
      double first_ts = 0.0, last_ts = 0.0;
      {
        std::lock_guard<std::mutex> lock(emb_dts_mutex_);
        int size = emb_dts_q_.size();
        for (int i = 0; i < size && i < calc_num+2; i++) {
          if (0 == i) first_ts = emb_dts_q_(i);
          last_ts = emb_dts_q_(i);
          vdts.push_back(emb_dts_q_[i]);
        }
      }

      int vsize = vdts.size();
      if (vsize > 5) {
        double mean_dts = 0.0;
        for (int i = 0; i < vsize; i++) {
          mean_dts += vdts[i];
        }
        mean_dts /= vsize;
        
        double ddts = mean_dts - emb_dt_.load();
        if (dts_init) {
          emb_dt_.store(mean_dts - ddts * 0.9);
        } else {
          emb_dt_.store(mean_dts);
          if (vsize > 100) {
            dts_init = true;
          }
        }

        static SimpleLogFilter log_filter(10000);
        if (std::abs(ddts)*0.1 > 0.01 || log_filter.Output(cur_ts)) {
          droslog(LogLevel::INFO, "OffsetTimer::Run(%s) 时间戳偏移估计: offset_ts=%.3f, 与上次偏移变动=%.3f, 样本量=%d, first_ts=%.3f, last_ts=%.3f", 
              info_.c_str(), emb_dt_.load(), ddts, vsize, first_ts, last_ts);
        }
      }
    }
  }
  droslog(LogLevel::INFO, "OffsetTimer::Run(%s) ------", info_.c_str());
}


