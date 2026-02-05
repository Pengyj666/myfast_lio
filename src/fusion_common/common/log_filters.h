#ifndef DROS_UTILS_COMMON_LOG_FILTERS_H_
#define DROS_UTILS_COMMON_LOG_FILTERS_H_

// 简单的按时间间隔过滤log
class SimpleLogFilter {
 public:
  SimpleLogFilter(long long dts = 500) : pre_log_ts_(0), dts_(dts), ck_cnt_(0) { }

  bool Output(long long now_ts) {
    ck_cnt_++;
    if (now_ts > pre_log_ts_ + dts_) {
      pre_log_ts_ = now_ts;
      return true;
    }
    return false;
  }
 
  long long pre_log_ts_;
  long long dts_;
  int ck_cnt_;
};

#endif//DROS_UTILS_COMMON_LOG_FILTERS_H_