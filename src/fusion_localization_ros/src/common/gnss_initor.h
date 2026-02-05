#ifndef COMMON_GNSS_INITOR_H
#define COMMON_GNSS_INITOR_H

#include <atomic>
#include <mutex>
#include "common/data_type.h"

class GnssInitor {
 public:
  static GnssInitor* Instance() {
    static GnssInitor ins;
    return &ins;
  }
  ~GnssInitor() {}

  void Reset();
  void FeedData(const common::Data_Gnss &gnss);
  void StartInit();

  // 0: init ok, 1: data not enough, 2: moving too short
  int FinishInit();

  bool IsInitFinished() const { return stage_.load() >= 2; }

  double GetInitHeading() const { return init_heading_; }
  double GetInitDist() const { return init_dist_; }
  common::Data_Gnss GetStartGnss();
  common::Data_Gnss GetEndGnss();

 private:
  GnssInitor() : stage_(0), init_heading_(0.0) {}
  GnssInitor(const GnssInitor&) = delete;
  GnssInitor& operator=(const GnssInitor&) = delete;

  std::atomic<int> stage_;    // 0: idle, 1: initing, 2: init finished, 3: init failed
  double init_heading_;
  double init_dist_;

  std::mutex mutex_;
  std::vector<common::Data_Gnss> gnss_vec_;
};
#endif //COMMON_GNSS_INITOR_H