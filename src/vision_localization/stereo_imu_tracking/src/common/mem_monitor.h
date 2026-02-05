#ifndef UTILS_MEM_MONITOR_H
#define UTILS_MEM_MONITOR_H

#include <mutex>
#include <map>
#include "droslog/log.h"
#include "common/sysutils.h"

namespace utils {

class MemMonitor {
 public:
  static MemMonitor* Instance() {
    static MemMonitor ins;
    return &ins;
  }
  ~MemMonitor() {}

  // size: Bytes
  // unit: 0: MB, 1: Bytes, 2: KB, 3: GB
  double CSize(int size, int unit = 0) {
    static double KB = 1024.0;
    static double MB = 1024.0 * 1024.0;
    static double GB = 1024.0 * 1024.0 * 1024.0;

    double ret = size / MB;
    if (unit == 1) {
      ret = size;
    } else if (unit == 2) {
      ret = size / KB;
    } else if (unit == 3) {
      ret = size / GB;
    }
    return ret;
  }

  // size: Bytes
  void NewMem(const std::string &user, int size) {
    std::lock_guard<std::mutex> lock(mutex_);
    occ_mem_ += size;
    if (mem_.find(user) == mem_.end()) {
      mem_[user] = 0;
    }
    mem_[user] += size;
  }
  // size: Bytes
  void DelMem(const std::string &user, int size) {
    std::lock_guard<std::mutex> lock(mutex_);
    occ_mem_ -= size;
    if (mem_.find(user) == mem_.end()) {
      droslog(LogLevel::ERROR, "MemMonitor::DelMem(): user %s not found", user.c_str());
      return;
    }
    mem_[user] -= size;
  }

  // unit: 0: MB, 1: Bytes, 2: KB, 3: GB
  double GetOccMem(int unit = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    return CSize(occ_mem_, unit);
  }

  void PrintMem() {
    std::lock_guard<std::mutex> lock(mutex_);
    droslog(LogLevel::INFO, "MemMonitor::PrintMem(): 申请内存总额: %.3f MB", CSize(occ_mem_));
    for (auto &it : mem_) {
      droslog(LogLevel::INFO, "MemMonitor::PrintMem(): 子模块申请内存总额%s: %.3f KB", it.first.c_str(), CSize(it.second, 2));
    }
  }

 private:
  MemMonitor() : occ_mem_(0) {}
  MemMonitor(const MemMonitor&) = delete;
  MemMonitor& operator=(const MemMonitor&) = delete;

  std::mutex mutex_;
  int occ_mem_; // Bytes, 占用总数
  std::map<std::string, int> mem_;  // Bytes, 各子模块占用
};

}  // namespace utils

#endif  // UTILS_MEM_MONITOR_H
