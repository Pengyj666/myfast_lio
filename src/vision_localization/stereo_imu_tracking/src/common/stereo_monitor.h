#ifndef VIO_COMMON_STEREO_MONITOR_H
#define VIO_COMMON_STEREO_MONITOR_H

#include <map>

class StereoMonitor {
 public:
  static StereoMonitor* Instance() {
    static StereoMonitor ins;
    return &ins;
  }
  ~StereoMonitor() {}

  // type: 0-/vio/imu, 1-/vio/left/image_raw, 2-/vio/right/image_raw
  void count_update(double ts, int type) {
    if (type >= 0 && type <= 3) {
      topic_counter_[type]++;
    }
  }
  
  // type: 0-/vio/imu, 1-/vio/left/image_raw, 2-/vio/right/image_raw
  // return < 0: invalid
  int get_count(int type) {
    if (type >= 0 && type <= 3) {
      return topic_counter_[type];
    }
    return -1;
  }

  std::map<int, int> get_all_count() {
    return topic_counter_;
  }

 private:
  StereoMonitor() {
    topic_counter_[0] = 0;
    topic_counter_[1] = 0;
    topic_counter_[2] = 0;
  }
  StereoMonitor(const StereoMonitor&) = delete;
  StereoMonitor& operator=(const StereoMonitor&) = delete;

  std::map<int, int> topic_counter_;
};

#endif//VIO_COMMON_STEREO_MONITOR_H