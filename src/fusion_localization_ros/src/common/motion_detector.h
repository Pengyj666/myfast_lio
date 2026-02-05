#ifndef COMMON_MOTION_DETECTOR_H
#define COMMON_MOTION_DETECTOR_H

#include <atomic>
#include <mutex>
#include <thread>
#include "common/data_type.h"
#include "common/timed_queue.h"

namespace utils {

// 运动状态检测器
// 正常运动、静止、原地打滑、移动打滑
class MotionDetector {
 public:
  static MotionDetector* Instance() {
    static MotionDetector ins;
    return &ins;
  }
  ~MotionDetector() {}

  // gnss, 固定解, 已转换到local_xyz
  void Update(const common::Data_Pose& gnss_xyz) {

  }

  // vio, 已转换到base
  void Update(const common::Data_VioResult& vio) {

  }

  // return 0:正常运动，1:静止，2:原地打滑(旋转/直行)，3:移动打滑
  int Update(const common::Data_WheelVel& vel) {

    return 0;
  }

  // imu
  void Update(const common::Data_Imu& imu) {
    
  }


 private:
  MotionDetector() {
  }
  MotionDetector(const MotionDetector&) = delete;
  MotionDetector& operator=(const MotionDetector&) = delete;

  void Run();

  std::atomic_bool to_stop_;
  std::thread thread_;

  TimedQueue<common::Pose> gnss_xyz_q_; // 已转换到base
  TimedQueue<common::Vel3D> vio_vel_q_; // 已转换到base
  TimedQueue<common::Vel3D> wheel_vel_q_; 
};

}  // namespace utils

#endif  // COMMON_MOTION_DETECTOR_H
