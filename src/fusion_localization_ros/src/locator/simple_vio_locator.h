#ifndef LOCATOR_SIMPLE_LOCATOR_H_
#define LOCATOR_SIMPLE_LOCATOR_H_

#include <atomic>
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "common/data_type.h"
#include "common/timed_queue.h"

#include "common/fusion_def.h"
#include "common/simple_tracker.h"
#include "common/dist_odometer.h"

class SimpleVioLocator {
 public:
  struct Config {
    double acc_noise = 0.01;
    double gyro_noise = 0.01;
    double acc_bias_noise = 1e-4;
    double gyro_bias_noise = 1e-6;

    double imu_freq = 100.0;
    double imu_cutoff_freq = 12.0;

    double max_init_off_rtk_dist = 20.0;
    double max_init_vio_track_dist = 15.0;

    double max_off_rtk_dist = 300.0;
    double max_off_reloc_dist = 60.0;
    double max_only_iw_dist = 20.0;

    Eigen::Vector3d imu_to_gps = Eigen::Vector3d(-0.01, 0.0, 0.0);
  };

  SimpleVioLocator();
  ~SimpleVioLocator();

  void Reset();

  void SetConfig(const Config& config);
  const Config& GetConfig() const;

  // 定位状态是否有效
  bool IsValid();
  bool CheckDist();

  // 0: map, 1: loc
  // 设置工作模式意味着首次进入建图、或重新加载地图: 必须清空所有状态, 但需要上层调用Reset()
  // 避免重复加载地图的逻辑在上层做
  bool SetWorkMode(int mode);
  // -1: unknown, 0: map, 1: loc 
  int GetWorkMode() { return work_mode_.load(); }

  // state: -2: error, -1: unknown, 0: ready(loaded map), 1: init-move, 2: no-rtk-initing, 3: tracking, 4: lost
  // err_code: 0: no-err, 1: loc-lost, 2: start-without-rtk & charging-station 
  bool SetWorkState(int state, int err_code = 0);
  // state: -2: error, -1: unknown, 0: ready(loaded map), 1: init-move, 2: no-rtk-initing, 3: tracking, 4: lost
  int GetWorkState() { return work_state_.load(); }
  std::string GetWorkStateStr();

  // 0: no-err, 1: loc-lost, 2: start-without-rtk & charging-station 
  int GetErrorCode() { return error_code_.load(); }

  // return is valid only return.ts > 0.0
  // @param timestamp: 0.0 for latest
  common::NavState GetNavState(double timestamp = 0.0);

  bool SetInitNavState(const common::NavState &state);
  
  bool ProcessImuData(const common::Data_Imu &imu);
  // 送进来的都是有效固定解,且已转换到地图坐标系
  bool ProcessGpsData(const common::Data_Gnss &gnss);
  bool ProcessEstHeading(double ts, const Eigen::Quaterniond &quat);
  bool ProcessWheelData(const common::Data_WheelVel &wheel_vel);
  // type: 0-纯vio, 1-reloc (均已经转到base)
  bool ProcessVioData(const common::Data_VioResult& vio, int type = 0);
  // type: 0-纯lio, 1-reloc (均已经转到base)
  bool ProcessLioData(const common::Data_ProbPose& lio, int type = 1);

  // 0: unknown, 1: start, 2: stop 
  bool SetComputeHeadingState(int state);

  common::Data_Gnss GetGnssInit();

 private:
  // ts: 0.0 for latest
  common::Data_WheelVel EstimateWheelVel(double ts = 0.0);

  // type: 0: rtk, 1: vio-gnss, 2: vio-init
  void CorrectNavTracker(int type);

 private:
  Config config_;
  std::mutex state_mutex_;
  SimpleTracker state_tracker_; // 驱动轮中心作为定位中心
  std::atomic_bool state_tracker_valid_;    // 标识state_tracker 是否可用

  DistOdometer dist_odom_;

  std::atomic<int> work_mode_;  // -1: unknown, 0: map, 1: loc 
  std::atomic<int> work_state_; // -2: error, -1: unknown, 0: ready(loaded map), 1: init-move, 2: no-rtk-initing, 3: tracking, 4: heading-lost
  std::atomic<int> error_code_; // 0: no-err, 1: loc-lost, 2: start-without-rtk & charging-station
  
  std::atomic<int> track_type_; // 0: unknown, 1: rtk, 2: vio, 3: iw

  //定位未初始化时, rtk固定解用以输出位置
  std::mutex gnss_init_mutex_;
  common::Data_Gnss gnss_init_;

  // 数据缓存
  std::mutex imu_q_mutex_;
  std::mutex wheel_vel_q_mutex_;
  std::mutex vio_q_mutex_, vmap_odom_q_mutex_;
  utils::TimedQueue<common::Data_Imu> imu_q_;
  utils::TimedQueue<common::Data_WheelVel> wheel_vel_q_;
  utils::TimedQueue<common::Data_VioResult> vio_q_, vmap_odom_q_;

  common::Data_Gnss heading_start_gnss_;
  common::Data_Gnss heading_stop_gnss_;
};
  
#endif  // LOCATOR_SIMPLE_LOCATOR_H_