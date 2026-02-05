#ifndef FUSION_SIMPLE_TRACKER_H
#define FUSION_SIMPLE_TRACKER_H

#include "common/data_type.h"
#include "common/data_utils.h"
#include "common/timed_queue.h"

#include "common/dist_odometer.h"

// 注意: 非多线程安全, 调用者负责多线程安全
// 以后轮中心为参考点
class SimpleTracker {
 public:
  SimpleTracker(int capacity = 0) : states_(capacity) {}

  void Reset();

  int Capacity() const { return states_.cap(); }
  int Size() const { return states_.size(); }
  double GetLastTimestamp() const;

  // @param ts: 0-the latest state, >0-the state corresponding to ts
  // The result is valid when its timestamp > 0
  common::NavState GetState(double ts = 0.0) const;

  bool InitOrigin(const common::NavState &state);

  bool Predict(double ts, const common::ImuData &imu, const common::WheelVel &wvel);
  
  // return 0: in between, 1: too new, 2: too old, -1: error, -2: error2
  // only 0 means successful
  // 0: iw, 1: rtk, 2: vio-init, 3: vio-gnss, 4: vio-vreloc, 5: lio-reloc
  int  Measurement(double ts, const Eigen::Vector3d &pos, int type = 0);
  // return 0: in between, 1: too new, 2: too old, -1: error, -2: error2
  // only 0 means successful
  // 0: iw, 1: rtk, 2: vio-init, 3: vio-gnss, 4: vio-vreloc, 5: lio-reloc
  int  Measurement(double ts, const Eigen::Quaterniond &quat, int type = 0);
  // return 0: in between, 1: too new, 2: too old, -1: error, -2: error2
  // only 0 means successful
  // 0: iw, 1: rtk, 2: vio-init, 3: vio-gnss, 4: vio-vreloc, 5: lio-reloc
  int  Measurement(double ts, const Eigen::Vector3d &pos, const Eigen::Quaterniond &quat, int type = 0);

  // return 0: in between, 1: too new, 2: too old, -1: error 
  int  FindTs(double ts) const;
  // return 0: in between, 1: too new(ts >= (0)), 2: too old, -1: error 
  int  EstimateState(double ts,  common::NavState *state, int *ind) const;

 private:
  common::NavState UpdateByVel(const common::NavState &pre_state, 
      const Eigen::Vector3d &vel, const Eigen::Vector3d &ang_vel, double new_ts) const;

  utils::TimedQueue<common::NavState> states_;

  DistOdometer dist_odometer_;
};

#endif//FUSION_SIMPLE_TRACKER_H