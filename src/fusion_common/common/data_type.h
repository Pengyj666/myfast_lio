#ifndef UTILS_COMMON_DATA_TYPE_H
#define UTILS_COMMON_DATA_TYPE_H

#include <memory>

#include "common/data_type_basic.h"

namespace common {

enum DataType {
  MARK_FIRST,     // NEVER change
  DATA_IMU,
  DATA_GNSS,
  DATA_WHEEL_VEL,
  DATA_VIO_RESULT,
  DATA_POSE,
  DATA_PROB_POSE,
  DATA_MOTION_STATE,
  DATA_RTK_GNSS,
  DATA_CHARGING_STATION_INFO,  // 充电桩状态数据
  DATA_NAV_STATE,
  DATA_ODOM,
  DATA_ODOM_WITH_COV,
};

class DataBase {
 public:
  virtual ~DataBase() {}
  virtual DataType GetType() const = 0;

  double timestamp = 0.0; // sec
};

const std::string& GetDataTypeStr(DataType type);
std::shared_ptr<DataBase> CreateData(DataType type);
std::shared_ptr<DataBase> CloneData(std::shared_ptr<DataBase> sp);

class Data_Imu : public DataBase {
 public:
  DataType GetType() const override {
    return DataType::DATA_IMU;
  }

  ImuData imu;
};

class Data_Gnss : public DataBase {
 public:
  DataType GetType() const override {
    return DataType::DATA_GNSS;
  }

  GnssData gnss;
};

class Data_RtkGnss : public DataBase {
 public:
  DataType GetType() const override {
    return DataType::DATA_RTK_GNSS;
  }

  RtkGnss rtk;
};

class Data_WheelVel : public DataBase {
 public:
  DataType GetType() const override {
    return DataType::DATA_WHEEL_VEL;
  }

  WheelVel vel;
};

class Data_VioResult : public DataBase {
 public:
  DataType GetType() const override {
    return DataType::DATA_VIO_RESULT;
  }
  
  int confidence = -1;   // -1: invalid, 0: failure, 1: low, 2: mid, 3: good 
  Odometry vio;
};

class Data_Pose : public DataBase {
 public:
  DataType GetType() const override {
    return DataType::DATA_POSE;
  }
  
  Pose pose;
};

class Data_ProbPose : public DataBase {
 public:
  DataType GetType() const override {
    return DataType::DATA_PROB_POSE;
  }
  
  ProbPose ppose;
};

class Data_Odometry  : public DataBase {
 public:
  DataType GetType() const override {
    return DataType::DATA_ODOM;
  }

  Odometry odom;
};

class Data_NavState : public DataBase {
 public:
  DataType GetType() const override {
    return DataType::DATA_NAV_STATE;
  }

  NavState nav_state;
};

class Data_ChargingStationInfo : public DataBase {
 public:
  DataType GetType() const override {
    return DataType::DATA_CHARGING_STATION_INFO;
  }
  
  bool is_charging = false;       // 充电桩是否在充电
  bool is_docking_done = false;   // 电极是否对接到位, 可能有几秒响应误差
};

class Data_OdomWithCov : public DataBase {
 public:
  DataType GetType() const override {
    return DataType::DATA_ODOM_WITH_COV;
  }
  
  OdomWithCov odom_with_cov;
};

} // namespace common
#endif//UTILS_COMMON_DATA_TYPE_H
