#include "common/data_type.h"

#include <cstring>
#include <map>

namespace common {
  
namespace {
const std::map<DataType, std::string> k_type_str_map{
    {DataType::MARK_FIRST, "MARK_FIRST"},
    {DataType::DATA_IMU, "DATA_IMU"},
    {DataType::DATA_GNSS, "DATA_GNSS"},
    {DataType::DATA_WHEEL_VEL, "DATA_WHEEL_VEL"},
    {DataType::DATA_VIO_RESULT, "DATA_VIO_RESULT"},
    {DataType::DATA_POSE, "DATA_POSE"},
    {DataType::DATA_PROB_POSE, "DATA_PROB_POSE"},
    {DataType::DATA_RTK_GNSS, "DATA_RTK_GNSS"},
    {DataType::DATA_CHARGING_STATION_INFO, "DATA_CHARGING_STATION_INFO"},
    {DataType::DATA_NAV_STATE, "DATA_NAV_STATE"},
    {DataType::DATA_ODOM, "DATA_ODOM"},
    {DataType::DATA_ODOM_WITH_COV, "DATA_ODOM_WITH_COV"},
    };
} // namespace

const std::string& GetDataTypeStr(DataType type) {
  if (k_type_str_map.count(type) > 0) {
    return k_type_str_map.at(type);
  }
  return k_type_str_map.at(DataType::MARK_FIRST);
}

std::shared_ptr<DataBase> CreateData(DataType type) {
  if (DataType::MARK_FIRST == type) {
  } else if (DataType::DATA_IMU == type) {
    return std::make_shared<Data_Imu>();
  } else if (DataType::DATA_GNSS == type) {
    return std::make_shared<Data_Gnss>();
  } else if (DataType::DATA_WHEEL_VEL == type) {
    return std::make_shared<Data_WheelVel>();
  } else if (DataType::DATA_VIO_RESULT == type) {
    return std::make_shared<Data_VioResult>();
  } else if (DataType::DATA_POSE == type) {
    return std::make_shared<Data_Pose>();
  } else if (DataType::DATA_PROB_POSE == type) {
    return std::make_shared<Data_ProbPose>();
  } else if (DataType::DATA_RTK_GNSS == type) {
    return std::make_shared<Data_RtkGnss>();
  } else if (DataType::DATA_CHARGING_STATION_INFO == type) {
    return std::make_shared<Data_ChargingStationInfo>();
  } else if (DataType::DATA_NAV_STATE == type) {
    return std::make_shared<Data_NavState>();
  } else if (DataType::DATA_ODOM == type) {
    return std::make_shared<Data_Odometry>();
  } else if (DataType::DATA_ODOM_WITH_COV == type) {
    return std::make_shared<Data_OdomWithCov>();
  } 
  return nullptr;
}

std::shared_ptr<DataBase> CloneData(std::shared_ptr<DataBase> sp) {
  if (!sp.get()) {
    return nullptr;
  }
  auto type = sp->GetType();
  if (DataType::MARK_FIRST == type) {
  } else if (DataType::DATA_IMU == type) {
    auto spp = std::dynamic_pointer_cast<Data_Imu>(sp);
    return std::make_shared<Data_Imu>(*spp);
  } else if (DataType::DATA_GNSS == type) {
    auto spp = std::dynamic_pointer_cast<Data_Gnss>(sp);
    return std::make_shared<Data_Gnss>(*spp);
  } else if (DataType::DATA_WHEEL_VEL == type) {
    auto spp = std::dynamic_pointer_cast<Data_WheelVel>(sp);
    return std::make_shared<Data_WheelVel>(*spp);
  } else if (DataType::DATA_VIO_RESULT == type) {
    auto spp = std::dynamic_pointer_cast<Data_VioResult>(sp);
    return std::make_shared<Data_VioResult>(*spp);
  } else if (DataType::DATA_POSE == type) {
    auto spp = std::dynamic_pointer_cast<Data_Pose>(sp);
    return std::make_shared<Data_Pose>(*spp);
  } else if (DataType::DATA_PROB_POSE == type) {
    auto spp = std::dynamic_pointer_cast<Data_ProbPose>(sp);
    return std::make_shared<Data_ProbPose>(*spp);
  } else if (DataType::DATA_RTK_GNSS == type) {
    auto spp = std::dynamic_pointer_cast<Data_RtkGnss>(sp);
    return std::make_shared<Data_RtkGnss>(*spp);
  } else if (DataType::DATA_CHARGING_STATION_INFO == type) {
    auto spp = std::dynamic_pointer_cast<Data_ChargingStationInfo>(sp);
    return std::make_shared<Data_ChargingStationInfo>(*spp);
  } else if (DataType::DATA_NAV_STATE == type) {
    auto spp = std::dynamic_pointer_cast<Data_NavState>(sp);
    return std::make_shared<Data_NavState>(*spp);
  } else if (DataType::DATA_ODOM == type) {
    auto spp = std::dynamic_pointer_cast<Data_Odometry>(sp);
    return std::make_shared<Data_Odometry>(*spp);
  } else if (DataType::DATA_ODOM_WITH_COV == type) {
    auto spp = std::dynamic_pointer_cast<Data_OdomWithCov>(sp);
    return std::make_shared<Data_OdomWithCov>(*spp);
  } 
  return nullptr;
}

} // namespace common