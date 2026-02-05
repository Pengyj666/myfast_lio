#include "common/sensor_monitor.h"

#include "common/common_def.h"
#include "common/log_filters.h"
#include "common/sysutils.h"
#include "droslog/log.h"

namespace utils {

SensorMonitor::SensorMonitor() {
  charging_station_state_q_.reset(128);   // 2hz, about 1min
  moving_state_q_.reset(512);
}

void SensorMonitor::FeedData(const std::shared_ptr<common::DataBase> &sp) {
  if (!(sp && sp->timestamp > 0.0))
    return;

  if (sp->GetType() == common::DataType::DATA_CHARGING_STATION_INFO) {
    static SimpleLogFilter fps_filter(200);
    if (fps_filter.Output(GetNow_Steady())) {
      auto data = std::dynamic_pointer_cast<common::Data_ChargingStationInfo>(sp);
      int state = (data->is_docking_done) ? 1 : 0;
      std::lock_guard<std::mutex> lock(charging_station_state_q_mutex_);
      charging_station_state_q_.emplace_back(state, data->timestamp);
    }
  } else if (sp->GetType() == common::DataType::DATA_WHEEL_VEL) {
    static SimpleLogFilter fps_filter(100);
    if (fps_filter.Output(GetNow_Steady())) {
      auto data = std::dynamic_pointer_cast<common::Data_WheelVel>(sp);
      int state = (data->vel.vel[0] > 0.05 || data->vel.vel[2] > 0.05) ? 1 : 0;
      std::lock_guard<std::mutex> lock(charging_station_state_q_mutex_);
      moving_state_q_.emplace_back(state, data->timestamp);
    }
  }
}

int SensorMonitor::GetChargingStationState(double ts, double dts) {
  std::lock_guard<std::mutex> lock(charging_station_state_q_mutex_);
  int cs_state = -1;
  {
    int idx = charging_station_state_q_.findAfter(ts);
    if (idx > 0) {
      cs_state = charging_station_state_q_[idx];
    }
    if (0 == idx) {
      if (charging_station_state_q_(0) + dts > ts) {
        cs_state = charging_station_state_q_[0];
      }
    }
  }
  int moving_state = -1;
  {
    int idx = moving_state_q_.findAfter(ts);
    if (idx > 0) {
      moving_state = moving_state_q_[idx];
    }
    for (int i = idx; i >= 0; i--) {
      if (moving_state_q_[i] == 1) {
        moving_state = 1;
      }
    }
    if (0 == idx) {
      if (moving_state_q_(0) + dts > ts) {
        moving_state = moving_state_q_[0];
      }
    }
  }

  if (cs_state >= 1) {
    if (moving_state == 1) {
      return 0;
    } else {
      return 1;
    }
  } else if (cs_state == 0) {
    return 0;
  }

  return -1;
}

} // namespace utils