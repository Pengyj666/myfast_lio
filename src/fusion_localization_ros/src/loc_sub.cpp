
#include "version.h"
#include "loc_node.h"

#include "common/common_def.h"
#include "common/data_type.h"
#include "common/gnss_initor.h"
#include "common/gnss_monitor.h"
#include "common/sysutils.h"
#include "common/log_filters.h"
#include "common/debug_client.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"
#include "geo_utils/tf_helper.h"
#include "common/gnss_converter.h"
#include "common/sensor_monitor.h"
#include "common/vio_checker.h"
#include "common/vio_tracker.h"
#include "common/vio_gnss_initor.h"
#include "common/heading_estimator.h"
#include "common/vio_reseter.h"
#include "common/vmap_monitor.h"

using namespace utils;

void loc_node::callback_imu(const sensor_msgs::Imu::ConstPtr msg)
{
  static double pre_msg_ts = 0.0;
  double msg_ts = msg->header.stamp.toSec();
  if (msg_ts <= pre_msg_ts) {
    droslog(LogLevel::ERROR, "LOC::callback_imu() 时间戳倒退, cur_ts=%.3f, pre_ts=%.3f", msg_ts, pre_msg_ts);
    return;
  }
  if (msg_ts - pre_msg_ts > 0.05) {
    droslog(LogLevel::WARN, "LOC::callback_imu() 丢帧, cur_ts=%.3f, pre_ts=%.3f", msg_ts, pre_msg_ts);
  }
  pre_msg_ts = msg_ts;

  common::Data_Imu imu_data;
  imu_data.imu.acc << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
  imu_data.imu.gyro << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
  imu_data.imu.quat = Eigen::Quaterniond(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
  imu_data.timestamp = msg->header.stamp.toSec();

  {
    static SimpleLogFilter fps_filter(5000);
    if (fps_filter.Output(GetNow_Steady())) {
      auto imu_rpy = GetEulerRPY(imu_data.imu.quat);
      droslog(LogLevel::INFO, "LOC::callback_imu() ts=%.3f, acc=(%.3f,%.3f,%.3f), gyro=(%.3f,%.3f,%.3f), rpy=(%.3f,%.3f,%.3f)", 
          imu_data.timestamp, imu_data.imu.acc[0], imu_data.imu.acc[1], imu_data.imu.acc[2], 
          imu_data.imu.gyro[0], imu_data.imu.gyro[1], imu_data.imu.gyro[2], imu_rpy[0], imu_rpy[1], imu_rpy[2]);
    }
  }

  if (fusion_type_.load() == 0) {
    HeadingEstimator::Instance()->FeedData(imu_data);
    // TODO 送入eskf
  }

  std::lock_guard<std::mutex> lock(loc_mutex_);
  locator_.ProcessImuData(imu_data);
}

void loc_node::callback_wheel_vel(const geometry_msgs::TwistStamped::ConstPtr msg) 
{
  static double pre_msg_ts = 0.0;
  double msg_ts = msg->header.stamp.toSec();
  if (msg_ts <= pre_msg_ts) {
    droslog(LogLevel::WARN, "LOC::callback_wheel_vel() 时间戳倒退, cur_ts=%.3f, pre_ts=%.3f", msg_ts, pre_msg_ts);
    return;
  }
  if (msg_ts - pre_msg_ts > 0.05) {
    droslog(LogLevel::WARN, "LOC::callback_wheel_vel() 丢帧, cur_ts=%.3f, pre_ts=%.3f", msg_ts, pre_msg_ts);
  }
  pre_msg_ts = msg_ts;

  common::Data_WheelVel wheel_vel;
  wheel_vel.timestamp = msg->header.stamp.toSec();
  wheel_vel.vel.vel << msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z;

  {
    static SimpleLogFilter log_filter(5000);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::INFO, "LOC::callback_wheel_vel() ts%.3f, vel: lv= %.3f, av= %.3f", 
          wheel_vel.timestamp, msg->twist.linear.x, msg->twist.angular.z);
    }
  }

  if (fusion_type_.load() == 0) {
    HeadingEstimator::Instance()->FeedData(wheel_vel);
    // TODO 送入eskf
  }

  std::lock_guard<std::mutex> lock(loc_mutex_);
  locator_.ProcessWheelData(wheel_vel);  
}

void loc_node::callback_sensor_info(const mower_msgs::MowerSensorInfo::ConstPtr &msg) {
  bool is_charging = msg->is_charging;
  bool is_docking_done = msg->is_docking_done;
  bool is_debug_docking = DebugClient::Instance()->GetDockingState();
  if (is_debug_docking) {
    {
      static SimpleLogFilter log_filter(4000);
      if (log_filter.Output(GetNow_Steady())) {
        droslog(LogLevel::WARN, "LOC::callback_sensor_info() 调试状态, 强制设置在桩, 原状态: is_docking_done=%d, is_charging=%d", is_docking_done, is_charging);
      }
    }
    is_docking_done = true;
    is_charging = true;
  }

  {
    static SimpleLogFilter log_filter(5000);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::INFO, "LOC::callback_sensor_info() sensor_info: 是否在桩: %d", is_docking_done);
    }
  }

  auto sp = std::make_shared<common::Data_ChargingStationInfo>();
  sp->timestamp = msg->header.stamp.toSec();
  sp->is_charging = is_charging;
  sp->is_docking_done = is_docking_done;
  SensorMonitor::Instance()->FeedData(sp);
}

void loc_node::callback_unicore_nav(const mower_gps_msgs::UnicoreNav::ConstPtr &msg)
{
  std::string rtk_type = msg->position_type;
  double msg_ts = msg->header.stamp.toSec();

  {
    static double pre_msg_ts = 0.0;
    if (msg_ts <= pre_msg_ts) {
      droslog(LogLevel::WARN, "LOC::callback_unicore_nav() rtk数据时间戳倒退, cur_ts=%.3f, pre_ts=%.3f", msg_ts, pre_msg_ts);
    }
    if (pre_msg_ts > 0.0 && msg_ts - pre_msg_ts > 0.5) {
      droslog(LogLevel::WARN, "LOC::callback_unicore_nav() rtk数据时间戳跳动较大, cur_ts=%.3f, pre_ts=%.3f", msg_ts, pre_msg_ts);
    }
    pre_msg_ts = msg_ts;
  }

  // 应用显示相关信息
  localization_info_msg_.num_satellites = msg->num_satellites_tracked;
  localization_info_msg_.num_satellites_used = msg->num_satellites_used_in_solution;
  localization_info_msg_.ref_station_status = msg->ref_station_status;
  localization_info_msg_.lora_rssi_dbm = msg->lora_rssi_dbm;
  localization_info_msg_.rtk_status = rtk_type;

  // 检查基站是否有效
  bool cur_rtk_ref_valid = GnssMonitor::Instance()->CheckRtkRef(msg_ts);
  bool cur_rtk_ref_setted = GnssConverter::Instance()->CurRtkBaseStationValid();
  bool local_map_offset_valid = GnssConverter::Instance()->LocalMapOffsetValid();
  
  if (!cur_rtk_ref_valid) {
    static double pre_ts = 0.0;
    if (msg_ts - pre_ts > 3.0 && common::RTK_SINGLE != rtk_type) {
      droslog(LogLevel::WARN, "LOC::callback_unicore_nav() rtk基站状态检查无效, 将rtk置为单点解, 原rtk解类型=%s, lla=%.8f, %.8f, %.3f", 
        rtk_type.c_str(), msg->latitude, msg->longitude, msg->height);
        pre_ts = msg_ts;
    }
    rtk_type = common::RTK_SINGLE;
  }

  // 调试相关状态
  const int debug_rtk_state = DebugClient::Instance()->GetRtkState();
  if (debug_rtk_state == 1) {
    rtk_type = common::RTK_SINGLE;
    
    static double pre_ts = 0.0;
    if (msg_ts - pre_ts > 5.0) {
      droslog(LogLevel::WARN, "LOC::callback_unicore_nav() 调试中, 强制将RTK状态设置为单点解");
      pre_ts = msg_ts;
    }
  } else if (debug_rtk_state == 2) {
    rtk_type = common::RTK_NARROW_INT;

    static double pre_ts = 0.0;
    if (msg_ts - pre_ts > 5.0) {
      droslog(LogLevel::WARN, "LOC::callback_unicore_nav() 调试中, 强制将RTK状态设置为固定解");
      pre_ts = msg_ts;
    }
  } else if (debug_rtk_state == 3) {
    static double pre_ts = 0.0;
    if (msg_ts - pre_ts > 5.0) {
      droslog(LogLevel::WARN, "LOC::callback_unicore_nav() 调试中, 强制屏蔽RTK不输入");
      pre_ts = msg_ts;
    }
    return;
  }

  // 转换到局部地理坐标
  auto enu = GnssConverter::Instance()->Gnss2Enu(Eigen::Vector3d(msg->latitude, msg->longitude, msg->height));
  // 转到局部地图坐标
  auto local_xyz = GnssConverter::Instance()->Enu2LocalPos(enu);

  // 输出数据log
  {
    static SimpleLogFilter log_filter(5000);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::INFO, "LOC::callback_unicore_nav() ts=%.3f, rtk_type=[%s], gps=%.7f,%.7f,%.7f, lla_sigma=%.3f,%.3f,%.3f enu= %.3f, %.3f, %.3f, local_xyz=%.3f,%.3f,%.3f",
          msg_ts, rtk_type.c_str(),
          msg->latitude, msg->longitude, msg->height,
          msg->lat_sigma, msg->lon_sigma, msg->height_sigma,
          enu.x(), enu.y(), enu.z(), local_xyz.x(), local_xyz.y(), local_xyz.z());
    }
  }

  if (fusion_type_.load() != 0) {
    return;
  }

  common::Data_Gnss gnss;
  gnss.timestamp = msg_ts;
  gnss.gnss.rtk_type = rtk_type;
  gnss.gnss.lla << msg->latitude, msg->longitude, msg->height;
  gnss.gnss.enu = enu;
  gnss.gnss.cov << msg->lat_sigma * msg->lat_sigma, 0.0, 0.0,
                  0.0, msg->lon_sigma * msg->lon_sigma, 0.0,
                  0.0, 0.0, msg->height_sigma * msg->height_sigma;

  last_gnss_data_ = gnss;

  // 数据送到GNSS监控器
  GnssMonitor::Instance()->Update(gnss);

  GnssInitor::Instance()->FeedData(gnss);
  HeadingEstimator::Instance()->FeedData(gnss);
  
  int work_mode = locator_.GetWorkMode();
  if (0 == work_mode) {
    bool gnss_init = GnssConverter::Instance()->LocalMapOffsetValid();
    if (!gnss_init) {
      VioGnssInitor::Instance()->FeedGnss(gnss);
      if (VioGnssInitor::Instance()->IsGnssMapOffsetValid()) {
        // 无RTK下桩gnss偏移已计算
        auto gnss_map_offset = VioGnssInitor::Instance()->GetGnssMapOffset();
        auto rpy = GetEulerRPY(gnss_map_offset.pose.quat);
        
        Eigen::Vector3d base_station_gps = GnssConverter::Instance()->Enu2Gnss(Eigen::Vector3d(0,0,0));
        Eigen::Vector3d charging_station_gps = GnssConverter::Instance()->Enu2Gnss(gnss_map_offset.pose.pos);
        Eigen::Vector3d charging_station_rpy;
        charging_station_rpy << 0.0, 0.0, rpy(2);
        GnssConverter::Instance()->SetLocalMapOffset(base_station_gps, charging_station_gps, charging_station_rpy);
        
        droslog(LogLevel::INFO, "LOC::callback_unicore_nav() 建图 融合器vio-init计算gnss-map完成, 桩点地理朝向: %f deg, 桩点地理坐标: %.3f,%.3f,%.3f", 
            rpy[2] * 180.0 / M_PI, gnss_map_offset.pose.pos[0], gnss_map_offset.pose.pos[1], gnss_map_offset.pose.pos[2]);
      } else {
        static SimpleLogFilter log_filter(3000);
        if (log_filter.Output(GetNow_Steady())) {
          droslog(LogLevel::INFO, "LOC::callback_unicore_nav() 建图 融合器vio-init计算gnss-map中......");
        }
      }
    }
  }
  
  if (cur_rtk_ref_valid && cur_rtk_ref_setted && local_map_offset_valid) {
    common::Data_Gnss gnss_xyz = gnss;
    gnss_xyz.gnss.enu = local_xyz;
    VioTracker::Instance()->FeedData(gnss_xyz);

    geometry_msgs::PoseWithCovarianceStamped gps_xyz_msg;
    gps_xyz_msg.header = msg->header;
    gps_xyz_msg.header.frame_id = "local_gps";
    gps_xyz_msg.pose.pose.position.x = local_xyz.x();
    gps_xyz_msg.pose.pose.position.y = local_xyz.y();
    gps_xyz_msg.pose.pose.position.z = local_xyz.z();
    if (rtk_type == common::RTK_NARROW_INT) {
      gps_xyz_msg.pose.covariance[0] = 1.0;   // rtk type: 0-NONE, 1-NARROW_INT, 2-NARROW_FLOAT, 3-SINGLE
    } else if (rtk_type == common::RTK_NARROW_FLOAT) {
      gps_xyz_msg.pose.covariance[0] = 2.0;
    } else if (rtk_type == common::RTK_SINGLE) {
      gps_xyz_msg.pose.covariance[0] = 3.0;
    } else {
      gps_xyz_msg.pose.covariance[0] = -1.0;
    }
    double sigma = std::sqrt(msg->lat_sigma * msg->lat_sigma + msg->lon_sigma * msg->lon_sigma);
    gps_xyz_msg.pose.covariance[1] = sigma;
    gps_xyz_msg.pose.covariance[2] = sigma;
    gps_xyz_msg.pose.covariance[3] = msg->height_sigma;

    local_gps_pub_.publish(gps_xyz_msg);
    
    // 当前rtk为固定解
    if (rtk_type == common::RTK_NARROW_INT) {
      // TODO 送入eskf
  
      auto enu_heading = HeadingEstimator::Instance()->GetEnuHeading();
      auto local_q = GnssConverter::Instance()->EnuQ2LocalQ(enu_heading.pose.quat);
  
      std::lock_guard<std::mutex> lock(loc_mutex_);
      locator_.ProcessGpsData(gnss_xyz);

      // TODO 如果vio姿态没有修正, 则使用此修正
      if (!VioTracker::Instance()->IsVioValid(30)) {
        locator_.ProcessEstHeading(enu_heading.timestamp, local_q);
        static SimpleLogFilter log_filter(2000);
        if (log_filter.Output(GetNow_Steady())) {
          auto rpy = GetEulerRPY(enu_heading.pose.quat);
          droslog(LogLevel::INFO, "LOC::callback_unicore_nav() 使用est-heading修正姿态: rpy=%.3f %.3f %.3f", rpy[0], rpy[1], rpy[2]);
        }
      }
    }
  }
}

// 基站状态由GnssRtkMonitor监控及过滤跳变数据, 更新时对GnssConverter进行更新
// 外部判断基站状态是否可用, 通过GnssConverter的接口判断
void loc_node::callback_rtk_ref(const sensor_msgs::NavSatFix::ConstPtr &msg) {
  if (fusion_type_.load() != 0) {
    return;
  }

  double lat = msg->latitude;
  double lon = msg->longitude;
  double alt = msg->altitude;

  if (DebugClient::Instance()->IsNeedRtkRefChange(GetNow_Steady())) {
    droslog(LogLevel::WARN, "LOC::callback_rtk_ref() 调试状态, 修改RTK基站坐标的纬度为0.0");
    lat = 0.0;
  }

  if (msg->status.status == sensor_msgs::NavSatStatus::STATUS_FIX) {
    if (1 == GnssMonitor::Instance()->UpdateRtkRef(msg->header.stamp.toSec(), lat, lon, alt, true)) {
      GnssConverter::Instance()->SetCurRtkBaseStation(Eigen::Vector3d(lat, lon, alt));
    }
  } else {
    GnssMonitor::Instance()->UpdateRtkRef(msg->header.stamp.toSec(), lat, lon, alt, false);
    static SimpleLogFilter log_filter(5000);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::WARN, "LOC::callback_rtk_ref() RTK基站尚未固定");
    }
  }
}

void loc_node::callback_cmd_vel(const geometry_msgs::Twist::ConstPtr &msg) {
  {
    static SimpleLogFilter log_filter(5000);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::INFO, "LOC::callback_cmd_vel() cmd_vel: lv= %.3f, av= %.3f", 
          msg->linear.x, msg->angular.z);
    }
  }
}