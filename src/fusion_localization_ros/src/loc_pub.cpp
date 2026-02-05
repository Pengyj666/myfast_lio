
#include "version.h"
#include "loc_node.h"

#include "common/gnss_monitor.h"
#include "common/sysutils.h"
#include "common/log_filters.h"
#include "common/sensor_monitor.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"
#include "geo_utils/tf_helper.h"
#include "common/vio_tracker.h"

#include "common/gnss_converter.h"

using namespace utils;
using namespace common;

void loc_node::PubDebugInfo()
{
  {
    static SimpleLogFilter fps_filter(100);
    if (fps_filter.Output(GetNow_Steady())) {
      auto vio_local_pose = VioTracker::Instance()->GetLastVioLocalXyz();
      if (vio_local_pose.timestamp > 0.0) {
        auto vio_gnss = GnssConverter::Instance()->LocalPos2Gnss(vio_local_pose.pose.pos);
        sensor_msgs::NavSatFix vio_gnss_msg;
        vio_gnss_msg.header.stamp.fromSec(vio_local_pose.timestamp);
        vio_gnss_msg.header.frame_id = "vio_utm";
        vio_gnss_msg.latitude = vio_gnss.x();
        vio_gnss_msg.longitude = vio_gnss.y();
        vio_gnss_msg.altitude = vio_gnss.z();
        if (GnssConverter::Instance()->CurRtkBaseStationValid()) {
          vio_gnss_msg.status.status = sensor_msgs::NavSatStatus::STATUS_FIX;
        } else {
          vio_gnss_msg.status.status = sensor_msgs::NavSatStatus::STATUS_NO_FIX;
        }
        vio_gnss_pub_.publish(vio_gnss_msg);

        auto vio_xyz = TFHelper::Instance()->TF_Gps2Base(vio_local_pose.pose.pos, vio_local_pose.pose.quat);
        geometry_msgs::PoseWithCovarianceStamped gps_xyz_msg;
        gps_xyz_msg.header.stamp.fromSec(vio_local_pose.timestamp);
        gps_xyz_msg.header.frame_id = "local_gps";
        gps_xyz_msg.pose.pose.position.x = vio_xyz.pos.x();
        gps_xyz_msg.pose.pose.position.y = vio_xyz.pos.y();
        gps_xyz_msg.pose.pose.position.z = vio_xyz.pos.z();
        gps_xyz_msg.pose.pose.orientation.x = vio_xyz.quat.x();
        gps_xyz_msg.pose.pose.orientation.y = vio_xyz.quat.y();
        gps_xyz_msg.pose.pose.orientation.z = vio_xyz.quat.z();
        gps_xyz_msg.pose.pose.orientation.w = vio_xyz.quat.w();
        
        local_vio_pub_.publish(gps_xyz_msg);
      }
    }
  }
  {
    static SimpleLogFilter fps_filter(1000);
    if (fps_filter.Output(GetNow_Steady())) {
      {
        auto tpose = VioTracker::Instance()->GetVioTF();
        nav_msgs::Odometry tpose_msg;
        tpose_msg.header.stamp = ros::Time::now();
        tpose_msg.header.frame_id = "vio_gnss_align_result";
        tpose_msg.pose.pose.position.x = tpose.data.pos.x();
        tpose_msg.pose.pose.position.y = tpose.data.pos.y();
        tpose_msg.pose.pose.position.z = tpose.data.pos.z();
        tpose_msg.pose.pose.orientation.x = tpose.data.quat.x();
        tpose_msg.pose.pose.orientation.y = tpose.data.quat.y();
        tpose_msg.pose.pose.orientation.z = tpose.data.quat.z();
        tpose_msg.pose.pose.orientation.w = tpose.data.quat.w();
        tpose_msg.pose.covariance[0] = tpose.ts;
        vio_gnss_align_pub_.publish(tpose_msg);
      }
      {
        auto tpose = lidar_tracker_.GetTF();
        nav_msgs::Odometry tpose_msg;
        tpose_msg.header.stamp = ros::Time::now();
        tpose_msg.header.frame_id = "lio_gnss_align_result";
        tpose_msg.pose.pose.position.x = tpose.data.pos.x();
        tpose_msg.pose.pose.position.y = tpose.data.pos.y();
        tpose_msg.pose.pose.position.z = tpose.data.pos.z();
        tpose_msg.pose.pose.orientation.x = tpose.data.quat.x();
        tpose_msg.pose.pose.orientation.y = tpose.data.quat.y();
        tpose_msg.pose.pose.orientation.z = tpose.data.quat.z();
        tpose_msg.pose.pose.orientation.w = tpose.data.quat.w();
        tpose_msg.pose.covariance[0] = tpose.ts;
        lio_gnss_align_pub_.publish(tpose_msg);
      }
    }
  }
}

void loc_node::PubLocalizationInfo()
{
  int work_mode = locator_.GetWorkMode();
  int work_state = locator_.GetWorkState();
  auto cur_ns = locator_.GetNavState();  // 已在运动中心
  
  if (cur_ns.pos.hasNaN() || cur_ns.quat.matrix().hasNaN()) {
    droslog(LogLevel::WARN, "LOC::PubLocalizationInfo() TMD 这是什么鬼 **************** base_fused_pose has nan");
    locator_.Reset();
    cur_ns = locator_.GetNavState();
  }
  common::Pose base_fused_pose;
  base_fused_pose.pos = cur_ns.pos;
  base_fused_pose.quat = cur_ns.quat;

  // 发布定位
  nav_msgs::Odometry odom_fused;
  odom_fused.header.frame_id = "map";  //map
  odom_fused.child_frame_id = "base_link";

  odom_fused.header.stamp = ros::Time::now();
  if (cur_ns.timestamp > 0.0) {
    odom_fused.pose.pose.position.x = base_fused_pose.pos.x();
    odom_fused.pose.pose.position.y = base_fused_pose.pos.y();
    odom_fused.pose.pose.position.z = base_fused_pose.pos.z();
    odom_fused.pose.pose.orientation.x = base_fused_pose.quat.x();
    odom_fused.pose.pose.orientation.y = base_fused_pose.quat.y();
    odom_fused.pose.pose.orientation.z = base_fused_pose.quat.z();
    odom_fused.pose.pose.orientation.w = base_fused_pose.quat.w();
  } else {
    odom_fused.pose.pose.position.x = 0.0;
    odom_fused.pose.pose.position.y = 0.0;
    odom_fused.pose.pose.position.z = 0.0;
    odom_fused.pose.pose.orientation.x = 0.0;
    odom_fused.pose.pose.orientation.y = 0.0;
    odom_fused.pose.pose.orientation.z = 0.0;
    odom_fused.pose.pose.orientation.w = 1.0;
  }
  {
    static SimpleLogFilter fps_filter(10);
    if (fps_filter.Output(GetNow_Steady())) {
      fused_odom_pub_.publish(odom_fused);
    }
  }
  {
    static SimpleLogFilter fps_filter(100);
    if (fps_filter.Output(GetNow_Steady())) {
      auto fused_gnss = GnssConverter::Instance()->LocalPos2MapGnss(base_fused_pose.pos);
      sensor_msgs::NavSatFix fix_fused_msg;
      fix_fused_msg.header.stamp = odom_fused.header.stamp;
      fix_fused_msg.header.frame_id = "utm";
      fix_fused_msg.latitude = fused_gnss.x();
      fix_fused_msg.longitude = fused_gnss.y();
      fix_fused_msg.altitude = fused_gnss.z();
      if (GnssConverter::Instance()->LocalMapOffsetValid()) {
        fix_fused_msg.status.status = sensor_msgs::NavSatStatus::STATUS_FIX;
      } else {
        fix_fused_msg.status.status = sensor_msgs::NavSatStatus::STATUS_NO_FIX;
      }
      fused_gps_pub_.publish(fix_fused_msg);
    }
  }

  // 发布定位状态
  bool loc_valid = false;
  double remain_rtk_dist = track_off_rtk_dist_;
  double remain_reloc_dist = track_off_reloc_dist_;
  double remain_iw_dist = track_only_iw_dist_;
  if (cur_ns.timestamp > 0.0) {
    loc_valid = true;
    
    localization_info_msg_.off_rtk_dist = cur_ns.off_rtk_dist;
    localization_info_msg_.off_reloc_dist = cur_ns.off_reloc_dist;
    localization_info_msg_.only_iw_dist = cur_ns.only_iw_dist;

    if (work_mode == 1) {
      if (cur_ns.only_iw_dist < track_only_iw_dist_) {
      } else {
        loc_valid = false;
      }
      remain_rtk_dist = track_off_rtk_dist_ - cur_ns.off_rtk_dist;
      remain_reloc_dist = track_off_reloc_dist_ - cur_ns.off_reloc_dist;
      remain_iw_dist = track_only_iw_dist_ - cur_ns.only_iw_dist;
    } else if (work_mode == 0) {
      float dist = track_off_rtk_dist_;
      if ((fusion_type_.load() == 0 && use_vmap_.load()) || fusion_type_.load() == 1 || fusion_type_.load() == 2) {
        dist = 10000.0;
      }
      if (cur_ns.off_rtk_dist < dist && cur_ns.off_reloc_dist < dist && cur_ns.only_iw_dist < track_only_iw_dist_) {
      } else {
        loc_valid = false;
      }
      remain_rtk_dist = dist - cur_ns.off_rtk_dist;
      remain_reloc_dist = dist - cur_ns.off_reloc_dist;
      remain_iw_dist = dist - cur_ns.only_iw_dist;
    } else {
      loc_valid = false;
    }
  }
  double remain_dist = std::min(remain_rtk_dist, remain_reloc_dist);
  remain_dist = std::min(remain_dist, remain_iw_dist);
  localization_info_msg_.remaining_vision_buffer = remain_dist;
  
  if (loc_valid) {
    localization_info_msg_.state = LocState::RTK_VISION_FUSION;
    localization_info_msg_.heading_initialized = true;
  } else {
    localization_info_msg_.heading_initialized = false;
    localization_info_msg_.state = LocState::LOST;
  }

  {
    static bool pre_loc_valid = true;
    if (pre_loc_valid != loc_valid) {
      droslog(LogLevel::INFO, "LOC::PubLocalizationInfo() ###### 定位状态变换 %d -> %d", pre_loc_valid, loc_valid);
      pre_loc_valid = loc_valid;
    }
  }
  
  // 在桩及固定解
  int CS_state = SensorMonitor::Instance()->GetChargingStationState(ros::Time::now().toSec());
  if (CS_state >= 1) {
    localization_info_msg_.state = LocState::RTK_VISION_FUSION;
  }
  bool gnss_valid = GnssMonitor::Instance()->IsGnssValid();
  if (gnss_valid) {
    localization_info_msg_.state = LocState::RTK_VISION_FUSION;
  }

  // 姿态角是否异常
  {
    Eigen::Quaterniond q(odom_fused.pose.pose.orientation.w, odom_fused.pose.pose.orientation.x,
                        odom_fused.pose.pose.orientation.y, odom_fused.pose.pose.orientation.z);
    auto rpy = GetEulerRPY(q);
    if (std::abs(rpy[0]) > 0.8 || std::abs(rpy[1]) > 0.8 || std::abs(rpy[0]) + std::abs(rpy[1]) > 1.5) {
      localization_info_msg_.heading_initialized = false;

      static SimpleLogFilter log_filter(1000);
      if (log_filter.Output(GetNow_Steady())) {
        droslog(LogLevel::WARN, "LOC::PubLocalizationInfo() ###### 姿态角异常rpy: %.3f %.3f %.3f", rpy[0], rpy[1], rpy[2]);
      }
    }
  }

  localization_info_msg_.datum_initialized = GnssConverter::Instance()->LocalMapOffsetValid();
  localization_info_msg_.fused_pose.header = odom_fused.header;
  localization_info_msg_.fused_pose.pose = odom_fused.pose.pose;
  {
    static SimpleLogFilter fps_filter(100);
    if (fps_filter.Output(GetNow_Steady())) {
      localization_info_pub_.publish(localization_info_msg_);
    }
  }

  {
    static long long pre_ts = 0;
    if(GetNow_Steady() > pre_ts +1000) {
      int error_code = locator_.GetErrorCode();
      pre_ts = GetNow_Steady();
      auto rpy = GetEulerRPY(cur_ns.quat);
      droslog(LogLevel::INFO, "LOC::PubLocalizationInfo() 当前融合状态: fusion_type=%d, use_vmap=%d, work_mode=%d, work_state=%d, error_code=%d, CS_state=%d, 3dist=%.3f,%.3f,%.3f, cur_ns.ts=%.3f, pos=%.3f,%.3f,%.3f, rpy=%.3f,%.3f,%.3f, localization_info_msg_: map_datum_setted:%d, state: %d, heading_inited: %d", 
          fusion_type_.load(), use_vmap_.load(), work_mode, work_state, error_code, 
          CS_state, cur_ns.off_rtk_dist, cur_ns.off_reloc_dist, cur_ns.only_iw_dist, 
          cur_ns.timestamp, cur_ns.pos[0], cur_ns.pos[1], cur_ns.pos[2], rpy[0], rpy[1], rpy[2], 
          localization_info_msg_.datum_initialized, localization_info_msg_.state, localization_info_msg_.heading_initialized);
    }
  }


  // ------------------ TF ------------------ //

  nav_msgs::Odometry odom_fused_for_tf = odom_fused;
  auto ts_for_tf = ros::Time::now();
  if ( tf_buffer_.canTransform("odom", "base_link", ros::Time(0)) )
  {
    try
    {
      // lookup transform odom -> base_link
      geometry_msgs::TransformStamped odomBaseLinkTransMsg = tf_buffer_.lookupTransform("odom", "base_link", ros::Time(0));

      // convert to tf2
      tf2::Transform odomBaseLinkTrans;
      tf2::fromMsg(odomBaseLinkTransMsg.transform, odomBaseLinkTrans);

      // get map -> base_link transform from fused state
      tf2::Transform mapBaseLinkTrans;
      tf2::fromMsg(odom_fused_for_tf.pose.pose, mapBaseLinkTrans);

      // get map -> odom transform from odom -> base_link transform
      tf2::Transform mapOdomTrans = mapBaseLinkTrans * odomBaseLinkTrans.inverse();

      // broadcast map -> odom transform
      geometry_msgs::TransformStamped mapOdomTransMsg;
      tf2::convert(mapOdomTrans, mapOdomTransMsg.transform);
      mapOdomTransMsg.header.stamp = ts_for_tf;
      mapOdomTransMsg.header.frame_id = "map";
      mapOdomTransMsg.child_frame_id = "odom";

      tf_broadcaster_.sendTransform(mapOdomTransMsg);
    } catch (tf2::TransformException &ex) {
      ROS_WARN("Could not transform map -> odom: %s", ex.what());
      droslog(LogLevel::WARN, "LOC::SendTf() Could not transform -> odom: %s", ex.what())
    }
  } else {       
    tf2::Transform mapBaseLinkTrans;
    tf2::fromMsg(odom_fused_for_tf.pose.pose, mapBaseLinkTrans);

    geometry_msgs::TransformStamped mapBaseLinkTransMsg;
    mapBaseLinkTransMsg.transform = tf2::toMsg(mapBaseLinkTrans);
    mapBaseLinkTransMsg.header.stamp = ts_for_tf;
    mapBaseLinkTransMsg.header.frame_id = "map";
    mapBaseLinkTransMsg.child_frame_id = "base_link";

    tf_broadcaster_.sendTransform(mapBaseLinkTransMsg);
  }
}