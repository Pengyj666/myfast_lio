
#include "version.h"
#include "loc_node.h"

#include "common/common_def.h"
#include "common/data_type.h"
#include "common/sysutils.h"
#include "common/log_filters.h"
#include "common/debug_client.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"
#include "geo_utils/tf_helper.h"

using namespace utils;

namespace {
  std::atomic<double> lio_offset_ts(0.0);
}

void loc_node::callback_lio_offset_ts(const std_msgs::Float64::ConstPtr &msg) {
  {
    static SimpleLogFilter log_filter(5000);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::INFO, "LOC::callback_lio_offset_ts() dts=%.3f", msg->data);
    }
  }
  
  lio_offset_ts.store(msg->data);
}

void loc_node::callback_lio(const nav_msgs::Odometry::ConstPtr &msg) {
  static double pre_msg_ts = 0.0;
  double msg_ts = msg->header.stamp.toSec();
  if (msg_ts <= pre_msg_ts) {
    droslog(LogLevel::WARN, "LOC::callback_lio() 时间戳倒退, cur_ts=%.3f, pre_ts=%.3f", msg_ts, pre_msg_ts);
    return;
  }
  if (msg_ts > pre_msg_ts + 0.3 && pre_msg_ts > 0.0) {
    droslog(LogLevel::WARN, "LOC::callback_lio() 时间戳间隔过大, cur_ts=%.3f, pre_ts=%.3f", msg_ts, pre_msg_ts);
  }
  pre_msg_ts = msg_ts;

  common::Data_ProbPose lio, lio_gps;
  lio.timestamp = msg->header.stamp.toSec() + lio_offset_ts.load();
  lio.ppose.pos << msg->pose.pose.position.x,
                        msg->pose.pose.position.y,
                        msg->pose.pose.position.z;
  lio.ppose.quat.w() = msg->pose.pose.orientation.w,
  lio.ppose.quat.x() = msg->pose.pose.orientation.x,
  lio.ppose.quat.y() = msg->pose.pose.orientation.y,
  lio.ppose.quat.z() = msg->pose.pose.orientation.z;
  lio_gps = lio;

  // auto lidar2base_t = TFHelper::Instance()->Lidar2Base_t();
  // auto lidar2base_pose = TFHelper::Instance()->TF_Lidar2Base(lio.ppose.pos, lio.ppose.quat);

  // int work_mode = locator_.GetWorkMode();
  // if (work_mode == 0) {
  //   lio.ppose.pos = lidar2base_pose.pos + lidar2base_t;
  //   lio.ppose.quat = lidar2base_pose.quat;
  //   locator_.ProcessLioData(lio, 0);
  // }

  auto lidar2gps_t = TFHelper::Instance()->Lidar2Gps_t();
  auto gps2base_t = TFHelper::Instance()->Gps2Base_t();
  auto lidar2gps_pose = TFHelper::Instance()->TF_Lidar2Gps(lio.ppose.pos, lio.ppose.quat);
  lio_gps.ppose.pos = lidar2gps_pose.pos + lidar2gps_t;
  lio_gps.ppose.quat = lidar2gps_pose.quat;

  int work_mode = locator_.GetWorkMode();
  if (fusion_type_.load() == 2) {
    lidar_tracker_.FeedPose(lio_gps);

    if (0 == work_mode) {
      // TODO LIO-RTK 初始化部分
    }
  }

  bool loc_valid = lidar_tracker_.IsLocValid();
  if (loc_valid) {
    auto lio_base = lidar_tracker_.GetPoseInLocalXyz(lio_gps);
    auto lio_tf = lidar_tracker_.GetTF();

    if (lio_base.timestamp > 0.0) {
      {
        geometry_msgs::PoseWithCovarianceStamped lio_xyz_msg;
        lio_xyz_msg.header.stamp.fromSec(lio_base.timestamp);
        lio_xyz_msg.header.frame_id = "local_lio";
        lio_xyz_msg.pose.pose.position.x = lio_base.ppose.pos.x();
        lio_xyz_msg.pose.pose.position.y = lio_base.ppose.pos.y();
        lio_xyz_msg.pose.pose.position.z = lio_base.ppose.pos.z();
        lio_xyz_msg.pose.pose.orientation.x = lio_base.ppose.quat.x();
        lio_xyz_msg.pose.pose.orientation.y = lio_base.ppose.quat.y();
        lio_xyz_msg.pose.pose.orientation.z = lio_base.ppose.quat.z();
        lio_xyz_msg.pose.pose.orientation.w = lio_base.ppose.quat.w();
        
        local_lio_pub_.publish(lio_xyz_msg);
      }

      static common::Data_ProbPose pre_lio_base;
      if (pre_lio_base.timestamp > 0.0 && lio_base.timestamp < pre_lio_base.timestamp + 2.0 && (pre_lio_base.ppose.pos - lio_base.ppose.pos).norm() < 0.4) {
        auto rpy = GetEulerRPY(lio_base.ppose.quat);
        auto lio_rpy = GetEulerRPY(lio.ppose.quat);
        auto lio_gps_rpy = GetEulerRPY(lio_gps.ppose.quat);
        auto lio_tf_rpy = GetEulerRPY(lio_tf.data.quat);
        if (std::abs(rpy[0]) < 0.8 && std::abs(rpy[1]) < 0.8 && std::abs(rpy[0])+std::abs(rpy[1]) < 1.0) {
          auto gps2base_pose = TFHelper::Instance()->TF_Gps2Base(lio_base.ppose.pos, lio_base.ppose.quat);
          lio_base.ppose.pos = gps2base_pose.pos;
          lio_base.ppose.quat = gps2base_pose.quat;
          locator_.ProcessLioData(lio_base, 0);
          droslog(LogLevel::INFO, "LOC::callback_lio() ts=%.3f, offset_ts=%.3f, 原始lio(%.3f,%.3f,%.3f;%.3f,%.3f,%.3f), lio.base(%.3f,%.3f,%.3f;%.3f,%.3f,%.3f), lio.tf(%.3f,%.3f,%.3f;%.3f,%.3f,%.3f)", 
              lio_base.timestamp, lio_offset_ts.load(), lio.ppose.pos[0], lio.ppose.pos[1], lio.ppose.pos[2], lio_rpy[0], lio_rpy[1], lio_rpy[2],
              lio_base.ppose.pos[0], lio_base.ppose.pos[1], lio_base.ppose.pos[2], rpy[0], rpy[1], rpy[2],
              lio_tf.data.pos[0], lio_tf.data.pos[1], lio_tf.data.pos[2], lio_tf_rpy[0], lio_tf_rpy[1], lio_tf_rpy[2]);
        } else {
          droslog(LogLevel::WARN, "LOC::callback_lio() 对齐后的lio姿态异常: rpy=(%.3f, %.3f, %.3f)", rpy[0], rpy[1], rpy[2]);
        }
      } else {
        droslog(LogLevel::WARN, "LOC::callback_lio() lio 尚未平稳, pre_ts=%.3f, pre_lio.pos=%.3f, %.3f, %.3f, cur_lio.pos=%.3f, %.3f, %.3f",
            pre_lio_base.timestamp, pre_lio_base.ppose.pos[0], pre_lio_base.ppose.pos[1], pre_lio_base.ppose.pos[2], lio_base.ppose.pos[0], lio_base.ppose.pos[1], lio_base.ppose.pos[2]);
      }
      pre_lio_base = lio_base;
    } else {
      droslog(LogLevel::WARN, "LOC::callback_lio() GetPoseInLocalXyz()返回异常");
    }
  }

  {
    static SimpleLogFilter log_filter(5000);
    if (log_filter.Output(GetNow_Steady())) {
      Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
      auto rpy = GetEulerRPY(q);
      droslog(LogLevel::INFO, "LOC::callback_lio() lio: ts=%.3f, pos=%.3f, %.3f, %.3f, rpy=%.3f, %.3f, %.3f",  
          lio.timestamp, msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z, rpy(0), rpy(1), rpy(2));
    }
  }
}

void loc_node::callback_lio_reloc_result(const nav_msgs::Odometry::ConstPtr &msg) {
  static double pre_msg_ts = 0.0;
  double msg_ts = msg->header.stamp.toSec();
  if (msg_ts <= pre_msg_ts) {
    droslog(LogLevel::WARN, "LOC::callback_lio_reloc_result() 时间戳倒退, cur_ts=%.3f, pre_ts=%.3f", msg_ts, pre_msg_ts);
    return;
  }
  pre_msg_ts = msg_ts;

  common::Data_ProbPose reloc_lio;
  reloc_lio.timestamp = msg_ts + lio_offset_ts.load();

  {
    static SimpleLogFilter log_filter(1000);
    if (log_filter.Output(GetNow_Steady())) {
      Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
      auto rpy = GetEulerRPY(q);
      droslog(LogLevel::INFO, "LOC::callback_lio_reloc_result() 收到激光重定位结果 reloc_result: ts=%.3f, pos=%.3f, %.3f, %.3f, rpy=%.3f, %.3f, %.3f",
          reloc_lio.timestamp, msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z, rpy(0), rpy(1), rpy(2));
    }
  }

  reloc_lio.ppose.pos << msg->pose.pose.position.x,
                        msg->pose.pose.position.y,
                        msg->pose.pose.position.z;
  reloc_lio.ppose.quat.w() = msg->pose.pose.orientation.w,
  reloc_lio.ppose.quat.x() = msg->pose.pose.orientation.x,
  reloc_lio.ppose.quat.y() = msg->pose.pose.orientation.y,
  reloc_lio.ppose.quat.z() = msg->pose.pose.orientation.z;

  auto lidar2gps_t = TFHelper::Instance()->Lidar2Gps_t();
  auto lreloc_gps = TFHelper::Instance()->TF_Lidar2Gps(reloc_lio.ppose.pos, reloc_lio.ppose.quat);
  reloc_lio.ppose.pos = lreloc_gps.pos + lidar2gps_t;
  reloc_lio.ppose.quat = lreloc_gps.quat;

  lidar_tracker_.FeedReloc(reloc_lio);
}

void loc_node::callback_lio_reloc(const nav_msgs::Odometry::ConstPtr &msg) {
  static double pre_msg_ts = 0.0;
  double msg_ts = msg->header.stamp.toSec();
  if (msg_ts <= pre_msg_ts) {
    droslog(LogLevel::WARN, "LOC::callback_lio_reloc() 时间戳倒退, cur_ts=%.3f, pre_ts=%.3f", msg_ts, pre_msg_ts);
    return;
  }
  pre_msg_ts = msg_ts;

  // double sys_ts = ros::Time::now().toSec();
  // if (lio_offset_ts.load() > 0.0) {
  //   lio_offset_ts.store(lio_offset_ts.load() * 0.9 + (sys_ts - msg_ts) * 0.1 - 0.2);
  // } else {
  //   lio_offset_ts.store(sys_ts - msg_ts - 0.2);
  // }

  common::Data_ProbPose lreloc;
  lreloc.timestamp = msg_ts + lio_offset_ts.load();

  lreloc.ppose.pos << msg->pose.pose.position.x,
                      msg->pose.pose.position.y,
                      msg->pose.pose.position.z;
  lreloc.ppose.quat.w() = msg->pose.pose.orientation.w,
  lreloc.ppose.quat.x() = msg->pose.pose.orientation.x,
  lreloc.ppose.quat.y() = msg->pose.pose.orientation.y,
  lreloc.ppose.quat.z() = msg->pose.pose.orientation.z;

  auto lidar2base_t = TFHelper::Instance()->Lidar2Base_t();
  auto lreloc_base = TFHelper::Instance()->TF_Lidar2Base(lreloc.ppose.pos, lreloc.ppose.quat);
  lreloc.ppose.pos = lreloc_base.pos + lidar2base_t;
  lreloc.ppose.quat = lreloc_base.quat;

  {
    static SimpleLogFilter log_filter(1000);
    if (log_filter.Output(GetNow_Steady())) {
      Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
      auto rpy = GetEulerRPY(q);
      auto rpy2 = GetEulerRPY(lreloc.ppose.quat);
      droslog(LogLevel::INFO, "LOC::callback_lio_reloc() 收到reloc_lio结果 lreloc: ts=%.3f, pose=(%.3f, %.3f, %.3f;%.3f, %.3f, %.3f), pose_base=(%.3f, %.3f, %.3f;%.3f, %.3f, %.3f)",
          lreloc.timestamp, msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z, rpy(0), rpy(1), rpy(2),
          lreloc.ppose.pos[0], lreloc.ppose.pos[1], lreloc.ppose.pos[2], rpy2[0], rpy2[1], rpy2[2]);
    }
  }
  return;

  // std::lock_guard<std::mutex> lock(loc_mutex_);
  // locator_.ProcessLioData(lreloc, 1);
}

void loc_node::callback_lmap_state(const std_msgs::String::ConstPtr &msg) {
  static SimpleLogFilter log_filter(5000);
  if (log_filter.Output(GetNow_Steady())) {
    droslog(LogLevel::INFO, "LOC::callback_lmap_state() vmap状态: %s", msg->data.c_str());
  }
}