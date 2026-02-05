
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

namespace {
  std::atomic<double> vio_offset_ts(0.0);
}

void loc_node::callback_vio_HB(const std_msgs::Float64::ConstPtr &msg) {
  static geometry_msgs::Pose pre_pose;
  static double pre_HB_ts = 0.0;
  static double pre_ts = 0.0;
  double HB_ts = msg->data;
  double now_ts = ros::Time::now().toSec();
  if (pre_ts > 0.0) {
    // 发布vio重启事件
    if (HB_ts - pre_HB_ts + 5.0 < now_ts - pre_ts) {
      mower_msgs::ControllerEvent ce_msg;
      ce_msg.start_pose = pre_pose;
      ce_msg.end_pose = localization_info_msg_.fused_pose.pose;
      ce_msg.start_time = pre_ts;
      ce_msg.end_time = now_ts;
      ce_msg.event_type = mower_msgs::ControllerEvent::event_vio_restart;
      ce_msg.event_result = mower_msgs::ControllerEvent::event_success;

      ctl_event_pub_.publish(ce_msg);

      droslog(LogLevel::WARN, "LOC::callback_vio_HB() 监测到vio心跳异常, 可能重启了, HB_ts(%.3f->%.3f), pre_ts=%.3f, now_ts=%.3f",
          pre_HB_ts, HB_ts, pre_ts, now_ts);
    }
  }
  {
    static SimpleLogFilter log_filter(5000);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::INFO, "LOC::callback_vio_HB() HB_ts=%.3f, pre_HB_ts=%.3f, pre_ts=%.3f, now_ts=%.3f",
          HB_ts, pre_HB_ts, pre_ts, now_ts);
    }
  }
  pre_HB_ts = HB_ts;
  pre_ts = now_ts;
  pre_pose = localization_info_msg_.fused_pose.pose;
}

void loc_node::callback_vio(const nav_msgs::Odometry::ConstPtr &msg) {
  static double pre_msg_ts = 0.0;
  // double msg_ts = msg->header.stamp.toSec();
  double msg_ts = msg->pose.covariance[1] + msg->pose.covariance[3];
  
  if (msg_ts <= pre_msg_ts) {
    droslog(LogLevel::WARN, "LOC::callback_vio() 时间戳倒退, cur_ts=%.3f, pre_ts=%.3f", msg_ts, pre_msg_ts);
    return;
  }
  pre_msg_ts = msg_ts;

  common::Data_VioResult vio_result;
  vio_result.timestamp = msg->header.stamp.toSec();
  vio_result.confidence = 3;
  vio_result.vio.pos << msg->pose.pose.position.x,
                        msg->pose.pose.position.y,
                        msg->pose.pose.position.z;
  vio_result.vio.q.w() = msg->pose.pose.orientation.w,
  vio_result.vio.q.x() = msg->pose.pose.orientation.x,
  vio_result.vio.q.y() = msg->pose.pose.orientation.y,
  vio_result.vio.q.z() = msg->pose.pose.orientation.z;

  auto vio2gps_t = TFHelper::Instance()->Vio2Gps_t();
  auto gps2base_t = TFHelper::Instance()->Gps2Base_t();
  auto vio2gps_pose = TFHelper::Instance()->TF_Vio2Gps(vio_result.vio.pos, vio_result.vio.q);
  vio_result.vio.pos = vio2gps_pose.pos + vio2gps_t;
  vio_result.vio.q = vio2gps_pose.quat;

  vio_result.vio.linear << msg->twist.twist.linear.x,
                        msg->twist.twist.linear.y, 
                        msg->twist.twist.linear.z;
  vio_result.vio.angular << msg->twist.twist.angular.x,
                        msg->twist.twist.angular.y,
                        msg->twist.twist.angular.z;
  
  vio_offset_ts.store(msg->pose.covariance[3]);
  {
    double cur_vio_fid = msg->pose.covariance[0];
    if (cur_vio_fid + 3.0 < pre_vio_fid_.load()) {
      droslog(LogLevel::WARN, "LOC::callback_vio(), vio fid 倒退, 可能是vio重置了, pre_vio_fid: %.1f, cur_vio_fid: %.1f", pre_vio_fid_.load(), cur_vio_fid);
      if (locator_.GetWorkState() > 1 && pre_reset_ts_.load() + 15000 < GetNow_Steady()) {
        droslog(LogLevel::WARN, "LOC::callback_vio(), 重置VioTracker & VioGnssInitor");
        ProcVioReset();
      } else {
        droslog(LogLevel::WARN, "LOC::callback_vio(), 不需要重置VioTracker & VioGnssInitor");
      }
    }

    pre_vio_fid_.store(cur_vio_fid);
  }

  int work_mode = locator_.GetWorkMode();
  if (fusion_type_.load() == 0 || fusion_type_.load() == 1) {
    VioTracker::Instance()->FeedData(vio_result);

    if (0 == work_mode) {
      VioGnssInitor::Instance()->FeedVio(vio_result);
    }
  }

  VioReseter::Instance()->FeedData(vio_result);

  bool vio_valid = VioTracker::Instance()->IsVioValid();
  if (vio_result.confidence > 1 && vio_valid) {
    auto vio_base = VioTracker::Instance()->GetVioInLocalXyz(vio_result);
    if (vio_base.timestamp > 0.0) {
      static common::Data_VioResult pre_vio_base;
      if (pre_vio_base.timestamp > 0.0 && vio_base.timestamp < pre_vio_base.timestamp + 0.4 && (pre_vio_base.vio.pos - vio_base.vio.pos).norm() < 0.3) {
        auto rpy = GetEulerRPY(vio_base.vio.q);
        if (std::abs(rpy[0]) < 0.8 && std::abs(rpy[1]) < 0.8 && std::abs(rpy[0])+std::abs(rpy[1]) < 1.0) {
          auto gps2base_pose = TFHelper::Instance()->TF_Gps2Base(vio_base.vio.pos, vio_base.vio.q);
          vio_base.vio.pos = gps2base_pose.pos;
          vio_base.vio.q = gps2base_pose.quat;
          locator_.ProcessVioData(vio_base, 0);
        } else {
          droslog(LogLevel::WARN, "LOC::callback_vio() 对齐后的vio姿态异常: rpy=(%.3f, %.3f, %.3f)", rpy[0], rpy[1], rpy[2]);
        }
      } else {
        droslog(LogLevel::WARN, "LOC::callback_vio() vio 尚未平稳, pre_ts=%.3f, pre_vio.pos=%.3f, %.3f, %.3f, cur_vio.pos=%.3f, %.3f, %.3f",
            pre_vio_base.timestamp, pre_vio_base.vio.pos[0], pre_vio_base.vio.pos[1], pre_vio_base.vio.pos[2], vio_base.vio.pos[0], vio_base.vio.pos[1], vio_base.vio.pos[2]);
      }
      pre_vio_base = vio_base;
    } else {
      droslog(LogLevel::WARN, "LOC::callback_vio() GetVioInLocalXyz()返回异常");
    }
  }

  {
    static SimpleLogFilter log_filter(5000);
    if (log_filter.Output(GetNow_Steady())) {
      double roll11, pitch11, yaw11;
      tf::Matrix3x3(tf::Quaternion(vio_result.vio.q.x(), vio_result.vio.q.y(), 
          vio_result.vio.q.z(), vio_result.vio.q.w())).getRPY(roll11, pitch11, yaw11);
      
      droslog(LogLevel::INFO, "LOC::callback_vio() vio_valid=%d, ts=%.3f, vio_ts=%.3f, offset_ts=%.3f, fid=%.1f, confidence=%d, pos=(%.3f, %.3f, %.3f), rpy=(%.3f, %.3f, %.3f), lv=(%.3f, %.3f, %.3f), av=(%.3f, %.3f, %.3f)",
          vio_valid, vio_result.timestamp, msg->pose.covariance[1], msg->pose.covariance[3], msg->pose.covariance[0], vio_result.confidence, 
          vio_result.vio.pos(0), vio_result.vio.pos(1), vio_result.vio.pos(2),
          roll11, pitch11, yaw11,
          vio_result.vio.linear(0), vio_result.vio.linear(1), vio_result.vio.linear(2),
          vio_result.vio.angular(0), vio_result.vio.angular(1), vio_result.vio.angular(2));
    }
  }
}

void loc_node::callback_vreloc_pose(const nav_msgs::Odometry::ConstPtr &msg) {
  static double pre_msg_ts = 0.0;
  double msg_ts = msg->header.stamp.toSec();
  if (msg_ts <= pre_msg_ts) {
    droslog(LogLevel::WARN, "LOC::callback_vreloc_pose() 时间戳倒退, cur_ts=%.3f, pre_ts=%.3f", msg_ts, pre_msg_ts);
    return;
  }
  pre_msg_ts = msg_ts;

  common::Data_VioResult vreloc;
  vreloc.timestamp = msg->header.stamp.toSec();

  {
    static SimpleLogFilter log_filter(200);
    if (log_filter.Output(GetNow_Steady())) {
      Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
      auto rpy = GetEulerRPY(q);
      droslog(LogLevel::INFO, "LOC::callback_vreloc_pose() 收到视觉重定位结果 vreloc_pose: ts=%.3f, pos=%.3f, %.3f, %.3f, rpy=%.3f, %.3f, %.3f",  
          vreloc.timestamp, msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z, rpy(0), rpy(1), rpy(2));
    }
  }

  vreloc.confidence = 3;
  vreloc.vio.pos << msg->pose.pose.position.x,
                        msg->pose.pose.position.y,
                        msg->pose.pose.position.z;
  vreloc.vio.q.w() = msg->pose.pose.orientation.w,
  vreloc.vio.q.x() = msg->pose.pose.orientation.x,
  vreloc.vio.q.y() = msg->pose.pose.orientation.y,
  vreloc.vio.q.z() = msg->pose.pose.orientation.z;

  auto vreloc_gps = TFHelper::Instance()->TF_Vio2Gps(vreloc.vio.pos, vreloc.vio.q);
  vreloc.vio.pos = vreloc_gps.pos;
  vreloc.vio.q = vreloc_gps.quat;

  VioTracker::Instance()->FeedVreloc(vreloc);
}

void loc_node::callback_vmap_odom(const nav_msgs::Odometry::ConstPtr &msg) {
  static double pre_msg_ts = 0.0;
  double msg_ts = msg->header.stamp.toSec();
  if (msg_ts <= pre_msg_ts) {
    droslog(LogLevel::WARN, "LOC::callback_vmap_odom() 时间戳倒退, cur_ts=%.3f, pre_ts=%.3f", msg_ts, pre_msg_ts);
    return;
  }
  pre_msg_ts = msg_ts;

  common::Data_VioResult vmap_odom;
  vmap_odom.timestamp = msg->header.stamp.toSec() + vio_offset_ts.load();

  {
    static SimpleLogFilter log_filter(1000);
    if (log_filter.Output(GetNow_Steady())) {
      Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
      auto rpy = GetEulerRPY(q);
      droslog(LogLevel::INFO, "LOC::callback_vmap_odom() 收到视觉重定位跟踪结果 vmap_odom: ts=%.3f, pos=%.3f, %.3f, %.3f, rpy=%.3f, %.3f, %.3f",  
          vmap_odom.timestamp, msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z, rpy(0), rpy(1), rpy(2));
    }
  }

  vmap_odom.confidence = 3;
  vmap_odom.vio.pos << msg->pose.pose.position.x,
                        msg->pose.pose.position.y,
                        msg->pose.pose.position.z;
  vmap_odom.vio.q.w() = msg->pose.pose.orientation.w,
  vmap_odom.vio.q.x() = msg->pose.pose.orientation.x,
  vmap_odom.vio.q.y() = msg->pose.pose.orientation.y,
  vmap_odom.vio.q.z() = msg->pose.pose.orientation.z;

  auto vreloc_gps = TFHelper::Instance()->TF_Vio2Gps(vmap_odom.vio.pos, vmap_odom.vio.q);
  auto vreloc_base = TFHelper::Instance()->TF_Gps2Base(vreloc_gps.pos, vreloc_gps.quat);
  vmap_odom.vio.pos = vreloc_base.pos;
  vmap_odom.vio.q = vreloc_base.quat;

  std::lock_guard<std::mutex> lock(loc_mutex_);
  locator_.ProcessVioData(vmap_odom, 1);
}

void loc_node::callback_vmap_state(const std_msgs::String::ConstPtr &msg) {
  static SimpleLogFilter log_filter(5000);
  if (log_filter.Output(GetNow_Steady())) {
    droslog(LogLevel::INFO, "LOC::callback_vmap_state() vmap: %s", msg->data.c_str());
  }

  int state = -1;
  if (msg->data == "idl") {
    state = 0;
  } else if (msg->data == "map") {
    state = 1;
  } else if (msg->data == "loc") {
    state = 2;
  }

  VmapMonitor::Instance()->FeedVmapState(GetNow_Steady(), state);
}