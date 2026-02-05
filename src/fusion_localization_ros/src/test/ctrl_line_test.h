#ifndef CTRL_LINE_TEST_H
#define CTRL_LINE_TEST_H

#include <atomic>
#include <stdint.h>

#include <ros/ros.h>

#include <geometry_msgs/TwistStamped.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include "mower_msgs/Trigger.h"

#include "common/data_type.h"
#include "common/timed_queue.h"

class ctrl_line_node {
 public:
  ctrl_line_node(ros::NodeHandle& n,ros::NodeHandle &m_param);
	virtual ~ctrl_line_node();
    
	ros::NodeHandle nh_, n_params_;

  void init();
  void loop();

 public:
  void callback_imu(const sensor_msgs::Imu::ConstPtr msg);
  void callback_wheel_vel(const geometry_msgs::TwistStamped::ConstPtr msg);
  void callback_odom_fused(const nav_msgs::Odometry::ConstPtr msg);

  bool srvDebug(mower_msgs::TriggerRequest &req, mower_msgs::TriggerResponse& rep);

 private:
  void TrackLine();

  // 基础数据订阅
  ros::Subscriber imu_sub_;
  ros::Subscriber wheel_vel_sub_;
  ros::Subscriber odom_fused_sub_;

  ros::ServiceServer debug_srv_;
  ros::Publisher cmd_vel_pub_;

  std::mutex cur_pose_mutex_, cur_imu_mutex_, cur_wheel_vel_mutex_;
  common::Data_Pose cur_pose_;
  common::Data_Imu cur_imu_;
  common::Data_WheelVel cur_wheel_vel_;

  std::mutex line_mutex_;
  std::atomic<int> state_;   // 0: 停止, 1: 航向角跟线, 2: 位置航向角跟线
  double line_angle_; // 跟线航向角
  common::Pose line_pose_A_, line_pose_B_;  // 跟线起点和终点
};

#endif  /* CTRL_LINE_TEST_H */
