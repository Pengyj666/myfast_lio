#include "test/ctrl_line_test.h"

#include "common/math_utils.h"
#include "common/sysutils.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"

#include <cstdio>

using namespace utils;

ctrl_line_node::ctrl_line_node(ros::NodeHandle& n,ros::NodeHandle &m_param) 
    : nh_(n), n_params_(m_param){
  droslog(LogLevel::INFO, "TEST::ctor() ++++++");
  
  imu_sub_        = nh_.subscribe("/imu", 3, &ctrl_line_node::callback_imu, this);
  wheel_vel_sub_  = nh_.subscribe("/wheel_vel", 3, &ctrl_line_node::callback_wheel_vel, this);
  odom_fused_sub_    = nh_.subscribe("/odom_fused", 3, &ctrl_line_node::callback_odom_fused, this);

  cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 3);
  
  debug_srv_ = nh_.advertiseService("/test/debug_srv", &ctrl_line_node::srvDebug, this);
  droslog(LogLevel::INFO, "TEST::ctor() ------");
}

ctrl_line_node::~ctrl_line_node() {}
  
void ctrl_line_node::init() {
  droslog(LogLevel::INFO, "LOC::init() ++++++");
  droslog(LogLevel::INFO, "LOC::init() ------");
}

void ctrl_line_node::loop() {
  droslog(LogLevel::INFO, "TEST::loop() ++++++");
  ros::Rate loop_rate(50);
  while (ros::ok()) 
  {
    TrackLine();

    ros::spinOnce();
    loop_rate.sleep();
  }
  droslog(LogLevel::INFO, "TEST::loop() ------");
}

void ctrl_line_node::callback_imu(const sensor_msgs::Imu::ConstPtr msg) {
  common::Data_Imu imu_data;
  imu_data.imu.acc << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
  imu_data.imu.gyro << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
  imu_data.imu.quat = Eigen::Quaterniond(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
  imu_data.timestamp = msg->header.stamp.toSec();

  std::lock_guard<std::mutex> lock(cur_imu_mutex_);
  cur_imu_ = imu_data;
}

void ctrl_line_node::callback_wheel_vel(const geometry_msgs::TwistStamped::ConstPtr msg) {
  common::Data_WheelVel wheel_vel;
  wheel_vel.timestamp = msg->header.stamp.toSec();
  wheel_vel.vel.vel << msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z;
  
  std::lock_guard<std::mutex> lock(cur_wheel_vel_mutex_);
  cur_wheel_vel_ = wheel_vel;
}

void ctrl_line_node::callback_odom_fused(const nav_msgs::Odometry::ConstPtr msg) {
  common::Data_Pose pose;
  pose.timestamp = msg->header.stamp.toSec();
  pose.pose.pos << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
  pose.pose.quat = Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
  
  std::lock_guard<std::mutex> lock(cur_pose_mutex_);
  cur_pose_ = pose;
}

bool ctrl_line_node::srvDebug(mower_msgs::TriggerRequest &req, mower_msgs::TriggerResponse& rep) {
  const std::string req_type = req.arg;
  rep.result = 1;
  if (req_type == "stop") {
    rep.message = "ok";
    droslog(LogLevel::INFO, "ctrl_line_node::srvDebug() 收到指令: 停止移动");
    std::lock_guard<std::mutex> lock(line_mutex_);
    state_.store(0);
  } else if (req_type == "ang_line") {
    rep.message = "ok";
    double line_angle = 0.0;
    {
      std::lock_guard<std::mutex> lock(cur_pose_mutex_);
      line_angle = GetEulerRPY(cur_pose_.pose.quat)[2];
    }
    droslog(LogLevel::INFO, "ctrl_line_node::srvDebug() 收到指令: 开始姿态控制直行, 直行航向: %.3f", line_angle);
    std::lock_guard<std::mutex> lock(line_mutex_);
    state_.store(1);
    line_angle_ = line_angle;
  } else if (req_type == "line") {
    rep.message = "ok";
    double line_angle = 0.0;
    common::Pose pose_A;
    {
      std::lock_guard<std::mutex> lock(cur_pose_mutex_);
      pose_A = cur_pose_.pose;
      line_angle = GetEulerRPY(cur_pose_.pose.quat)[2];
    }
    common::Pose pose_B = pose_A;
    pose_B.pos[0] += 20.0 * std::cos(line_angle);
    pose_B.pos[1] += 20.0 * std::sin(line_angle);
    droslog(LogLevel::INFO, "ctrl_line_node::srvDebug() 收到指令: 开始位置+姿态控制直行: theta:%.3f, A(%.3f,%.3f), B(%.3f,%.3f)",
        line_angle, pose_A.pos[0], pose_A.pos[1], pose_B.pos[0], pose_B.pos[1]);
    std::lock_guard<std::mutex> lock(line_mutex_);
    state_.store(2);
    line_angle_ = line_angle;
    line_pose_A_ = pose_A;
    line_pose_B_ = pose_B;
  } else {
    rep.message = "unknown cmd";
    droslog(LogLevel::INFO, "ctrl_line_node::srvDebug() 收到无效指令: %s", req_type.c_str());
  }

  return true;
}

void ctrl_line_node::TrackLine() {
  static int cnt = 0;
  static int pre_cnt = 0;

  long long cur_ts = GetNow_Steady();
  if (state_.load() == 1) {
    cnt++;

    float cur_yaw, cur_roll;
    {
      std::lock_guard<std::mutex> lock(cur_pose_mutex_);
      auto rpy = GetEulerRPY(cur_pose_.pose.quat);
      cur_yaw = rpy[2];
      cur_roll = rpy[0];
    }
    float dyaw = KeepAngleInPI(line_angle_ - cur_yaw);
    float av = dyaw * 10;
    // float av = 0.f;
    // if (dyaw > 0.02) {
    //   av = (dyaw * 10) * (dyaw * 10);
    // } else if (dyaw < -0.02) {
    //   av = (dyaw * 10) * (dyaw * 10);
    // }
    if (std::abs(av) < 0.1) av = 0.0;
    if (av > 0.3) av = 0.3;
    if (av < -0.3) av = -0.3;
    std::printf("dyaw= %.3f, av=%.3f\n", dyaw, av);

    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = 0.4;
    cmd_vel.angular.z = av;

    cmd_vel_pub_.publish(cmd_vel);
    
    static long long pre_log_ts = 0;
    if (cur_ts > pre_log_ts + 1000) {
      pre_log_ts = cur_ts;
      auto rpy = GetEulerRPY(cur_pose_.pose.quat);
      droslog(LogLevel::INFO, "ctrl_line_node::TrackLine(1) 控制频率: %d, cur_rpy=%.3f,%.3f,%.3f, cmd_vel=%.3f,%.3f, imu_av=%.3f, wheel_lv=%.3f", 
          cnt - pre_cnt, rpy[0], rpy[1], rpy[2], cmd_vel.linear.x, cmd_vel.angular.z, cur_imu_.imu.gyro[2], cur_wheel_vel_.vel.vel[0]);
      pre_cnt = cnt;
    }
  } else if (state_.load() == 2) {
    cnt++;

    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = 0.4;
    cmd_vel.angular.z = 0.0;

    cmd_vel_pub_.publish(cmd_vel);
    
    static long long pre_log_ts = 0;
    if (cur_ts > pre_log_ts + 1000) {
      pre_log_ts = cur_ts;
      droslog(LogLevel::INFO, "ctrl_line_node::TrackLine(2) 控制频率: %d", cnt - pre_cnt);
      pre_cnt = cnt;
    }
  }

  return;
}