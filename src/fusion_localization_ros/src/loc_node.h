#ifndef LOC_NODE_H
#define LOC_NODE_H

#include <atomic>
#include <stdint.h>

#include <ros/ros.h>

#include <std_msgs/Float64.h>
#include <std_msgs/String.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>

#include "mower_gps_msgs/UnicoreNav.h"
#include "mower_msgs/MowerLocalizationInfo.h"
#include "mower_msgs/MowerSensorInfo.h"
#include "mower_msgs/IotNotice.h"
#include "mower_msgs/Trigger.h"
#include "mower_msgs/EskfState.h"
#include "mower_msgs/LocatorGetMapInfo.h"
#include "mower_msgs/LocatorLoadMap.h"
#include "mower_msgs/LocatorSaveMap.h"
#include "mower_msgs/ControllerEvent.h"

#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/impl/utils.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <GeographicLib/LocalCartesian.hpp>

#include "locator/simple_vio_locator.h"
#include "common/loc_tracker.h"

enum LocState : uint8_t {
  INIT = mower_msgs::MowerLocalizationInfo::STATE_INIT,
  RTK_VISION_FUSION = mower_msgs::MowerLocalizationInfo::STATE_RTK_VISION,
  VISION_ONLY = mower_msgs::MowerLocalizationInfo::STATE_VISION_ONLY,
  LOST = mower_msgs::MowerLocalizationInfo::STATE_LOST,
};

enum ErrorCode : uint8_t {
  ERROR_NONE = mower_msgs::MowerLocalizationInfo::ERROR_NONE,
  ERROR_POSITION_INIT_FAILED = mower_msgs::MowerLocalizationInfo::ERROR_POSITION_INIT_FAILED,
  ERROR_HEADING_INIT_FAILED = mower_msgs::MowerLocalizationInfo::ERROR_HEADING_INIT_FAILED,
  ERROR_ALIGN_RTK_FAILED = mower_msgs::MowerLocalizationInfo::ERROR_ALIGN_RTK_FAILED,
  ERROR_RTK_VISION_LOST = mower_msgs::MowerLocalizationInfo::ERROR_RTK_VISION_LOST,
};

class loc_node {
 public:
  loc_node(ros::NodeHandle& n,ros::NodeHandle &m_param);
	virtual ~loc_node();
    
	ros::NodeHandle nh_, n_params_;

  void init();
  void loop();

  void MonitorThread();

 public:
  // 基础数据订阅
  void callback_imu(const sensor_msgs::Imu::ConstPtr msg);
  void callback_wheel_vel(const geometry_msgs::TwistStamped::ConstPtr msg);
  void callback_sensor_info(const mower_msgs::MowerSensorInfo::ConstPtr &msg);
  // RTK相关
  void callback_unicore_nav(const mower_gps_msgs::UnicoreNav::ConstPtr &msg);
  void callback_rtk_ref(const sensor_msgs::NavSatFix::ConstPtr &msg);
  // 视觉相关
  void callback_vio_HB(const std_msgs::Float64::ConstPtr &msg);
  void callback_vio(const nav_msgs::Odometry::ConstPtr &msg);
  void callback_vreloc_pose(const nav_msgs::Odometry::ConstPtr &msg);
  void callback_vmap_odom(const nav_msgs::Odometry::ConstPtr &msg);
  void callback_vmap_state(const std_msgs::String::ConstPtr &msg);
  // 激光雷达相关
  void callback_lio_offset_ts(const std_msgs::Float64::ConstPtr &msg);
  void callback_lio(const nav_msgs::Odometry::ConstPtr &msg);
  void callback_lio_reloc_result(const nav_msgs::Odometry::ConstPtr &msg);
  void callback_lio_reloc(const nav_msgs::Odometry::ConstPtr &msg);
  void callback_lmap_state(const std_msgs::String::ConstPtr &msg);

  // 控制相关
  void callback_cmd_vel(const geometry_msgs::Twist::ConstPtr &msg);

  // 设置地图原点经纬坐标以及该地图建图时基站的参考坐标
  bool srvCheckHeading(mower_msgs::TriggerRequest &req, mower_msgs::TriggerResponse& rep);
  bool srvComputeHeading(mower_msgs::TriggerRequest &req, mower_msgs::TriggerResponse& rep);
  // 通用服务接口，用于设置状态
  bool srvSetState(mower_msgs::TriggerRequest &req, mower_msgs::TriggerResponse& rep);
  
  bool srvLoadMap(mower_msgs::LocatorLoadMap::Request &req, mower_msgs::LocatorLoadMap::Response &res);
  bool srvSaveMap(mower_msgs::LocatorSaveMap::Request &req, mower_msgs::LocatorSaveMap::Response &res);
  bool srvGetMapInfo(mower_msgs::LocatorGetMapInfo::Request &req, mower_msgs::LocatorGetMapInfo::Response &res);
  
  // 调试接口
  bool srvDebug(mower_msgs::TriggerRequest &req, mower_msgs::TriggerResponse& rep);

 private:  // 状态主逻辑
  void PubDebugInfo();
  void PubLocalizationInfo();

  void ProcVioReset();
  void ProcLioReset();

 private:
  // 基础数据订阅
  ros::Subscriber imu_sub_;
  ros::Subscriber wheel_vel_sub_;
  ros::Subscriber sensor_info_sub_;

  ros::Subscriber unicore_nav_sub_;
  ros::Subscriber rtk_ref_sub_;

  ros::Subscriber vio_sub_;
  ros::Subscriber vio_HB_sub_;
  ros::Subscriber vreloc_sub_;
  ros::Subscriber vmap_odom_sub_;
  ros::Subscriber vmap_state_sub_;

  ros::Subscriber lio_offset_ts_sub_;
  ros::Subscriber lio_sub_;
  ros::Subscriber lio_reloc_sub_;
  ros::Subscriber lio_reloc_result_sub_;
  ros::Subscriber lmap_state_sub_;

  ros::Subscriber cmd_vel_sub_;

  // 核心数据发布
  ros::Publisher fused_gps_pub_;
  ros::Publisher fused_odom_pub_;
  ros::Publisher fused_heading_pub_;
  ros::Publisher localization_info_pub_;

  ros::Publisher local_gps_pub_;      // gps转到局部坐标系xyz结果, base
  ros::Publisher local_vio_pub_;      // vio转到局部坐标系xyz结果, base
  ros::Publisher local_lio_pub_;      // lio转到局部坐标系xyz结果, base
  ros::Publisher vio_gnss_pub_;       // vio转gps结果, gps天线中心
  ros::Publisher iot_notice_pub_;
  ros::Publisher ctl_event_pub_;
  
  // 核心服务
  ros::ServiceServer compute_heading_srv_;
  ros::ServiceServer heading_check_srv_;
  ros::ServiceServer set_state_srv_;
  
  ros::ServiceServer load_map_srv_;
  ros::ServiceServer save_map_srv_;
  ros::ServiceServer get_map_info_srv_;
  
  ros::ServiceClient vio_reset_clt_;
  ros::ServiceClient save_vmap_clt_;
  ros::ServiceClient load_vmap_clt_;
  ros::ServiceClient vmap_ctrl_clt_;

  ros::ServiceClient lio_ctrl_clt_;
  ros::ServiceClient lio_savemap_clt_;
  ros::ServiceClient lio_loadmap_clt_;

  // 调试相关
  ros::ServiceServer debug_srv_;
  ros::Publisher fusion_state_pub_;
  ros::Publisher vio_gnss_align_pub_;
  ros::Publisher lio_gnss_align_pub_;

  // 导航地图到里程计坐标TF: map -> odom
  geometry_msgs::TransformStamped map_odom_tf_;
  // 导航里程计到车体中心TF: odom -> base_link
  geometry_msgs::TransformStamped odom_base_tf_;

  // TF发布器
  tf::TransformBroadcaster tf_broadcaster_utm_map_;
  tf::TransformBroadcaster tf_broadcaster_map_odom_;
  tf::TransformBroadcaster tf_broadcaster_odom_baselink_;
  tf::TransformBroadcaster tf_broadcaster_odom_locallink_;

  tf::TransformBroadcaster tf_broadcaster_ins_;

  // odom -> odom vio
  tf::TransformBroadcaster tf_odom_2_odomvio_;
  // odomvio-> vio_base_link
  tf::TransformBroadcaster tf_odomvio_2_vio_baselink_;
  // vio_base_link -> base_link
  tf::TransformBroadcaster tf_vio_baselink_2_baselink_;
  tf::TransformBroadcaster tf_broadcaster_vio_odom2_baselink_;

  tf::TransformListener listener;

  tf2_ros::TransformBroadcaster tf_broadcaster_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  
 private:
  mower_msgs::MowerLocalizationInfo localization_info_msg_;

  std::mutex loc_mutex_;
  SimpleVioLocator locator_;
  SimpleVioLocator::Config loc_config_;  

  utils::LocTracker lidar_tracker_;

  common::Data_Gnss last_gnss_data_;

  std::atomic<int> fusion_type_cfg_;
  std::atomic<int> fusion_type_;    // 0: RTK+VSLAM, 1: VISION, 2: LIDAR, 3: LIDAR+RTK

  std::atomic<double> pre_vio_fid_;
  std::atomic<long long> pre_reset_ts_;

 private:  // JOHN_NOTE 后面这些参数分到ConfigServer中管理
  std::string base_frame_id_;
  std::string global_frame_id_;
  std::string odom_frame_id_;

  std::atomic_bool use_vmap_cfg_;   // 配置参数值
  std::atomic_bool use_vmap_;       // 是否使用视觉地图与视觉重定位
  std::string map_name_;            // 地图名

  double track_off_rtk_dist_ = 300.0;   // rtk
  double track_off_reloc_dist_ = 60.0;  // 在桩, vio/lio-reloc 
  double track_only_iw_dist_ = 20.0;    // iw跟踪(vio/lio 里程计)

  double init_off_rtk_dist_ = 50.0;
  double init_off_reloc_dist_ = 15.0;
  double init_only_iw_dist_ = 10.0;
};

#endif  /* LOC_NODE_H */
