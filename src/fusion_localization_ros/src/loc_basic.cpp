
#include "version.h"
#include "loc_node.h"

#include "common/sysutils.h"
#include "droslog/log.h"
#include "geo_utils/tf_helper.h"
#include "common/heading_estimator.h"
#include "common/vio_tracker.h"

using namespace utils;
using namespace common;

loc_node::loc_node(ros::NodeHandle& n,ros::NodeHandle &m_param):
    nh_(n),
    n_params_(m_param),
    tf_listener_(tf_buffer_)
{  
  droslog(LogLevel::INFO, "LOC::ctor() ++++++");

  pre_vio_fid_.store(-1.0);

  std::string vio_topic_name;
  n_params_.param<std::string>("vio_topic", vio_topic_name, "/as_vio/vio_pose_result");
  std::string vio_reset_srv_name;
  n_params_.param<std::string>("vio_reset_srv", vio_reset_srv_name, "/as_vio/ctrl");

  imu_sub_        = nh_.subscribe("/imu", 10, &loc_node::callback_imu, this);
  wheel_vel_sub_  = nh_.subscribe("/wheel_vel", 10, &loc_node::callback_wheel_vel, this);
  sensor_info_sub_ = nh_.subscribe("/mower_sensor_info", 10, &loc_node::callback_sensor_info, this);

  unicore_nav_sub_= nh_.subscribe("/unicore_nav", 10, &loc_node::callback_unicore_nav, this);
  rtk_ref_sub_ = nh_.subscribe("/ref", 1, &loc_node::callback_rtk_ref, this);

  vio_sub_ = nh_.subscribe(vio_topic_name, 10, &loc_node::callback_vio, this);
  vio_HB_sub_ = nh_.subscribe("/as_vio/heartbeat", 2, &loc_node::callback_vio_HB, this);
  vreloc_sub_ = nh_.subscribe("/as_vmap/reloc_result", 10, &loc_node::callback_vreloc_pose, this);
  vmap_odom_sub_ = nh_.subscribe("/as_vmap/reloc_pose", 10, &loc_node::callback_vmap_odom, this);
  vmap_state_sub_ = nh_.subscribe("/as_vmap/vmap_state", 10, &loc_node::callback_vmap_state, this);

  lio_offset_ts_sub_ = nh_.subscribe("/as_lio/offset_ts", 10, &loc_node::callback_lio_offset_ts, this);
  lio_sub_ = nh_.subscribe("/as_lio/lio", 10, &loc_node::callback_lio, this);
  lio_reloc_sub_ = nh_.subscribe("/as_lio/reloc_lio", 10, &loc_node::callback_lio_reloc, this);
  lio_reloc_result_sub_ = nh_.subscribe("/as_lio/reloc_result", 10, &loc_node::callback_lio_reloc_result, this);
  lmap_state_sub_ = nh_.subscribe("/as_lio/lmap_state", 10, &loc_node::callback_lmap_state, this);

  cmd_vel_sub_    = nh_.subscribe("/cmd_vel", 10, &loc_node::callback_cmd_vel, this);
  
  // 设置局部导航坐标系在经纬坐标系中的坐标, 用于将GNSS转导航坐标系
  heading_check_srv_ = nh_.advertiseService("/localization/CheckHeading", &loc_node::srvCheckHeading, this);
  compute_heading_srv_ = nh_.advertiseService("/localization/ComputeHeading", &loc_node::srvComputeHeading, this);
  set_state_srv_ = nh_.advertiseService("/localization/SetState", &loc_node::srvSetState, this);

  load_map_srv_ = nh_.advertiseService("/localization/LoadMap", &loc_node::srvLoadMap, this);
  save_map_srv_ = nh_.advertiseService("/localization/SaveMap", &loc_node::srvSaveMap, this);
  get_map_info_srv_ = nh_.advertiseService("/localization/GetMapInfo", &loc_node::srvGetMapInfo, this);
  
  // 视觉相关服务
  vio_reset_clt_ = nh_.serviceClient<mower_msgs::Trigger>(vio_reset_srv_name);
  save_vmap_clt_ = nh_.serviceClient<mower_msgs::Trigger>("/as_vmap/savemap");
  load_vmap_clt_ = nh_.serviceClient<mower_msgs::Trigger>("/as_vmap/loadmap");
  vmap_ctrl_clt_ = nh_.serviceClient<mower_msgs::Trigger>("/as_vmap/ctrl");

  // 激光相关服务
  lio_ctrl_clt_ = nh_.serviceClient<mower_msgs::Trigger>("/as_lio/ctrl");
  lio_savemap_clt_ = nh_.serviceClient<mower_msgs::Trigger>("/as_lio/savemap");
  lio_loadmap_clt_ = nh_.serviceClient<mower_msgs::Trigger>("/as_lio/loadmap");
  
  // 融合结果发布
  fused_gps_pub_  = nh_.advertise<sensor_msgs::NavSatFix>("/fix_fused", 1);
  fused_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/odom_fused", 1);
  fused_heading_pub_ = nh_.advertise<std_msgs::Float64>("/heading_fused", 1); 
  fusion_state_pub_ = nh_.advertise<mower_msgs::EskfState>("/eskf_fusion_state", 1);

  local_gps_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("/gps_local_xyz", 1);
  local_vio_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("/loc/local_vio_xyz", 1);
  local_lio_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("/loc/local_lio_xyz", 1);
  vio_gnss_pub_ = nh_.advertise<sensor_msgs::NavSatFix>("/loc/vio_gnss", 1);
  
  iot_notice_pub_ = nh_.advertise<mower_msgs::IotNotice>("/notice_code", 1);
  ctl_event_pub_ = nh_.advertise<mower_msgs::ControllerEvent>("/controller/event", 1);

  // 调试相关
  vio_gnss_align_pub_ = nh_.advertise<nav_msgs::Odometry>("/loc/vio_gnss_align", 1);
  lio_gnss_align_pub_ = nh_.advertise<nav_msgs::Odometry>("/loc/lio_gnss_align", 1);

  // 用来与逻辑层通信同步各种状态
  localization_info_pub_ = nh_.advertise<mower_msgs::MowerLocalizationInfo>("/mower_localization_info", 1);

  // 调试信息接口
  debug_srv_ = nh_.advertiseService("/fusion/debug_srv", &loc_node::srvDebug, this);

  // TF
  std::cout.precision(18); 

  droslog(LogLevel::INFO, "LOC::ctor() ------");
}

loc_node::~loc_node() {} 

void loc_node::init() { 
  droslog(LogLevel::INFO, "LOC::init() ++++++");

  // frame id
  n_params_.param<std::string>("map_frame", global_frame_id_, "map");
  n_params_.param<std::string>("odom", odom_frame_id_, "odom");
  n_params_.param<std::string>("base_frame", base_frame_id_, "base_link");

  double acc_noise;
  n_params_.param<double>("acc_noise", acc_noise,1e-2);
  double gyro_noise;
  n_params_.param<double>("gyro_noise", gyro_noise,1e-4);
  double acc_bias_noise;
  n_params_.param<double>("acc_bias_noise", acc_bias_noise,1e-6);
  double gyro_bias_noise;
  n_params_.param<double>("gyro_bias_noise", gyro_bias_noise,1e-8);

  loc_config_.acc_noise = acc_noise;
  loc_config_.gyro_noise = gyro_noise;
  loc_config_.acc_bias_noise = acc_bias_noise;
  loc_config_.gyro_bias_noise = gyro_bias_noise;

  double TF_vio2gps_x, TF_vio2gps_y, TF_vio2gps_z;
  n_params_.param<double>("TF_vio2gps_x", TF_vio2gps_x, 0.353);
  n_params_.param<double>("TF_vio2gps_y", TF_vio2gps_y, -0.041);
  n_params_.param<double>("TF_vio2gps_z", TF_vio2gps_z, 0.0);
  double TF_lidar2gps_x, TF_lidar2gps_y, TF_lidar2gps_z;
  n_params_.param<double>("TF_lidar2gps_x", TF_lidar2gps_x, 0.332);
  n_params_.param<double>("TF_lidar2gps_y", TF_lidar2gps_y, 0.001);
  n_params_.param<double>("TF_lidar2gps_z", TF_lidar2gps_z, 0.001);
  double TF_imu2gps_x, TF_imu2gps_y, TF_imu2gps_z;
  n_params_.param<double>("TF_imu2gps_x", TF_imu2gps_x, 0.0);
  n_params_.param<double>("TF_imu2gps_y", TF_imu2gps_y, 0.0);
  n_params_.param<double>("TF_imu2gps_z", TF_imu2gps_z, 0.0);
  double TF_gps2base_x, TF_gps2base_y, TF_gps2base_z;
  n_params_.param<double>("TF_gps2base_x", TF_gps2base_x, 0.098);
  n_params_.param<double>("TF_gps2base_y", TF_gps2base_y, 0.0);
  n_params_.param<double>("TF_gps2base_z", TF_gps2base_z, 0.0);
  double TF_lidar2base_x, TF_lidar2base_y, TF_lidar2base_z;
  n_params_.param<double>("TF_lidar2base_x", TF_lidar2base_x, 0.429);
  n_params_.param<double>("TF_lidar2base_y", TF_lidar2base_y, 0.0);
  n_params_.param<double>("TF_lidar2base_z", TF_lidar2base_z, 0.0);
  TFHelper::Instance()->SetParams_Vio2Gps(TF_vio2gps_x, TF_vio2gps_y, TF_vio2gps_z, 0.0, 0.0, 0.0);
  TFHelper::Instance()->SetParams_Lidar2Gps(TF_lidar2gps_x, TF_lidar2gps_y, TF_lidar2gps_z, 0.0, 0.0, 0.0);
  TFHelper::Instance()->SetParams_Imu2Gps(TF_imu2gps_x, TF_imu2gps_y, TF_imu2gps_z, 0.0, 0.0, 0.0);
  TFHelper::Instance()->SetParams_Gps2Base(TF_gps2base_x, TF_gps2base_y, TF_gps2base_z, 0.0, 0.0, 0.0);
  TFHelper::Instance()->SetParams_Lidar2Base(TF_lidar2base_x, TF_lidar2base_y, TF_lidar2base_z, 0.0, 0.0, 0.0);
  droslog(LogLevel::INFO, "LOC::init(): 参数: TF_vio2gps_x: %.3f, TF_vio2gps_y: %.3f, TF_vio2gps_z: %.3f", TF_vio2gps_x, TF_vio2gps_y, TF_vio2gps_z);
  droslog(LogLevel::INFO, "LOC::init(): 参数: TF_lidar2gps_x: %.3f, TF_lidar2gps_y: %.3f, TF_lidar2gps_z: %.3f", TF_lidar2gps_x, TF_lidar2gps_y, TF_lidar2gps_z);
  droslog(LogLevel::INFO, "LOC::init(): 参数: TF_imu2gps_x: %.3f, TF_imu2gps_y: %.3f, TF_imu2gps_z: %.3f", TF_imu2gps_x, TF_imu2gps_y, TF_imu2gps_z);
  droslog(LogLevel::INFO, "LOC::init(): 参数: TF_gps2base_x: %.3f, TF_gps2base_y: %.3f, TF_gps2base_z: %.3f", TF_gps2base_x, TF_gps2base_y, TF_gps2base_z);
  droslog(LogLevel::INFO, "LOC::init(): 参数: TF_lidar2base_x: %.3f, TF_lidar2base_y: %.3f, TF_lidar2base_z: %.3f", TF_lidar2base_x, TF_lidar2base_y, TF_lidar2base_z);
  
  loc_config_.imu_to_gps = Eigen::Vector3d(TF_imu2gps_x, TF_imu2gps_y, 0.0);

  double imu_frequency;
  n_params_.param<double>("imu_frequency", imu_frequency,100.0);
  double imu_cutoff_frequency;
  n_params_.param<double>("imu_cutoff_frequency", imu_cutoff_frequency,12.0);

  loc_config_.imu_freq = imu_frequency;
  loc_config_.imu_cutoff_freq = imu_cutoff_frequency;

  // 设置融合器参数
  locator_.Reset();
  locator_.SetConfig(loc_config_);

  double line_heading_KF_dist;
  int line_heading_KF_window;
  double line_heading_dist_min, line_heading_dist_max;
  double line_heading_imu_diff_yaw, line_heading_imu_sigma_yaw, line_heading_imu_sum_yaw;
  n_params_.param<double>("line_heading_KF_dist", line_heading_KF_dist, 0.05);
  n_params_.param<int>("line_heading_KF_window", line_heading_KF_window, 10);
  n_params_.param<double>("line_heading_dist_min", line_heading_dist_min, 0.3);
  n_params_.param<double>("line_heading_dist_max", line_heading_dist_max, 0.8);
  n_params_.param<double>("line_heading_imu_diff_yaw", line_heading_imu_diff_yaw, 0.17);
  n_params_.param<double>("line_heading_imu_sigma_yaw", line_heading_imu_sigma_yaw, 0.08);
  n_params_.param<double>("line_heading_imu_sum_yaw", line_heading_imu_sum_yaw, 0.08);
  droslog(LogLevel::INFO, "LOC::init(): 参数: line_heading_KF_dist = %.3f, line_heading_KF_window = %d", line_heading_KF_dist, line_heading_KF_window);
  droslog(LogLevel::INFO, "LOC::init(): 参数: line_heading_dist_min = %.3f, line_heading_dist_max = %.3f", line_heading_dist_min, line_heading_dist_max);
  droslog(LogLevel::INFO, "LOC::init(): 参数: line_heading_imu_diff_yaw = %.3f, line_heading_imu_sigma_yaw = %.3f, line_heading_imu_sum_yaw = %.3f",
      line_heading_imu_diff_yaw, line_heading_imu_sigma_yaw, line_heading_imu_sum_yaw);

  HeadingEstimator::Config est_heading_config;
  est_heading_config.line_heading_KF_dist = line_heading_KF_dist;
  est_heading_config.line_heading_KF_window = line_heading_KF_window;
  est_heading_config.line_heading_dist_min = line_heading_dist_min;
  est_heading_config.line_heading_dist_max = line_heading_dist_max;
  est_heading_config.line_heading_imu_diff_yaw = line_heading_imu_diff_yaw;
  est_heading_config.line_heading_imu_sigma_yaw = line_heading_imu_sigma_yaw;
  est_heading_config.line_heading_imu_sum_yaw = line_heading_imu_sum_yaw;
  HeadingEstimator::Instance()->SetConfig(est_heading_config); 

  bool use_vmap = false;
  n_params_.param<bool>("use_vmap", use_vmap, false);
  use_vmap_cfg_.store(use_vmap);
  use_vmap_.store(use_vmap_cfg_.load());
  
  int fusion_type = 0;
  n_params_.param<int>("fusion_type", fusion_type, 0);
  fusion_type_cfg_.store(fusion_type);
  fusion_type_.store(fusion_type_cfg_.load());
  droslog(LogLevel::INFO, "LOC::init(): 参数: 融合类型fusion_type: %d, use_vmap: %d", fusion_type, use_vmap);

  n_params_.param<double>("track_off_ref_dist", track_off_rtk_dist_, 300.1);
  n_params_.param<double>("track_vio_dist", track_off_reloc_dist_, 60.1);
  n_params_.param<double>("track_only_iw_dist", track_only_iw_dist_, 20.1);
  droslog(LogLevel::INFO, "LOC::init(): 参数: track_off_rtk_dist: %.3f, track_off_reloc_dist: %.3f, track_only_iw_dist: %.3f", 
      track_off_rtk_dist_, track_off_reloc_dist_, track_only_iw_dist_);

  double rtk_fix_ll_sigma = 0.02;
  double rtk_float_ll_sigma = 0.03;
  double pose_adj_factor = 1.0;
  double pose_align_factor = 0.16;
  double pose_rp_factor = 0.04;
  double rtk_fix_info_sigma = 0.49;
  double rtk_float_info_sigma = 0.03;
  double reloc_info_pos_sigma = 0.3;
  double reloc_info_quat_sigma = 0.1;
  n_params_.param<double>("rtk_fix_ll_sigma", rtk_fix_ll_sigma, 0.02);
  n_params_.param<double>("rtk_float_ll_sigma", rtk_float_ll_sigma, 0.03);
  n_params_.param<double>("pose_adj_factor", pose_adj_factor, 1.0);
  n_params_.param<double>("pose_align_factor", pose_align_factor, 0.16);
  n_params_.param<double>("pose_rp_factor", pose_rp_factor, 0.04);
  n_params_.param<double>("rtk_fix_info_sigma", rtk_fix_info_sigma, 0.49);
  n_params_.param<double>("rtk_float_info_sigma", rtk_float_info_sigma, 0.03);
  n_params_.param<double>("reloc_info_pos_sigma", reloc_info_pos_sigma, 0.3);
  n_params_.param<double>("reloc_info_quat_sigma", reloc_info_quat_sigma, 0.1);
  droslog(LogLevel::INFO, "LOC::init() 参数: rtk_fix_ll_sigma: %.3f, rtk_float_ll_sigma: %.3f, pose_adj_factor: %.3f, pose_align_factor: %.3f", 
      rtk_fix_ll_sigma, rtk_float_ll_sigma, pose_adj_factor, pose_align_factor);
  droslog(LogLevel::INFO, "LOC::init() 参数: rtk_fix_info_sigma: %.3f, rtk_float_info_sigma: %.3f, reloc_info_pos_sigma: %.3f, reloc_info_quat_sigma: %.3f", 
      rtk_fix_info_sigma, rtk_float_info_sigma, reloc_info_pos_sigma, reloc_info_quat_sigma);
  VioTracker::VioTrackerParams vt_params;
  vt_params.rtk_fix_ll_sigma = rtk_fix_ll_sigma;
  vt_params.rtk_float_ll_sigma = rtk_float_ll_sigma;
  vt_params.pose_adj_factor = pose_adj_factor;
  vt_params.pose_align_factor = pose_align_factor;
  vt_params.pose_rp_factor = pose_rp_factor;
  vt_params.rtk_fix_info_sigma = rtk_fix_info_sigma;
  vt_params.rtk_float_info_sigma = rtk_float_info_sigma;
  vt_params.reloc_info_pos_sigma = reloc_info_pos_sigma;
  vt_params.reloc_info_quat_sigma = reloc_info_quat_sigma;
  VioTracker::Instance()->SetParams(vt_params);
  
  LocTracker::LocTrackerParams loc_params;
  loc_params.rtk_fix_ll_sigma = rtk_fix_ll_sigma;
  loc_params.rtk_float_ll_sigma = rtk_float_ll_sigma;
  loc_params.pose_adj_factor = pose_adj_factor;
  loc_params.pose_align_factor = pose_align_factor;
  loc_params.pose_rp_factor = pose_rp_factor;
  loc_params.rtk_fix_info_sigma = rtk_fix_info_sigma;
  loc_params.rtk_float_info_sigma = rtk_float_info_sigma;
  loc_params.reloc_info_pos_sigma = reloc_info_pos_sigma;
  loc_params.reloc_info_quat_sigma = reloc_info_quat_sigma;
  lidar_tracker_.SetParams(loc_params);

  localization_info_msg_.datum_initialized = false;
  localization_info_msg_.heading_initialized = false;
  localization_info_msg_.state = mower_msgs::MowerLocalizationInfo::STATE_INIT;

  droslog(LogLevel::INFO, "LOC::init() ------");
}