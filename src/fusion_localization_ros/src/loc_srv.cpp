
#include "version.h"
#include "loc_node.h"

#include "common/math_utils.h"
#include "common/sysutils.h"
#include "common/debug_client.h"
#include "droslog/log.h"

#include "common/gnss_converter.h"
#include "common/gnss_initor.h"
#include "common/sensor_monitor.h"
#include "common/vio_reseter.h"
#include "common/vio_tracker.h"
#include "common/vio_gnss_initor.h"
#include "common/vmap_monitor.h"

#include <fstream>

using namespace utils;

std::string g_map_name = "-;-";
std::string g_map_root_dir = "/userdata/RobotData/map/";

const std::string Debug_help_str = "\
[pwd]: run path; \
[version]: version + compile time; \
[rtk_off]:; rtk -> single\
[rtk_on]:; rtk -> origin\
[rtk_fix]:; rtk -> narrow_int \
[rtk_ref_change]:; rtk_ref -> change 1.5s \
";
bool loc_node::srvDebug(mower_msgs::TriggerRequest &req, mower_msgs::TriggerResponse& rep)
{
  const std::string req_type = req.arg;
  rep.result = 1;
  if (req_type == "help") {
    rep.message = Debug_help_str;
  } else if (req_type == "pwd") {
    rep.message = ExecWithStdout("pwd");
  } else if (req_type == "version") {
    rep.message = std::string(NODE_NAME) + "_" + std::string(NODE_VERSION) + "_" + std::string(NODE_VERSION_DATE) + "--" + std::string(COMPILE_TIME);
  } else if (req_type == "rtk_off") {
    DebugClient::Instance()->SetRtkState(1);
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, 强制设置RTK为单点解");
  } else if (req_type == "rtk_on") {
    DebugClient::Instance()->SetRtkState(0);
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, RTK恢复原状");
  } else if (req_type == "rtk_fix") {
    DebugClient::Instance()->SetRtkState(2);
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, 强制设置RTK为固定解");
  } else if (req_type == "no_rtk") {
    DebugClient::Instance()->SetRtkState(3);
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, 强制不输入RTK数据");
  } else if (req_type == "rtk_ref_change") {
    DebugClient::Instance()->UpdateRtkRefChange(GetNow_Steady());
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, 强制设置RTK基站坐标跳变");
  } else if (req_type == "docked") {
    DebugClient::Instance()->SetDockingState(1);
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, 强制设置机器在桩, 持续10s");
  } else if (req_type == "only_vio") {
    fusion_type_.store(1);
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, 设置为纯vio模式, 激活vmap");
  } else if (req_type == "reset_only_vio") {
    fusion_type_.store(fusion_type_cfg_.load());
    use_vmap_.store(use_vmap_cfg_.load());
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, 解除纯vio模式, vmap恢复配置状态");
  } else if (req_type == "only_lidar") {
    fusion_type_.store(2);
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, 设置为纯lidar模式");
  } else if (req_type == "reset_only_lidar") {
    fusion_type_.store(fusion_type_cfg_.load());
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, 解除纯lidar模式");
  } else if (req_type == "wheel_on") {
    DebugClient::Instance()->SetUseWheelVel(true);
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, 开启轮速融合");
  } else if (req_type == "wheel_off") {
    DebugClient::Instance()->SetUseWheelVel(false);
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, 关闭轮速融合");
  } else if (req_type == "loc_on") {
    DebugClient::Instance()->SetLocAlwaysValid(true);
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, 开启定位常有效");
  } else if (req_type == "loc_off") {
    DebugClient::Instance()->SetLocAlwaysValid(false);
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, 关闭定位常有效");
  } else if (req_type == "align_off") {
    DebugClient::Instance()->SetUseAlign(false);
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, 关闭loc_tracker的对齐功能");
  } else if (req_type == "align_on") {
    DebugClient::Instance()->SetUseAlign(true);
    droslog(LogLevel::WARN, "LOC::srvDebug() 调试指令, 恢复loc_tracker的对齐功能");
  } else {
    droslog(LogLevel::WARN, "LOC::srvDebug() req NOT VALID : %s", req_type.c_str());
  }
  droslog(LogLevel::INFO, "LOC::srvDebug() req=%s, res=%s", req_type.c_str(), rep.message.c_str());
  return true;
}

bool loc_node::srvCheckHeading(mower_msgs::TriggerRequest &req, mower_msgs::TriggerResponse& rep) {
  rep.result = locator_.IsValid();
  return true;
}

bool loc_node::srvComputeHeading(mower_msgs::TriggerRequest &req,
                                  mower_msgs::TriggerResponse &rep) {
  if (req.arg == "start") {
    ROS_WARN("LOC::srvComputeHeading() START");
    droslog(LogLevel::INFO, "LOC::srvComputeHeading() START, 航向角初始化, 开始直行...");
    GnssInitor::Instance()->Reset();
    GnssInitor::Instance()->StartInit();
    {
      std::lock_guard<std::mutex> lock(loc_mutex_);
      int work_mode = locator_.GetWorkMode();
      int work_state = locator_.GetWorkState();
      if (work_state == 0) {
        locator_.SetWorkState(1);
        droslog(LogLevel::INFO, "LOC::srvComputeHeading() START, 航向角初始化, 非在桩初始化...");
      } else {
        droslog(LogLevel::INFO, "LOC::srvComputeHeading() START, 航向角初始化, mode=%d, state=%d", work_mode, work_state);
      }
    }
    Sleep(500);
  } else if (req.arg == "stop") {
    ROS_WARN("LOC::srvComputeHeading() STOP");
    droslog(LogLevel::INFO, "LOC::srvComputeHeading() STOP, 逻辑层直行完成, 可以计算航向角");
    
    int ret = GnssInitor::Instance()->FinishInit();
    // 仅带RTK融合模式才需要gnss_init
    if (fusion_type_.load() == 0) {
      if (ret == 0) {
        double enu_heading = GnssInitor::Instance()->GetInitHeading();
        double dist = GnssInitor::Instance()->GetInitDist();
        std::lock_guard<std::mutex> lock(loc_mutex_);
        int mode = locator_.GetWorkMode();
        int work_state = locator_.GetWorkState();
        if (mode == 0) {
          auto start_gnss = GnssInitor::Instance()->GetStartGnss();
          // 建图gnss初始化成功, 设置地图偏移量
          Eigen::Vector3d base_station_gps = GnssConverter::Instance()->Enu2Gnss(Eigen::Vector3d(0,0,0));
          Eigen::Vector3d charging_station_gps = GnssConverter::Instance()->Enu2Gnss(start_gnss.gnss.enu);
          Eigen::Vector3d charging_station_rpy;
          charging_station_rpy << 0.0, 0.0, enu_heading;
          GnssConverter::Instance()->SetLocalMapOffset(base_station_gps, charging_station_gps, charging_station_rpy);
          VioGnssInitor::Instance()->StopInit();
          droslog(LogLevel::INFO, "LOC::srvComputeHeading() STOP, 建图gnss初始化成功, 已设置地图偏移量");
        } else if (mode == 1) {
          // 定位gnss初始化成功(非在桩启动), loc修正/初始化
          if (work_state != 3) {
            auto end_gnss = GnssInitor::Instance()->GetEndGnss();
            Eigen::Vector3d init_pos = GnssConverter::Instance()->Enu2LocalPos(end_gnss.gnss.enu);
            double map_heading = GnssConverter::Instance()->GetChargingStationOrientation().z();
            double init_heading = KeepAngleInPI(enu_heading - map_heading);
            Eigen::Matrix3d init_q = Eigen::AngleAxisd(init_heading, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            common::NavState nav_state;
            nav_state.pos = init_pos;
            nav_state.quat = init_q;
            nav_state.timestamp = end_gnss.timestamp;
            
            locator_.SetInitNavState(nav_state);
            droslog(LogLevel::INFO, "LOC::srvComputeHeading() STOP, 定位gnss初始化成功, enu_heading=%.3f, map_heading=%.3f, init_heading=%.3f", enu_heading, map_heading, init_heading);
          } else {
            droslog(LogLevel::INFO, "LOC::srvComputeHeading() STOP, 无需gnss初始化, 已在桩初始化");
          }
          
        } else {
          droslog(LogLevel::ERROR, "LOC::srvComputeHeading() STOP, 未知工作模式, mode: %d", mode);
        }
      } else {
        droslog(LogLevel::WARN, "LOC::srvComputeHeading() STOP, GNSS直行初始化失败, ret: %d", ret);
      }
    } else {
      droslog(LogLevel::INFO, "LOC::srvComputeHeading() STOP, 无需GNSS直行初始化, fusion_type=%d", fusion_type_.load());
    }
  }
  return true;
}

bool loc_node::srvSetState(mower_msgs::TriggerRequest &req,
                            mower_msgs::TriggerResponse &rep) {
  if (req.arg == "start_build_map") {
    ROS_WARN("LOC::srvSetState() start build map");
    droslog(LogLevel::INFO, "LOC::srvSetState() 收到建图开始指令: %s, 设置为建图模式, 重置VIO/LIO, fusion_type: %d", req.arg.c_str(), fusion_type_.load());
    
    if (fusion_type_.load() == 0 || fusion_type_.load() == 1) {
      mower_msgs::Trigger reset_vio;
      reset_vio.request.arg = "reset_vio";
      if (vio_reset_clt_.waitForExistence(ros::Duration(2.0))) {
        if (!vio_reset_clt_.call(reset_vio)) {
          droslog(LogLevel::ERROR, "LOC::srvSetState() vio_reset_clt_ 调用重置VIO服务失败, rep=%d, err_msg=%s", reset_vio.response.result, reset_vio.response.message.c_str());
        } else {
          droslog(LogLevel::INFO, "LOC::srvSetState() vio_reset_clt_ 调用重置VIO服务成功");
        }
        VioReseter::Instance()->UpdateResetTime();
      } else {
        ROS_ERROR("LOC::srvSetState() vio_reset_clt_ wait for existence timeout");
        droslog(LogLevel::ERROR, "LOC::srvSetState() vio_reset_clt_ 等待重置VIO服务超时, VIO节点可能未启动或者异常");
      }
      Sleep(200);
      ProcVioReset();
      VioTracker::Instance()->InitAtStation(ros::Time::now().toSec());
      if (fusion_type_.load() == 0) {
        VioGnssInitor::Instance()->InitAtStation(ros::Time::now().toSec());
      }
      
      if (fusion_type_.load() == 1 || (fusion_type_.load() == 0 && use_vmap_.load())) {  
        mower_msgs::Trigger ctl_msg;
        ctl_msg.request.arg = "start_mapping";
        if (vmap_ctrl_clt_.waitForExistence(ros::Duration(2.0))) {
          if (!vmap_ctrl_clt_.call(ctl_msg)) {
            droslog(LogLevel::ERROR, "LOC::srvSetState() vmap_ctrl_clt_ 调用开始视觉建图服务失败, rep=%d, err_msg=%s", ctl_msg.response.result, ctl_msg.response.message.c_str());
          } else {
            droslog(LogLevel::INFO, "LOC::srvSetState() vmap_ctrl_clt_ 调用开始视觉建图服务成功");
          }
        } else {
          droslog(LogLevel::ERROR, "LOC::srvSetState() vmap_ctrl_clt_ 等待视觉建图服务超时, VMAP节点可能未启动或者异常");
        }
      }
    }
    if (fusion_type_.load() == 2) {
      mower_msgs::Trigger reset_lio;
      reset_lio.request.arg = "reset_lio";
      if (lio_ctrl_clt_.waitForExistence(ros::Duration(2.0))) {
        if (!lio_ctrl_clt_.call(reset_lio)) {
          droslog(LogLevel::ERROR, "LOC::srvSetState() lio_ctrl_clt_ 调用重置LIO服务失败, rep=%d, err_msg=%s", reset_lio.response.result, reset_lio.response.message.c_str());
        } else {
          droslog(LogLevel::INFO, "LOC::srvSetState() lio_ctrl_clt_ 调用重置LIO服务成功");
        }
      } else {
        ROS_ERROR("LOC::srvSetState() lio_ctrl_clt_ wait for existence timeout");
        droslog(LogLevel::ERROR, "LOC::srvSetState() lio_ctrl_clt_ 等待重置LIO服务超时, LIO节点可能未启动或者异常");
      }
      Sleep(200);
      ProcLioReset();
      lidar_tracker_.InitAtStation(ros::Time::now().toSec());
      if (fusion_type_.load() == 3) {
        // TODO LioGnssInitor(): 用于处理建图时 lidar-map与RTK的对齐关系
      }

      mower_msgs::Trigger ctl_msg;
      ctl_msg.request.arg = "start_mapping";
      if (lio_ctrl_clt_.waitForExistence(ros::Duration(2.0))) {
        if (!lio_ctrl_clt_.call(ctl_msg)) {
          droslog(LogLevel::ERROR, "LOC::srvSetState() lio_ctrl_clt_ 调用开始激光建图服务失败, rep=%d, err_msg=%s", ctl_msg.response.result, ctl_msg.response.message.c_str());
        } else {
          droslog(LogLevel::INFO, "LOC::srvSetState() lio_ctrl_clt_ 调用开始激光建图服务成功");
        }
      } else {
        droslog(LogLevel::ERROR, "LOC::srvSetState() lio_ctrl_clt_ 等待激光建图服务超时, 节点可能未启动或者异常");
      }
    }

    GnssConverter::Instance()->Reset();

    {
      common::NavState nav_state;
      nav_state.timestamp = ros::Time::now().toSec();
      std::lock_guard<std::mutex> lock(loc_mutex_);
      locator_.Reset();
      locator_.SetWorkMode(0);
      locator_.SetInitNavState(nav_state);
    }
    
    rep.result = 1;
    rep.message = "OK";
    Sleep(1000);
  } else if (req.arg == "stop_build_map") {
    // 这里结束建图, 释放建图相关资源, 且不会保存地图
    ROS_WARN("LOC::srvSetState() stop build map");
    droslog(LogLevel::INFO, "LOC::srvSetState() 收到建图结束指令: %s", req.arg.c_str());
    // JOHN_NOTE 预留给建图接口
    rep.result = 0;
    rep.message = "OK";
  } else if (req.arg == "start_extend_map") {
    // JOHN_NOTE 这里切入拓展建图状态
    ROS_WARN("LOC::srvSetState() start extend map");
    droslog(LogLevel::INFO, "LOC::srvSetState() 收到拓展建图指令: %s", req.arg.c_str());
    Sleep(1000);    
    rep.result = 1;
    rep.message = "OK";
  } else if (req.arg == "stop_work") {
    // 2026-01-17: 工作结束指令，通知 VMAP 刷盘并执行全局优化
    ROS_WARN("LOC::srvSetState() stop work");
    droslog(LogLevel::INFO, "LOC::srvSetState() 收到工作结束指令: %s", req.arg.c_str());
    
    if ((fusion_type_.load() == 0 && use_vmap_.load()) || fusion_type_.load() == 1) {
      mower_msgs::Trigger stop_msg;
      stop_msg.request.arg = "stop_reloc";
      if (vmap_ctrl_clt_.waitForExistence(ros::Duration(2.0))) {
        if (vmap_ctrl_clt_.call(stop_msg)) {
          droslog(LogLevel::INFO, "LOC::srvSetState() 已发送 stop_reloc 指令，VMAP 将异步执行刷盘和全局优化");
        } else {
          droslog(LogLevel::WARN, "LOC::srvSetState() stop_reloc 调用失败");
        }
      } else {
        droslog(LogLevel::WARN, "LOC::srvSetState() vmap_ctrl 服务不可用");
      }
    }
    
    rep.result = 1;
    rep.message = "OK";
  } else {
    droslog(LogLevel::WARN, "LOC::srvSetState() req NOT VALID: %s", req.arg.c_str());
    rep.result = 0;
    rep.message = "req NOT VALID";
  }
  return true;
}

bool loc_node::srvLoadMap(mower_msgs::LocatorLoadMap::Request &req, mower_msgs::LocatorLoadMap::Response &res) {
  ROS_WARN("LOC::srvLoadMap() ++++++");
  droslog(LogLevel::WARN, "LOC::srvLoadMap() 收到载入地图请求, map_name: %s, charging_station_gps: %.8f, %.8f, %.2f, base_station_gps: %.8f, %.8f, %.2f, charging_station_orientation: %.6f(rad)", 
      req.map_name.c_str(), 
      req.charging_station_gps.position.latitude, req.charging_station_gps.position.longitude, req.charging_station_gps.position.altitude,
      req.base_station_gps.position.latitude, req.base_station_gps.position.longitude, req.base_station_gps.position.altitude,
      req.charging_station_orientation);
  
  Eigen::Vector3d map_rtk_base_gnss(req.base_station_gps.position.latitude, req.base_station_gps.position.longitude, req.base_station_gps.position.altitude);
  Eigen::Vector3d dock_station_gnss(req.charging_station_gps.position.latitude, req.charging_station_gps.position.longitude, req.charging_station_gps.position.altitude);
  Eigen::Vector3d local_map_rpy(0, 0, req.charging_station_orientation);
  
  g_map_name = req.map_name;
  map_name_ = req.map_name;  // 同步设置类成员，供 MonitorVioThread 使用
  std::string map_path = g_map_root_dir + g_map_name;
  if (map_path.back() != '/')
    map_path += "/";
  std::string rtk_info_fn = map_path + "rtk_info.txt";
  if (IsFileExisting(rtk_info_fn.c_str())) {
    std::ifstream file(rtk_info_fn);
    double base_lon, base_lat, base_alt;
    double dock_lon, dock_lat, dock_alt;
    file >> base_lat >> base_lon >> base_alt;
    file >> dock_lat >> dock_lon >> dock_alt;
    file.close();
    map_rtk_base_gnss[2] = base_alt;
    dock_station_gnss[2] = dock_alt;
    droslog(LogLevel::INFO, "LOC::srvLoadMap() 加载rtk信息文件: base_lla: %.8f,%.8f,%.3f, dock_lla: %.8f,%.8f,%.3f", base_lat, base_lon, base_alt, dock_lat, dock_lon, dock_alt);
  } else {
    droslog(LogLevel::WARN, "LOC::srvLoadMap() rtk信息文件: %s 不存在", rtk_info_fn.c_str());
  }
  
  droslog(LogLevel::WARN, "LOC::srvLoadMap() 收到载入地图请求, 设置为定位模式, 重置Gnss转换器, 重置vio/lio, fusion_type: %d", fusion_type_.load());
  
  auto cur_ts = ros::Time::now().toSec();
  auto CS_state = SensorMonitor::Instance()->GetChargingStationState(cur_ts, 2.0);

  if (fusion_type_.load() == 0 || fusion_type_.load() == 1) {
    mower_msgs::Trigger reset_vio;
    reset_vio.request.arg = "reset_vio";
    if (vio_reset_clt_.waitForExistence(ros::Duration(2.0))) {
      if (!vio_reset_clt_.call(reset_vio)) {
        droslog(LogLevel::ERROR, "LOC::srvLoadMap() vio_reset_clt_ 调用重置VIO服务失败, rep=%d, err_msg=%s", reset_vio.response.result, reset_vio.response.message.c_str());
      } else {
        droslog(LogLevel::INFO, "LOC::srvLoadMap() vio_reset_clt_ 调用重置VIO服务成功");
      }
      VioReseter::Instance()->UpdateResetTime();
    } else {
      ROS_ERROR("LOC::srvLoadMap() vio_reset_clt_ wait for existence timeout");
      droslog(LogLevel::ERROR, "LOC::srvLoadMap() vio_reset_clt_ 等待重置VIO服务超时, VIO节点可能未启动或者异常");
    }
    Sleep(200);
    ProcVioReset();
  }
  if (fusion_type_.load() == 2) {
    mower_msgs::Trigger reset_lio;
    reset_lio.request.arg = "reset_lio";
    if (lio_ctrl_clt_.waitForExistence(ros::Duration(2.0))) {
      if (!lio_ctrl_clt_.call(reset_lio)) {
        droslog(LogLevel::ERROR, "LOC::srvLoadMap() lio_ctrl_clt_ 调用重置LIO服务失败, rep=%d, err_msg=%s", reset_lio.response.result, reset_lio.response.message.c_str());
      } else {
        droslog(LogLevel::INFO, "LOC::srvLoadMap() lio_ctrl_clt_ 调用重置LIO服务成功");
      }
    } else {
      ROS_ERROR("LOC::srvLoadMap() lio_ctrl_clt_ wait for existence timeout");
      droslog(LogLevel::ERROR, "LOC::srvLoadMap() lio_ctrl_clt_ 等待重置LIO服务超时, LIO节点可能未启动或者异常");
    }
    Sleep(200);
    ProcLioReset();
  }

  locator_.Reset();
  locator_.SetWorkMode(1);
  locator_.SetWorkState(0);
  if (CS_state > 0) {
    if (fusion_type_.load() == 0 || fusion_type_.load() == 1) {
      VioTracker::Instance()->InitAtStation(ros::Time::now().toSec());
    } else if (fusion_type_.load() == 2) {
      lidar_tracker_.InitAtStation(ros::Time::now().toSec());
    }      

    common::NavState nav_state;
    nav_state.timestamp = ros::Time::now().toSec();
    std::lock_guard<std::mutex> lock(loc_mutex_);
    locator_.SetInitNavState(nav_state);
    ROS_INFO("LOC::srvLoadMap() CS_state=%d, IN CHARGING STATION", CS_state);
    droslog(LogLevel::INFO, "LOC::srvLoadMap() CS_state=%d, 在桩启动", CS_state);
  } else {
    ROS_ERROR("LOC::srvLoadMap() CS_state=%d, NOT IN CHARGING STATION", CS_state);
    droslog(LogLevel::ERROR, "LOC::srvLoadMap() CS_state=%d, 非在桩启动", CS_state);
  }

  if ((fusion_type_.load() == 0 && use_vmap_.load()) || fusion_type_.load() == 1) {
    VmapMonitor::Instance()->SetLoadMapTs(GetNow_Steady());

    mower_msgs::Trigger ctl_msg;
    ctl_msg.request.arg = req.map_name;
    if (load_vmap_clt_.waitForExistence(ros::Duration(2.0))) {
      if (!load_vmap_clt_.call(ctl_msg)) {
        droslog(LogLevel::ERROR, "LOC::srvLoadMap() load_vmap_clt_ 调用视觉加载地图服务失败, rep=%d, err_msg=%s", ctl_msg.response.result, ctl_msg.response.message.c_str());
      } else {
        droslog(LogLevel::INFO, "LOC::srvLoadMap() load_vmap_clt_ 调用视觉加载地图服务成功");
      }
    } else {
      droslog(LogLevel::ERROR, "LOC::srvLoadMap() load_vmap_clt_ 等待视觉加载地图服务超时, VMAP节点可能未启动或者异常");
    }
  }
  if (fusion_type_.load() == 2) {
    mower_msgs::Trigger ctl_msg;
    ctl_msg.request.arg = req.map_name;
    if (lio_loadmap_clt_.waitForExistence(ros::Duration(2.0))) {
      if (!lio_loadmap_clt_.call(ctl_msg)) {
        droslog(LogLevel::ERROR, "LOC::srvLoadMap() lio_loadmap_clt_ 调用激光加载地图服务失败, rep=%d, err_msg=%s", ctl_msg.response.result, ctl_msg.response.message.c_str());
      } else {
        droslog(LogLevel::INFO, "LOC::srvLoadMap() lio_loadmap_clt_ 调用激光加载地图服务成功");
      }
    } else {
      droslog(LogLevel::ERROR, "LOC::srvLoadMap() lio_loadmap_clt_ 等待激光加载地图服务超时, 节点可能未启动或者异常");
    }
  }
  
  GnssConverter::Instance()->Reset();
  GnssConverter::Instance()->SetLocalMapOffset(map_rtk_base_gnss, dock_station_gnss, local_map_rpy);
  
  res.result = 0;
  res.messages = "OK";
  Sleep(1000);
  return true;
}

bool loc_node::srvSaveMap(mower_msgs::LocatorSaveMap::Request &req, mower_msgs::LocatorSaveMap::Response &res) {
  ROS_WARN("LOC::srvSaveMap() ++++++");
  droslog(LogLevel::WARN, "LOC::srvSaveMap() 收到保存地图请求, map_name: %s, fusion_type: %d", req.map_name.c_str(), fusion_type_.load());

  g_map_name = req.map_name;
  // 记录地图rtk信息
  if (GnssConverter::Instance()->LocalMapOffsetValid()) {
    droslog(LogLevel::INFO, "LOC::srvSaveMap() 保存rtk_info.txt......");

    auto base_station = GnssConverter::Instance()->GetMapRtkGnss();
    auto charging_station = GnssConverter::Instance()->GetChargingStationGnss();
    auto local_map_rpy = GnssConverter::Instance()->GetChargingStationOrientation();
    auto base_station_map_position = GnssConverter::Instance()->Enu2LocalPos(Eigen::Vector3d(0,0,0));

    std::string map_path = g_map_root_dir + g_map_name;
    if (map_path.back() != '/')
      map_path += "/";
    std::string rtk_info_fn = map_path + "rtk_info.txt";
    std::ofstream file(rtk_info_fn);
    file << base_station(0) << " " << base_station(1) << " " << base_station(2) << std::endl;
    file << charging_station(0) << " " << charging_station(1) << " " << charging_station(2) << std::endl;
    file << local_map_rpy(0) << " " << local_map_rpy(1) << " " << local_map_rpy(2) << std::endl;
    file << base_station_map_position(0) << " " << base_station_map_position(1) << " " << base_station_map_position(2) << std::endl;
    file.close();
  } else {
    droslog(LogLevel::ERROR, "LOC::srvSaveMap() 无RTK地图偏移信息, 不保存rtk_info.txt");
  }

  if ((fusion_type_.load() == 0 && use_vmap_.load()) || fusion_type_.load() == 1) {
    mower_msgs::Trigger ctl_msg;
    ctl_msg.request.arg = req.map_name;
    if (save_vmap_clt_.waitForExistence(ros::Duration(2.0))) {
      if (!save_vmap_clt_.call(ctl_msg)) {
        droslog(LogLevel::ERROR, "LOC::srvSaveMap() save_vmap_clt_ 调用视觉保存地图服务失败, rep=%d, err_msg=%s", ctl_msg.response.result, ctl_msg.response.message.c_str());
      } else {
        droslog(LogLevel::INFO, "LOC::srvSaveMap() save_vmap_clt_ 调用视觉保存地图服务成功");
      }
    } else {
      droslog(LogLevel::ERROR, "LOC::srvSaveMap() save_vmap_clt_ 等待视觉保存地图服务超时, VMAP节点可能未启动或者异常");
    }
  }
  if (fusion_type_.load() == 2) {
    mower_msgs::Trigger ctl_msg;
    ctl_msg.request.arg = req.map_name;
    if (lio_savemap_clt_.waitForExistence(ros::Duration(2.0))) {
      if (!lio_savemap_clt_.call(ctl_msg)) {
        droslog(LogLevel::ERROR, "LOC::srvSaveMap() lio_savemap_clt_ 调用激光保存地图服务失败, rep=%d, err_msg=%s", ctl_msg.response.result, ctl_msg.response.message.c_str());
      } else {
        droslog(LogLevel::INFO, "LOC::srvSaveMap() lio_savemap_clt_ 调用激光保存地图服务成功");
      }
    } else {
      droslog(LogLevel::ERROR, "LOC::srvSaveMap() lio_savemap_clt_ 等待激光保存地图服务超时, VMAP节点可能未启动或者异常");
    }
  }

  res.result = 0;
  res.messages = "OK";
  
  return true;
}

bool loc_node::srvGetMapInfo(mower_msgs::LocatorGetMapInfo::Request &req, mower_msgs::LocatorGetMapInfo::Response &res) {
  ROS_WARN("LOC::srvGetMapInfo() ++++++");
  droslog(LogLevel::WARN, "LOC::srvGetMapInfo() 收到获取地图信息请求, arg: %s", req.arg.c_str());

  if (GnssConverter::Instance()->LocalMapOffsetValid()) {
    auto base_station = GnssConverter::Instance()->GetMapRtkGnss();
    auto charging_station = GnssConverter::Instance()->GetChargingStationGnss();
    auto local_map_rpy = GnssConverter::Instance()->GetChargingStationOrientation();
    auto base_station_map_position = GnssConverter::Instance()->Enu2LocalPos(Eigen::Vector3d(0,0,0));

    res.map_name = "holdplace";
    res.charging_station_gps.position.latitude = charging_station(0);
    res.charging_station_gps.position.longitude = charging_station(1);
    res.charging_station_gps.position.altitude = charging_station(2);
    res.base_station_gps.position.latitude = base_station(0);
    res.base_station_gps.position.longitude = base_station(1);
    res.base_station_gps.position.altitude = base_station(2);
    
    res.base_station_map_position.x = base_station_map_position(0);
    res.base_station_map_position.y = base_station_map_position(1);
    res.base_station_map_position.z = base_station_map_position(2);
  
    res.charging_station_orientation = local_map_rpy(2);
    res.extra_info = "holdplace";
  } else if (fusion_type_.load() == 1 || fusion_type_.load() == 2 || (fusion_type_.load() == 0 && use_vmap_.load())) {
    res.map_name = "holdplace";
    res.charging_station_gps.position.latitude = 0.1;
    res.charging_station_gps.position.longitude = 0.1;
    res.charging_station_gps.position.altitude = 0.0;
    res.base_station_gps.position.latitude = 0.1;
    res.base_station_gps.position.longitude = 0.1;
    res.base_station_gps.position.altitude = 0.0;
    res.base_station_map_position.x = 0.0;
    res.base_station_map_position.y = 0.0;
    res.base_station_map_position.z = 0.0;
  
    res.charging_station_orientation = 0.0;
    res.extra_info = "holdplace";
  } else {
    res.map_name = "holdplace";
    res.charging_station_gps.position.latitude = 0.0;
    res.charging_station_gps.position.longitude = 0.0;
    res.charging_station_gps.position.altitude = 0.0;
    res.base_station_gps.position.latitude = 0.0;
    res.base_station_gps.position.longitude = 0.0;
    res.base_station_gps.position.altitude = 0.0;
    res.base_station_map_position.x = 0.0;
    res.base_station_map_position.y = 0.0;
    res.base_station_map_position.z = 0.0;
  
    res.charging_station_orientation = 0.0;
    res.extra_info = "holdplace";
  }

  droslog(LogLevel::WARN, "LOC::srvGetMapInfo() 返回地图信息, map_name: %s, charging_station_gps: %.8f, %.8f, %.2f, base_station_gps: %.8f, %.8f, %.2f, base_station_map_position: %.3f,%.3f,%.3f, charging_station_orientation: %.6f(rad), extra_info: %s",
      res.map_name.c_str(), 
      res.charging_station_gps.position.latitude, res.charging_station_gps.position.longitude, res.charging_station_gps.position.altitude,
      res.base_station_gps.position.latitude, res.base_station_gps.position.longitude, res.base_station_gps.position.altitude,
      res.base_station_map_position.x, res.base_station_map_position.y, res.base_station_map_position.z,
      res.charging_station_orientation, res.extra_info.c_str());
  
  return true;
}
