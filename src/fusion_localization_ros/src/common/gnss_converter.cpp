#include "common/gnss_converter.h"

#include "droslog/log.h"

using namespace utils;

void GnssConverter::Reset() {
  // cur_rtk_base_station_valid_.store(false);
  local_map_offset_valid_.store(false);
  droslog(LogLevel::INFO, "GnssConverter::Reset() ++++++");
}

void GnssConverter::SetCurRtkBaseStation(const Eigen::Vector3d& rtk_base_gnss) {
  cur_rtk_base_station_valid_.store(true);
  cur_rtk_geo_.Reset(rtk_base_gnss(0), rtk_base_gnss(1), rtk_base_gnss(2));
  droslog(LogLevel::INFO, "GnssConverter::SetCurRtkBaseStation(): rtk_base_gnss: %.8f, %.8f, %.3f", 
      rtk_base_gnss(0), rtk_base_gnss(1), rtk_base_gnss(2));
}

void GnssConverter::SetLocalMapOffset(const Eigen::Vector3d& map_rtk_base_gnss, 
                        const Eigen::Vector3d& dock_station_gnss, 
                        const Eigen::Vector3d& local_map_rpy) {
  local_map_offset_valid_.store(true);
  map_rtk_geo_.Reset(map_rtk_base_gnss(0), map_rtk_base_gnss(1), map_rtk_base_gnss(2));
  double e, n, u;
  map_rtk_geo_.Forward(dock_station_gnss(0), dock_station_gnss(1), dock_station_gnss(2), e, n, u);
  local_map_t_ << e, n, u;
  local_map_q_ = Eigen::Quaterniond(Eigen::AngleAxisd(local_map_rpy(2), Eigen::Vector3d::UnitZ()));

  map_rtk_base_gnss_ = map_rtk_base_gnss;
  dock_station_gnss_ = dock_station_gnss;
  local_map_rpy_ = local_map_rpy;
  rtk_base_map_xyz_ = Enu2LocalPos(Eigen::Vector3d(0,0,0));

  droslog(LogLevel::INFO, "GnssConverter::SetLocalMapOffset(): 设置gnss地图修正量 map_rtk_bas_gnss:%.8f, %.8f, %.3f, dock_station_gnss: %.8f, %.8f, %.3f", 
      map_rtk_base_gnss(0), map_rtk_base_gnss(1), map_rtk_base_gnss(2),
      dock_station_gnss(0), dock_station_gnss(1), dock_station_gnss(2));
  droslog(LogLevel::INFO, "GnssConverter::SetLocalMapOffset(): local_map_t:%.3f, %.3f, %.3f, local_map_rpy: %.4f, %.4f, %.4f, rtk_base_map_xyz: %.3f, %.3f, %.3f", 
      e, n, u, local_map_rpy(0), local_map_rpy(1), local_map_rpy(2), rtk_base_map_xyz_(0), rtk_base_map_xyz_(1), rtk_base_map_xyz_(2));
}
