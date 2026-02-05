#ifndef COMMON_GNSS_CONVERTER_H
#define COMMON_GNSS_CONVERTER_H

#include <atomic>
#include <GeographicLib/LocalCartesian.hpp>

#include "common/data_type.h"

// Gnss转换器, 用于Gnss坐标与局部地图坐标的相互转换
// 1. Gnss坐标<-->局部地理坐标(东北天)
// 2. 局部地理坐标(东北天)<-->局部地图坐标(XYZ)
// 
// 局部地图坐标系: 以桩点(机器电极对接状态下机器运动中心)为原点, X轴指向桩点正前方, Y轴指向桩点正右方, Z轴指向桩点正上方
//
// 建图:
// 1. Gnss首个有效值: 将对应的基站坐标设置为地理坐标转换原点
// 2. 下桩后dist米内, 将最早的连续运动点对(ENU <--> local_pos), 计算地理坐标与局部地图坐标转换矩阵 
//
// 定位:
// 1. Gnss首个有效值: 将对应的基站坐标设置为地理坐标转换原点
// 2. 加载地图时获取的基站Gnss+桩点Gnss+桩点朝向: 计算地理坐标与局部地图坐标转换矩阵
// 3. 基站偏移修正: 修正基站偏移
class GnssConverter {
 public:
  // 均是以gps天线为参照中心 
  static GnssConverter* Instance() {
    static GnssConverter ins;
    return &ins;
  }
  ~GnssConverter() {}

  void Reset();

  void SetCurRtkBaseStation(const Eigen::Vector3d& rtk_base_gnss);

  // map_rtk_base_gnss: 建图基站Gnss坐标
  // dock_station_gnss: 桩点(局部地图坐标系原点)在建图时的Gnss坐标
  // local_map_rpy: 局部地图坐标系在局部地理下的姿态, 目前仅考虑绕Z轴的旋转, 弧度
  void SetLocalMapOffset(const Eigen::Vector3d& map_rtk_base_gnss, 
                         const Eigen::Vector3d& dock_station_gnss, 
                         const Eigen::Vector3d& local_map_rpy);

  bool CurRtkBaseStationValid() const {
    return cur_rtk_base_station_valid_.load();
  }

  bool LocalMapOffsetValid() const {
    return local_map_offset_valid_.load();
  }

  Eigen::Vector3d Gnss2Enu(const Eigen::Vector3d& gnss) {
    double e, n, u;
    cur_rtk_geo_.Forward(gnss(0), gnss(1), gnss(2), e, n, u);
    return Eigen::Vector3d(e, n, u);
  }
  Eigen::Vector3d Enu2Gnss(const Eigen::Vector3d& enu_pos) {
    double lat, lon, alt;
    cur_rtk_geo_.Reverse(enu_pos(0), enu_pos(1), enu_pos(2), lat, lon, alt);
    return Eigen::Vector3d(lat, lon, alt);
  }

  Eigen::Quaterniond EnuQ2LocalQ(const Eigen::Quaterniond& enu_q) {
    return local_map_q_.inverse() * enu_q;
  }
  Eigen::Vector3d Enu2LocalPos(const Eigen::Vector3d& enu_pos) {
    return local_map_q_.inverse() * (enu_pos - local_map_t_);
  }
  Eigen::Vector3d LocalPos2Enu(const Eigen::Vector3d& local_pos) {
    return local_map_q_ * local_pos + local_map_t_;
  }

  Eigen::Vector3d Gnss2LocalPos(const Eigen::Vector3d& gnss) {
    return Enu2LocalPos(Gnss2Enu(gnss));
  }
  Eigen::Vector3d LocalPos2Gnss(const Eigen::Vector3d& local_pos) {
    return Enu2Gnss(LocalPos2Enu(local_pos));
  }
  
  // 调试用, 转回地图显示的gnss坐标

  Eigen::Vector3d LocalPos2MapGnss(const Eigen::Vector3d& local_pos) {
    return Enu2MapGnss(LocalPos2Enu(local_pos));
  }
  Eigen::Vector3d Enu2MapGnss(const Eigen::Vector3d& enu) {
    double lat, lon, alt;
    map_rtk_geo_.Reverse(enu(0), enu(1), enu(2), lat, lon, alt);
    return Eigen::Vector3d(lat, lon, alt);
  }
  Eigen::Vector3d Gnss2MapGnss(const Eigen::Vector3d& gnss) {
    return Enu2MapGnss(Gnss2Enu(gnss));
  }

  Eigen::Vector3d GetMapRtkGnss() const {
    return map_rtk_base_gnss_;
  }
  Eigen::Vector3d GetChargingStationGnss() const {
    return dock_station_gnss_;
  }
  Eigen::Vector3d GetChargingStationOrientation() const {
    return local_map_rpy_;
  }
  Eigen::Vector3d GetRtkMapPos() const {
    return rtk_base_map_xyz_;
  }

  Eigen::Vector3d GetChargingStationEnuAtMap() {
    return local_map_t_;
  }

 private:
  GnssConverter() : cur_rtk_base_station_valid_(false), local_map_offset_valid_(false) {}
  GnssConverter(const GnssConverter&) = delete;
  GnssConverter& operator=(const GnssConverter&) = delete;

  std::atomic_bool cur_rtk_base_station_valid_;
  std::atomic_bool local_map_offset_valid_;

  Eigen::Vector3d local_map_t_ = Eigen::Vector3d::Zero();     // 局部地图坐标系原点在局部地理下的位置
  Eigen::Quaterniond local_map_q_ = Eigen::Quaterniond::Identity();  // 局部地图坐标系在局部地理下的姿态
  
  GeographicLib::LocalCartesian cur_rtk_geo_;
  GeographicLib::LocalCartesian map_rtk_geo_;

  Eigen::Vector3d map_rtk_base_gnss_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d dock_station_gnss_ = Eigen::Vector3d::Zero(); 
  Eigen::Vector3d local_map_rpy_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d rtk_base_map_xyz_ = Eigen::Vector3d::Zero();
};

#endif // COMMON_GNSS_CONVERTER_H