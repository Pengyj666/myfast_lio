#include "common/gnss_initor.h"

#include "common/sysutils.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"

using namespace utils;

void GnssInitor::Reset() {
  droslog(LogLevel::INFO, "GnssInitor::Reset() +++++++");
  stage_.store(0);
  std::lock_guard<std::mutex> lock(mutex_);
  gnss_vec_.clear();
  droslog(LogLevel::INFO, "GnssInitor::Reset() -------");  
}

void GnssInitor::FeedData(const common::Data_Gnss &gnss) {
  if (gnss.gnss.rtk_type != common::RTK_NARROW_INT) 
    return;
  if (stage_.load() == 1) {
    std::lock_guard<std::mutex> lock(mutex_);
    gnss_vec_.push_back(gnss);
  }
}

void GnssInitor::StartInit() {
  droslog(LogLevel::INFO, "GnssInitor::StartInit() 开始GNSS初始化行走");
  stage_.store(1);
}

int GnssInitor::FinishInit() {
  droslog(LogLevel::INFO, "GnssInitor::FinishInit() 结束GNSS初始化行走, 计算GNSS航向角");
  
  std::lock_guard<std::mutex> lock(mutex_);
  if (gnss_vec_.size() < 2) {
    stage_.store(3);
    droslog(LogLevel::ERROR, "GnssInitor::FinishInit() GNSS初始化行走数据不足, size=%d", gnss_vec_.size());
    return 1;
  }

  common::Data_Gnss gnss1 = gnss_vec_[0];
  common::Data_Gnss gnss2 = gnss_vec_[gnss_vec_.size() - 1];

  init_dist_ = (gnss1.gnss.enu - gnss2.gnss.enu).norm();
  if (init_dist_ < 0.2) {
    stage_.store(3);
    droslog(LogLevel::ERROR, "GnssInitor::FinishInit() GNSS初始化行走距离不足, init_dist_=%.3f", init_dist_);
    return 2;
  }

  init_heading_ = get_yaw(gnss2.gnss.enu[0], gnss2.gnss.enu[1], gnss1.gnss.enu[0], gnss1.gnss.enu[1]);
  droslog(LogLevel::INFO, "GnssInitor::FinishInit() GNSS初始化航向角成功, 地理朝向: %.1f deg, dist=%.3f, enu:(%.3f, %.3f, %.3f)->(%.3f, %.3f, %.3f)", 
      init_heading_*180.0/M_PI, init_dist_, gnss1.gnss.enu[0], gnss1.gnss.enu[1], gnss1.gnss.enu[2], gnss2.gnss.enu[0], gnss2.gnss.enu[1], gnss2.gnss.enu[2]);

  stage_.store(2);
  return 0;
}

common::Data_Gnss GnssInitor::GetStartGnss() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (gnss_vec_.size() > 0) {
    return gnss_vec_[0];
  }
  return common::Data_Gnss();
}

common::Data_Gnss GnssInitor::GetEndGnss() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (gnss_vec_.size() > 0) {
    return gnss_vec_.back();
  }
  return common::Data_Gnss();
}