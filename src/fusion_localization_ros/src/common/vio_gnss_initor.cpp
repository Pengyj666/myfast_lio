#include "common/vio_gnss_initor.h"

#include "common/common_def.h"
#include "common/log_filters.h"
#include "common/math_utils.h"
#include "common/sysutils.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"
#include "common/vio_gnss_align.h"

namespace utils {

VioGnssInitor::VioGnssInitor() :is_init_at_station_(false), is_need_init_(false) {
  droslog(LogLevel::INFO, "VioGnssInitor::ctor() ++++++");
  std::lock_guard<std::mutex> lock(reset_mutex_);
  Init();
  droslog(LogLevel::INFO, "VioGnssInitor::ctor() ------");
}

VioGnssInitor::~VioGnssInitor() {
  droslog(LogLevel::INFO, "VioGnssInitor::dtor() ++++++");
  std::lock_guard<std::mutex> lock(reset_mutex_);
  Quit();
  droslog(LogLevel::INFO, "VioGnssInitor::dtor() ------");
}

void VioGnssInitor::Hello() {
  droslog(LogLevel::INFO, "VioGnssInitor::Hello() ~");
}

void VioGnssInitor::Reset() {
  droslog(LogLevel::INFO, "VioGnssInitor::Reset() ++++++");
  std::lock_guard<std::mutex> lock(reset_mutex_);
  Quit();
  Init();
  droslog(LogLevel::INFO, "VioGnssInitor::Reset() ------");
}

void VioGnssInitor::Init() {
  droslog(LogLevel::INFO, "VioGnssInitor::Init() ++++++");
  gnss_q_.reset(128);             // 10hz, about 100s
  pose_q_.reset(64);              // 15hz, about 100s
  spa_node_q_.reset(96);

  stopped_.store(true);
  to_stop_.store(true);
  is_init_at_station_.store(false);
  is_need_init_.store(false);
  
  gnss_map_offset_.timestamp = -1.0;
  tracker_thread_ = std::thread(&VioGnssInitor::TrackerThread, this);
  while (to_stop_.load()) {
    Sleep(100);
    droslog(LogLevel::INFO, "VioGnssInitor::Init() 等待 thread_ 线程启动...");
  }
  droslog(LogLevel::INFO, "VioGnssInitor::Init() ------");
}

void VioGnssInitor::Quit() {
  droslog(LogLevel::INFO, "VioGnssInitor::Quit() ++++++");
  to_stop_.store(true);
  while (!stopped_.load()) {
    to_stop_.store(true);
    Sleep(100);
    droslog(LogLevel::INFO, "VioGnssInitor::Quit() 等待 thread_ 结束...");
  }
  if (tracker_thread_.joinable()) {
    tracker_thread_.join();
  }
  droslog(LogLevel::INFO, "VioGnssInitor::Quit() ------");
}

void VioGnssInitor::InitAtStation(double ts) {
  is_init_at_station_.store(true);
  is_need_init_.store(true);
  droslog(LogLevel::INFO, "VioGnssInitor::InitAtStation() 在桩启动vio-gnss-initor, ts=%.3f", ts);
}

void VioGnssInitor::StopInit() {
  is_need_init_.store(false);
  droslog(LogLevel::INFO, "VioGnssInitor::StopInit() 外部调用停止vio-gnss-initor");
}

void VioGnssInitor::FeedGnss(const common::Data_Gnss &gnss) {
  if (IsGnssMapOffsetValid()) {
    return; // 已经初始化过了
  }
  static double pre_ts = 0.0;
  if (gnss.timestamp <= pre_ts || gnss.gnss.rtk_type != common::RTK_NARROW_INT) {
    return;
  }
  pre_ts = gnss.timestamp;
  
  std::lock_guard<std::mutex> lock(gnss_q_mutex_);
  gnss_q_.emplace_back(gnss, gnss.timestamp);
}

void VioGnssInitor::FeedVio(const common::Data_VioResult &vio_result) {
  if (IsGnssMapOffsetValid()) {
    return; // 已经初始化过了
  }
  
  static common::Data_VioResult pre_vio;
  if (vio_result.timestamp > pre_vio.timestamp + 1.0 && pre_vio.timestamp > 0.0) {
    droslog(LogLevel::WARN, "VioGnssInitor::FeedVioAndConvertToLocalPos() vio时间戳变化大于1s, ts: %.3f, pre_ts: %.3f", vio_result.timestamp, pre_vio.timestamp);
  }
  pre_vio = vio_result;

  common::Data_Pose vio_pose;
  vio_pose.timestamp = vio_result.timestamp;
  vio_pose.pose.pos = vio_result.vio.pos;
  vio_pose.pose.quat = vio_result.vio.q;
  
  std::lock_guard<std::mutex> lock(pose_q_mutex_);
  pose_q_.emplace_back(vio_pose, vio_pose.timestamp);
}

common::Data_Pose VioGnssInitor::GetGnssMapOffset() {
  return gnss_map_offset_;
}

common::Data_Gnss VioGnssInitor::GetGnssByTime(const double &ts) {
  common::Data_Gnss gnss;
  gnss.timestamp = -1.0;
  std::lock_guard<std::mutex> lock(gnss_q_mutex_);
  int idx = gnss_q_.findAfter(ts);
  if (idx > 0) {
    auto pre = gnss_q_[idx];
    auto next = gnss_q_[idx - 1];
    if (next.timestamp - pre.timestamp < 0.3) {
      gnss = next;
      gnss.timestamp = ts;
      gnss.gnss.enu = pre.gnss.enu + (next.gnss.enu - pre.gnss.enu) * (ts - pre.timestamp) / (next.timestamp - pre.timestamp);
    }
  }
  return gnss;
}

void VioGnssInitor::TrackerThread() {
  droslog(LogLevel::INFO, "VioGnssInitor::TrackerThread() start+++");
  stopped_.store(false);
  to_stop_.store(false);

  Eigen::Vector3d pre_vio_pos(-100.0, -100.0, -100.0);
  while (!to_stop_.load()) {
    Sleep(30);

    if (IsGnssMapOffsetValid() || !is_init_at_station_.load() || !is_need_init_.load()) {
      Sleep(1000);
      continue;
    }

    // 对齐vio-gnss数据
    common::Data_Pose cur_pose;
    {
      std::lock_guard<std::mutex> lock(pose_q_mutex_);
      if (pose_q_.size() > 7) {
        cur_pose = pose_q_[6];
      } else {
        continue;
      }
    }

    if ((cur_pose.pose.pos - pre_vio_pos).norm() < 0.05) {
      continue;
    }
    pre_vio_pos = cur_pose.pose.pos;

    SpaNode node;
    node.timestamp = cur_pose.timestamp;
    node.pose.pos = cur_pose.pose.pos;
    node.pose.quat = cur_pose.pose.quat;

    auto gnss = GetGnssByTime(cur_pose.timestamp);
    if (gnss.timestamp > 0.0) {
      node.gnss_ref = std::make_shared<common::ProbPose>();
      node.gnss_ref->pos = gnss.gnss.enu;
    }
    spa_node_q_.emplace_back(node, node.timestamp);

    // 定时估计转换, 每秒一次
    static SimpleLogFilter fps_filter(1000);
    if (fps_filter.Output(GetNow_Steady())) {
      int q_size = spa_node_q_.size();
      
      int gnss_cnt = 0;
      std::vector<SpaNode> node_vec;
      for (int i = 0; i < q_size; i++) {
        node_vec.push_back(spa_node_q_[i]);
        if (spa_node_q_[i].gnss_ref.get()) {
          gnss_cnt++;
        }
      }

      droslog(LogLevel::INFO, "VioGnssInitor::TrackerThread() VIO下桩, 计算gnss转换中..., q_size: %d, gnss_cnt: %d", q_size, gnss_cnt);

      if (gnss_cnt < 64) {
        continue;
      }

      auto map_offset_tf = vio_gnss_init(node_vec);
      if (map_offset_tf.timestamp > 0.0) {
        gnss_map_offset_ = map_offset_tf;
        droslog(LogLevel::INFO, "VioGnssInitor::TrackerThread() VIO下桩, 计算gnss转换成功");
      } else {
        
      }
    }
  }
  stopped_.store(true);
  droslog(LogLevel::INFO, "VioGnssInitor::TrackerThread() stop---");
}

} // namespace utils