#include "common/vio_gnss_align.h"
#include "common/vio_tracker.h"

#include "common/common_def.h"
#include "common/log_filters.h"
#include "common/math_utils.h"
#include "common/sysutils.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"

namespace utils {

VioTracker::VioTracker() {
  droslog(LogLevel::INFO, "VioTracker::ctor() ++++++");
  std::lock_guard<std::mutex> lock(reset_mutex_);
  Init();
  droslog(LogLevel::INFO, "VioTracker::ctor() ------");
}

VioTracker::~VioTracker() {
  droslog(LogLevel::INFO, "VioTracker::dtor() ++++++");
  std::lock_guard<std::mutex> lock(reset_mutex_);
  Quit();
  droslog(LogLevel::INFO, "VioTracker::dtor() ------");
}

void VioTracker::Hello() {
  droslog(LogLevel::INFO, "VioTracker::Hello() ~");
}

void VioTracker::Reset() {
  droslog(LogLevel::INFO, "VioTracker::Reset() ++++++");
  std::lock_guard<std::mutex> lock(reset_mutex_);
  Quit();
  Init();
  droslog(LogLevel::INFO, "VioTracker::Reset() ------");
}

void VioTracker::Init() {
  droslog(LogLevel::INFO, "VioTracker::Init() ++++++");
  gnss_q_.reset(1024);      // 10hz, about 100s
  vio_q_.reset(1024);       // 15hz, about 100s
  vreloc_q_.reset(256);
  spa_node_q_.reset(64);

  stopped_.store(true);
  to_stop_.store(true);
  off_rtk_dist_.store(0.0);
  off_reloc_dist_.store(0.0);
  tf_pose_.ts = -1.0;
  pre_vio_.timestamp = -1.0;
  pre_vio_ts_ = 0.0;
  vio_local_pose_.timestamp = -1.0;
  tracker_thread_ = std::thread(&VioTracker::TrackerThread, this);
  while (to_stop_.load()) {
    Sleep(100);
    droslog(LogLevel::INFO, "VioTracker::Init() 等待 tracker_thread_ 线程启动...");
  }
  droslog(LogLevel::INFO, "VioTracker::Init() ------");
}

void VioTracker::Quit() {
  droslog(LogLevel::INFO, "VioTracker::Quit() ++++++");
  Sleep(100);
  to_stop_.store(true);
  while (!stopped_.load()) {
    to_stop_.store(true);
    Sleep(200);
    droslog(LogLevel::INFO, "VioTracker::Quit() 等待 tracker_thread_ 结束...");
  }
  if (tracker_thread_.joinable()) {
    tracker_thread_.join();
  }
  droslog(LogLevel::INFO, "VioTracker::Quit() ------");
}

bool VioTracker::IsVioValid(long long ck_dts) {
  return tf_pose_.ts > 0.0 && GetNow_Steady() < ck_dts * 1000 + pre_vio_ts_;
}

void VioTracker::DebugPrint() {
  droslog(LogLevel::WARN, "VioTracker::DebugPrint() pre_vio_ts=%lld, tf_pose_.ts=%.3f", pre_vio_.timestamp, tf_pose_.ts);
}

void VioTracker::SetParams(const VioTrackerParams &params) {
  params_ = params;
}

void VioTracker::InitAtStation(double ts) {
  droslog(LogLevel::INFO, "VioTracker::InitAtStation() 在桩初始化, ts=%.3f", ts);
  std::lock_guard<std::mutex> lock(tf_pose_mutex_);
  tf_pose_.ts = ts;
  tf_pose_.data.pos << 0.0, 0.0, 0.0;
  tf_pose_.data.quat = Eigen::Quaterniond::Identity();
}

void VioTracker::FeedData(const common::Data_Gnss &gnss_data) {
  common::Data_Gnss gnss = gnss_data;
  static double pre_ts = 0.0;
  if (gnss.timestamp <= pre_ts) {
    return;
  }
  // gnss.gnss.enu[2] = 0.0; // TODO 后续去掉这个, 暂时由于地图缺乏高度信息
  pre_ts = gnss.timestamp;

  // gnss.gnss.enu[2] = 0.0;
  double lat_sig = std::sqrt(gnss.gnss.cov(0,0) + 1e-12);
  double lon_sig = std::sqrt(gnss.gnss.cov(1,1) + 1e-12);
  double fix_info = params_.rtk_fix_info_sigma * params_.rtk_fix_info_sigma;
  double float_info = params_.rtk_float_info_sigma * params_.rtk_float_info_sigma;
  if (common::RTK_NARROW_INT == gnss.gnss.rtk_type) {
    if (lat_sig < params_.rtk_fix_ll_sigma || lon_sig < params_.rtk_fix_ll_sigma) {
      gnss.gnss.cov << fix_info, 0.0, 0.0,
                      0.0, fix_info, 0.0,
                      0.0, 0.0, fix_info/9.0; 
    } else {
      gnss.gnss.cov << float_info, 0.0, 0.0,
                      0.0, float_info, 0.0,
                      0.0, 0.0, float_info/9.0;
    }
  } else if (common::RTK_NARROW_FLOAT == gnss.gnss.rtk_type) {
    if (lat_sig < params_.rtk_float_ll_sigma || lon_sig < params_.rtk_float_ll_sigma) {
      gnss.gnss.cov << float_info, 0.0, 0.0,
                      0.0, float_info, 0.0,
                      0.0, 0.0, float_info/9.0; 
    }
    return;
  } else {
    return;
  }

  std::lock_guard<std::mutex> lock(gnss_q_mutex_);
  gnss_q_.emplace_back(gnss, gnss.timestamp);
}

void VioTracker::FeedData(const common::Data_VioResult &vio_result) {
  if (vio_result.timestamp <= 0.0 || vio_result.confidence <= 1) {
    static SimpleLogFilter log_filter(500);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::WARN, "VioTracker::FeedData(vio_result) invalid data, ts: %.3f, confidence: %d", vio_result.timestamp, vio_result.confidence);
    }
    return;
  }
  if (vio_result.timestamp <= pre_vio_.timestamp) {
    droslog(LogLevel::WARN, "VioTracker::FeedData(vio_result) 时间戳未单调递增, ts: %.3f, pre_ts: %.3f", vio_result.timestamp, pre_vio_.timestamp);
    return;
  }
  if (vio_result.timestamp > pre_vio_.timestamp + 1.0 && pre_vio_.timestamp > 0.0) {
    droslog(LogLevel::WARN, "VioTracker::FeedData(vio_result) 时间戳变化大于1s, ts: %.3f, pre_ts: %.3f", vio_result.timestamp, pre_vio_.timestamp);
  }

  double dist = (vio_result.vio.pos - pre_vio_.vio.pos).norm();
  off_rtk_dist_.store(off_rtk_dist_.load() + dist);
  off_reloc_dist_.store(off_reloc_dist_.load() + dist);

  pre_vio_ = vio_result;
  pre_vio_ts_ = GetNow_Steady();

  if (tf_pose_.ts > 0.0) {
    // 用于调试
    std::lock_guard<std::mutex> lock(tf_pose_mutex_);
    vio_local_pose_.pose.quat = tf_pose_.data.quat * vio_result.vio.q;
    vio_local_pose_.pose.pos = tf_pose_.data.pos + tf_pose_.data.quat * vio_result.vio.pos;
    vio_local_pose_.timestamp = vio_result.timestamp;
  }

  std::lock_guard<std::mutex> lock(vio_q_mutex_);
  vio_q_.emplace_back(vio_result, vio_result.timestamp);
}

void VioTracker::FeedVreloc(const common::Data_VioResult &vreloc) {
  if (vreloc.timestamp <= 0 || vreloc.confidence <= 1) {
    static SimpleLogFilter log_filter(500);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::WARN, "VioTracker::FeedData(vreloc) invalid data, ts: %.3f, confidence: %d", vreloc.timestamp, vreloc.confidence);
    }
    return;
  }

  static double pre_vreloc_ts = 0.0;
  if (vreloc.timestamp <= pre_vreloc_ts) {
    droslog(LogLevel::WARN, "VioTracker::FeedData(vreloc) 时间戳未单调递增, ts: %.3f, pre_ts: %.3f", vreloc.timestamp, pre_vreloc_ts);
    return;
  }
  pre_vreloc_ts = vreloc.timestamp;

  std::lock_guard<std::mutex> lock(vreloc_q_mutex_);
  vreloc_q_.emplace_back(vreloc, vreloc.timestamp);
}

common::Timed<common::Pose> VioTracker::GetVioTF() {
  std::lock_guard<std::mutex> lock(tf_pose_mutex_);
  return tf_pose_;
}

common::Data_VioResult VioTracker::GetVioInLocalXyz(const common::Data_VioResult &_vio_result) {
  common::Data_VioResult vio_result = _vio_result;
  // vio_result.vio.pos[2] = 0.0;
  common::Data_VioResult result = vio_result;
  result.timestamp = -1.0;
  {
    std::lock_guard<std::mutex> lock(tf_pose_mutex_);
    result.vio.q = tf_pose_.data.quat * vio_result.vio.q;
    result.vio.pos = tf_pose_.data.pos + tf_pose_.data.quat * vio_result.vio.pos;
  }

  const bool is_valid = IsVioValid();
  if (is_valid) {
    result.timestamp = vio_result.timestamp;
  } 
  {
    static SimpleLogFilter log_filter(4000);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::INFO, "VioTracker::GetVioInLocalXyz() VIO跟踪状态, VioValid=%d, dts: %.3f, tf_pose.ts: %.3f, tf_pose.pos=%.3f,%.3f,%.3f, local_vio.pos=%.3f, %.3f, %.3f", 
          is_valid, vio_result.timestamp-tf_pose_.ts, tf_pose_.ts, tf_pose_.data.pos[0], tf_pose_.data.pos[1], tf_pose_.data.pos[2], result.vio.pos[0], result.vio.pos[1], result.vio.pos[2]);
    }
  }
  return result;
}

void VioTracker::TrackerThread() {
  droslog(LogLevel::INFO, "VioTracker::TrackerThread() start+++");
  stopped_.store(false);
  to_stop_.store(false);

  Eigen::Vector3d pre_vio_pos = Eigen::Vector3d::Zero();
  common::Data_Pose pre_tf1, pre_tf2;
  while (!to_stop_.load()) {
    Sleep(30);

    // 对齐vio-gnss数据
    common::Data_VioResult cur_vio;
    {
      std::lock_guard<std::mutex> lock(vio_q_mutex_);
      if (vio_q_.size() > 7) {
        cur_vio = vio_q_[6];
      } else {
        continue;
      }
    }

    if ((cur_vio.vio.pos - pre_vio_pos).norm() < 0.10) {
      continue;
    }
    pre_vio_pos = cur_vio.vio.pos;

    SpaNode node;
    node.timestamp = cur_vio.timestamp;
    node.pose.pos = cur_vio.vio.pos;
    node.pose.quat = cur_vio.vio.q;

    {
      std::lock_guard<std::mutex> lock(tf_pose_mutex_);
      if (tf_pose_.ts > 0.0) {
        node.align_pose.quat = tf_pose_.data.quat * cur_vio.vio.q;
        node.align_pose.pos = tf_pose_.data.pos + tf_pose_.data.quat * cur_vio.vio.pos;
      }
    }

    bool has_gnss_cc = false;
    bool has_reloc_cc = false;
    // 查找时间同步的gnss
    {
      std::lock_guard<std::mutex> lock(gnss_q_mutex_);
      int idx = gnss_q_.findAfter(node.timestamp);
      if (idx > 0) {
        auto pre_gnss = gnss_q_[idx];
        auto next_gnss = gnss_q_[idx - 1];
        if (next_gnss.timestamp - pre_gnss.timestamp < 0.3) {
          node.gnss_ref = std::make_shared<common::ProbPose>();
          double alpha = (node.timestamp - pre_gnss.timestamp) / (next_gnss.timestamp - pre_gnss.timestamp);
          node.gnss_ref->pos = pre_gnss.gnss.enu * (1 - alpha) + next_gnss.gnss.enu * alpha;
          node.gnss_ref->pos_cov = pre_gnss.gnss.cov;

          has_gnss_cc = true;

          static SimpleLogFilter log_filter(4000);
          if (log_filter.Output(GetNow_Steady())) {
            droslog(LogLevel::INFO, "VioTracker::TrackerThread() 查找到对齐的gnss, ts=%.3f, vio=(%.3f,%.3f,%.3f), gnss=(%.3f,%.3f,%.3f)",
                node.timestamp, node.pose.pos[0], node.pose.pos[1], node.pose.pos[2], node.gnss_ref->pos[0], node.gnss_ref->pos[1], node.gnss_ref->pos[2]);
          }
        }
      }
    }
    // 查找vreloc
    {
      int idx = vreloc_q_.findAfter(node.timestamp);
      if (idx > 0) {
        auto pre_vreloc = vreloc_q_[idx];
        auto next_vreloc = vreloc_q_[idx - 1];
        if (next_vreloc.timestamp - pre_vreloc.timestamp < 0.5) {
          double pos_info = params_.reloc_info_pos_sigma * params_.reloc_info_pos_sigma;
          double quat_info = params_.reloc_info_quat_sigma * params_.reloc_info_quat_sigma;

          node.reloc_ref = std::make_shared<common::ProbPose>();
          double alpha = (node.timestamp - pre_vreloc.timestamp) / (next_vreloc.timestamp - pre_vreloc.timestamp);
          node.reloc_ref->pos = pre_vreloc.vio.pos * (1.0 - alpha) + next_vreloc.vio.pos * alpha;
          node.reloc_ref->quat = pre_vreloc.vio.q.slerp(alpha, next_vreloc.vio.q);
          node.reloc_ref->quat.normalize();

          node.reloc_ref->pos_cov << pos_info, 0.0, 0.0,  0.0, pos_info, 0.0,  0.0, 0.0, pos_info;
          node.reloc_ref->quat_cov << quat_info, 0.0, 0.0,  0.0, quat_info, 0.0,  0.0, 0.0, quat_info;

          has_reloc_cc = true;

          auto rpy = GetEulerRPY(node.pose.quat);
          auto reloc_rpy = GetEulerRPY(node.reloc_ref->quat);
          droslog(LogLevel::INFO, "VioTracker::TrackerThread() 查找到对齐的reloc, ts=%.3f, vio=(%.3f,%.3f,%.3f; %.3f,%.3f,%.3f), reloc=(%.3f,%.3f,%.3f; %.3f,%.3f,%.3f)",
              node.timestamp, node.pose.pos[0], node.pose.pos[1], node.pose.pos[2], rpy[0], rpy[1], rpy[2],
              node.reloc_ref->pos[0], node.reloc_ref->pos[1], node.reloc_ref->pos[2], reloc_rpy[0], reloc_rpy[1], reloc_rpy[2]);
        }
      }
    }

    spa_node_q_.emplace_back(node, node.timestamp);

    if (spa_node_q_.size() < 31 || (!has_gnss_cc && !has_reloc_cc))
      continue;

    // 定时估计转换, 每秒一次
    int valid_vreloc_cnt = 0;
    static SimpleLogFilter fps_filter(1000);
    if (fps_filter.Output(GetNow_Steady())) {
      int valid_cc_cnt = 0;
      std::vector<SpaNode> node_vec;
      int window_size = spa_node_q_.size();
      for (int i = 0; i < window_size; i++) {
        node_vec.push_back(spa_node_q_[i]);
        if (spa_node_q_[i].gnss_ref.get() || spa_node_q_[i].reloc_ref.get()) {
          valid_cc_cnt++;
        }
      }

      SpaConfig spa_config;
      spa_config.use_align_pose = true;
      spa_config.pose_adj_factor = params_.pose_adj_factor;
      spa_config.pose_align_factor = params_.pose_align_factor;
      spa_config.pose_rp_factor = params_.pose_rp_factor;
      if (tf_pose_.ts > 0.0 || valid_cc_cnt > spa_node_q_.size() / 2) {
        if (tf_pose_.ts <= 0.0) {
          spa_config.use_align_pose = false;
        }

        auto ts1 = GetNow_Steady();
        common::Data_Pose tf_pose = spa_align(node_vec, spa_config);
        auto ts2 = GetNow_Steady();
  
        if (tf_pose.timestamp <= 0.0) {
          droslog(LogLevel::WARN, "VioTracker::TrackerThread() Align failed......use_time=%lld ms, cc_cnt=%d", ts2-ts1, valid_cc_cnt);
        } else {
          auto new_rpy = GetEulerRPY(tf_pose.pose.quat);
          auto pre_rpy = GetEulerRPY(tf_pose_.data.quat);
          if (std::abs(new_rpy[0]) > 0.3 || std::abs(new_rpy[1]) > 0.3 || std::abs(new_rpy[0]) + std::abs(new_rpy[1]) > 0.5) {
            droslog(LogLevel::WARN, "VioTracker::TrackerThread() -----------对齐异常---------- rpy=(%.3f, %.3f, %.3f)", new_rpy[0], new_rpy[1], new_rpy[2]);
          } else {
            auto new_ap = tf_pose.pose.pos;
            auto pre_ap = tf_pose_.data.pos;
    
            double dpos = (new_ap - pre_ap).norm();
            double dyaw = KeepAngleInPI(new_rpy[2] - pre_rpy[2]);
            
            droslog(LogLevel::INFO, "VioTracker::TrackerThread() 新的对齐结果: use_time=%lld ms, dpos=%.3f, dyaw=%.3f, old_tf->new_tf:(%.3f,%.3f,%.3f; %.3f,%.3f,%.3f)->(%.3f,%.3f,%.3f; %.3f,%.3f,%.3f)",
                ts2-ts1, dpos, dyaw, pre_ap[0], pre_ap[1], pre_ap[2], pre_rpy[0], pre_rpy[1], pre_rpy[2],
                new_ap[0], new_ap[1], new_ap[2], new_rpy[0], new_rpy[1], new_rpy[2]);
            
            if (tf_pose_.ts <= 0.0) {
              // 首次对齐
              droslog(LogLevel::INFO, "VioTracker::TrackerThread() 首次计算对齐: 更新spa_node 队列");
              for (int i=0; i<spa_node_q_.size(); i++) {
                spa_node_q_[i].align_pose.quat = tf_pose.pose.quat * spa_node_q_[i].pose.quat;
                spa_node_q_[i].align_pose.pos = tf_pose.pose.pos + tf_pose.pose.quat * spa_node_q_[i].pose.pos;
              }
            }
  
            off_rtk_dist_.store(0.0);
            off_reloc_dist_.store(0.0);
  
            std::lock_guard<std::mutex> lock(tf_pose_mutex_);
            tf_pose_.ts = tf_pose.timestamp;
            tf_pose_.data = tf_pose.pose;
          }
        }
      }
    }
  }
  stopped_.store(true);
  droslog(LogLevel::INFO, "VioTracker::TrackerThread() stop---");
}

} // namespace utils