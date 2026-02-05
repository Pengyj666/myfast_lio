#include "common/vio_gnss_align.h"
#include "common/loc_tracker.h"

#include "common/common_def.h"
#include "common/debug_client.h"
#include "common/log_filters.h"
#include "common/math_utils.h"
#include "common/sysutils.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"

namespace utils {

LocTracker::LocTracker() {
  droslog(LogLevel::INFO, "LocTracker::ctor() ++++++");
  std::lock_guard<std::mutex> lock(reset_mutex_);
  Init();
  droslog(LogLevel::INFO, "LocTracker::ctor() ------");
}

LocTracker::~LocTracker() {
  droslog(LogLevel::INFO, "LocTracker::dtor() ++++++");
  std::lock_guard<std::mutex> lock(reset_mutex_);
  Quit();
  droslog(LogLevel::INFO, "LocTracker::dtor() ------");
}

void LocTracker::Hello() {
  droslog(LogLevel::INFO, "LocTracker::Hello() ~");
}

void LocTracker::Reset() {
  droslog(LogLevel::INFO, "LocTracker::Reset() ++++++");
  std::lock_guard<std::mutex> lock(reset_mutex_);
  Quit();
  Init();
  droslog(LogLevel::INFO, "LocTracker::Reset() ------");
}

void LocTracker::Init() {
  droslog(LogLevel::INFO, "LocTracker::Init() ++++++");
  gnss_q_.reset(1024);      // 10hz, about 100s
  pose_q_.reset(1024);      // 15hz, about 100s
  reloc_q_.reset(256);
  spa_node_q_.reset(64);

  stopped_.store(true);
  to_stop_.store(true);
  off_rtk_dist_.store(0.0);
  off_reloc_dist_.store(0.0);
  tf_pose_.ts = -1.0;
  pre_pose_.timestamp = -1.0;
  pre_pose_ts_ = 0.0;
  local_pose_.timestamp = -1.0;
  tracker_thread_ = std::thread(&LocTracker::TrackerThread, this);
  while (to_stop_.load()) {
    Sleep(100);
    droslog(LogLevel::INFO, "LocTracker::Init() 等待 tracker_thread_ 线程启动...");
  }
  droslog(LogLevel::INFO, "LocTracker::Init() ------");
}

void LocTracker::Quit() {
  droslog(LogLevel::INFO, "LocTracker::Quit() ++++++");
  Sleep(100);
  to_stop_.store(true);
  while (!stopped_.load()) {
    to_stop_.store(true);
    Sleep(200);
    droslog(LogLevel::INFO, "LocTracker::Quit() 等待 tracker_thread_ 结束...");
  }
  if (tracker_thread_.joinable()) {
    tracker_thread_.join();
  }
  droslog(LogLevel::INFO, "LocTracker::Quit() ------");
}

bool LocTracker::IsLocValid(long long ck_dts) {
  return tf_pose_.ts > 0.0 && GetNow_Steady() < ck_dts * 1000 + pre_pose_ts_;
}

void LocTracker::DebugPrint() {
  droslog(LogLevel::WARN, "LocTracker::DebugPrint() pre_pose_ts=%lld, tf_pose_.ts=%.3f", pre_pose_.timestamp, tf_pose_.ts);
}

void LocTracker::SetParams(const LocTrackerParams &params) {
  params_ = params;
}

void LocTracker::InitAtStation(double ts) {
  droslog(LogLevel::INFO, "LocTracker::InitAtStation() 在桩初始化, ts=%.3f", ts);
  std::lock_guard<std::mutex> lock(tf_pose_mutex_);
  tf_pose_.ts = ts;
  tf_pose_.data.pos << 0.0, 0.0, 0.0;
  tf_pose_.data.quat = Eigen::Quaterniond::Identity();
}

void LocTracker::FeedGnss(const common::Data_Gnss &gnss_data) {
  common::Data_Gnss gnss = gnss_data;
  static double pre_ts = 0.0;
  if (gnss.timestamp <= pre_ts) {
    return;
  }
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

void LocTracker::FeedPose(const common::Data_ProbPose &pose) {
  if (pose.timestamp <= 0.0) {
    static SimpleLogFilter log_filter(500);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::WARN, "LocTracker::FeedData(pose) invalid data, ts: %.3f", pose.timestamp);
    }
    return;
  }
  if (pose.timestamp <= pre_pose_.timestamp) {
    droslog(LogLevel::WARN, "LocTracker::FeedData(pose) 时间戳未单调递增, ts: %.3f, pre_ts: %.3f", pose.timestamp, pre_pose_.timestamp);
    return;
  }
  if (pose.timestamp > pre_pose_.timestamp + 1.0 && pre_pose_.timestamp > 0.0) {
    droslog(LogLevel::WARN, "LocTracker::FeedData(pose) 时间戳变化大于1s, ts: %.3f, pre_ts: %.3f", pose.timestamp, pre_pose_.timestamp);
  }

  double dist = (pose.ppose.pos - pre_pose_.ppose.pos).norm();
  off_rtk_dist_.store(off_rtk_dist_.load() + dist);
  off_reloc_dist_.store(off_reloc_dist_.load() + dist);

  pre_pose_ = pose;
  pre_pose_ts_ = GetNow_Steady();

  if (tf_pose_.ts > 0.0) {
    // 用于调试
    std::lock_guard<std::mutex> lock(tf_pose_mutex_);
    local_pose_.pose.quat = tf_pose_.data.quat * pose.ppose.quat;
    local_pose_.pose.pos = tf_pose_.data.pos + tf_pose_.data.quat * pose.ppose.pos;
    local_pose_.timestamp = pose.timestamp;
  }

  std::lock_guard<std::mutex> lock(pose_q_mutex_);
  pose_q_.emplace_back(pose, pose.timestamp);
}

void LocTracker::FeedReloc(const common::Data_ProbPose &reloc) {
  if (reloc.timestamp <= 0) {
    static SimpleLogFilter log_filter(500);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::WARN, "LocTracker::FeedData(reloc) invalid data, ts: %.3f", reloc.timestamp);
    }
    return;
  }

  static double pre_reloc_ts = 0.0;
  if (reloc.timestamp <= pre_reloc_ts) {
    droslog(LogLevel::WARN, "LocTracker::FeedData(reloc) 时间戳未单调递增, ts: %.3f, pre_ts: %.3f", reloc.timestamp, pre_reloc_ts);
    return;
  }
  pre_reloc_ts = reloc.timestamp;

  std::lock_guard<std::mutex> lock(reloc_q_mutex_);
  reloc_q_.emplace_back(reloc, reloc.timestamp);
}

common::Timed<common::Pose> LocTracker::GetTF() {
  std::lock_guard<std::mutex> lock(tf_pose_mutex_);
  return tf_pose_;
}

common::Data_ProbPose LocTracker::GetPoseInLocalXyz(const common::Data_ProbPose &_pose) {
  common::Data_ProbPose res_pose = _pose;
  common::Data_ProbPose result = res_pose;
  result.timestamp = -1.0;
  {
    std::lock_guard<std::mutex> lock(tf_pose_mutex_);
    result.ppose.quat = tf_pose_.data.quat * res_pose.ppose.quat;
    result.ppose.pos = tf_pose_.data.pos + tf_pose_.data.quat * res_pose.ppose.pos;
  }

  const bool is_valid = IsLocValid();
  if (is_valid) {
    result.timestamp = res_pose.timestamp;
  } 
  {
    static SimpleLogFilter log_filter(4000);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::INFO, "LocTracker::GetPoseInLocalXyz() 跟踪状态, LocValid=%d, dts: %.3f, tf_pose.ts: %.3f, tf_pose.pos=%.3f,%.3f,%.3f, local_pose.pos=%.3f, %.3f, %.3f", 
          is_valid, res_pose.timestamp-tf_pose_.ts, tf_pose_.ts, tf_pose_.data.pos[0], tf_pose_.data.pos[1], tf_pose_.data.pos[2], result.ppose.pos[0], result.ppose.pos[1], result.ppose.pos[2]);
    }
  }
  return result;
}

void LocTracker::TrackerThread() {
  droslog(LogLevel::INFO, "LocTracker::TrackerThread() start+++");
  stopped_.store(false);
  to_stop_.store(false);

  double pre_pose_ts = 0.0;
  Eigen::Vector3d pre_pos = Eigen::Vector3d::Zero();
  Eigen::Vector3d pre_reloc_pos = Eigen::Vector3d::Zero();
  common::Data_Pose pre_tf1, pre_tf2;
  while (!to_stop_.load()) {
    Sleep(30);

    bool used_align = DebugClient::Instance()->GetUseAlign();

    {
      static SimpleLogFilter log_filter(10000);
      if (log_filter.Output(GetNow_Steady())) {
        droslog(LogLevel::INFO, "LocTracker::TrackerThread() 调试log, pose_q.size()=%d, use_align=%d", pose_q_.size(), used_align);
      }
    }

    common::Data_ProbPose cur_pose;
    {
      std::lock_guard<std::mutex> lock(pose_q_mutex_);
      if (pose_q_.size() > 7) {
        cur_pose = pose_q_[6];
      } else {
        continue;
      }
    }

    if (cur_pose.timestamp < pre_pose_ts + 0.05) {
      continue;
    }
    pre_pose_ts = cur_pose.timestamp;

    if (!used_align) {
      continue;
    }

    SpaNode node;
    node.timestamp = cur_pose.timestamp;
    node.pose.pos = cur_pose.ppose.pos;
    node.pose.quat = cur_pose.ppose.quat;

    {
      std::lock_guard<std::mutex> lock(tf_pose_mutex_);
      if (tf_pose_.ts > 0.0) {
        node.align_pose.quat = tf_pose_.data.quat * cur_pose.ppose.quat;
        node.align_pose.pos = tf_pose_.data.pos + tf_pose_.data.quat * cur_pose.ppose.pos;
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
            droslog(LogLevel::INFO, "LocTracker::TrackerThread() 查找到对齐的gnss, ts=%.3f, pose=(%.3f,%.3f,%.3f), gnss=(%.3f,%.3f,%.3f)",
                node.timestamp, node.pose.pos[0], node.pose.pos[1], node.pose.pos[2], node.gnss_ref->pos[0], node.gnss_ref->pos[1], node.gnss_ref->pos[2]);
          }
        }
      }
    }
    // 查找reloc
    {
      int idx = reloc_q_.findAfter(node.timestamp);
      if (idx > 0) {
        auto pre_reloc = reloc_q_[idx];
        auto next_reloc = reloc_q_[idx - 1];
        if (next_reloc.timestamp - pre_reloc.timestamp < 0.4) {
          double pos_info = params_.reloc_info_pos_sigma * params_.reloc_info_pos_sigma;
          double quat_info = params_.reloc_info_quat_sigma * params_.reloc_info_quat_sigma;

          node.reloc_ref = std::make_shared<common::ProbPose>();
          double alpha = (node.timestamp - pre_reloc.timestamp) / (next_reloc.timestamp - pre_reloc.timestamp);
          node.reloc_ref->pos = pre_reloc.ppose.pos * (1.0 - alpha) + next_reloc.ppose.pos * alpha;
          node.reloc_ref->quat = pre_reloc.ppose.quat.slerp(alpha, next_reloc.ppose.quat);
          node.reloc_ref->quat.normalize();

          node.reloc_ref->pos_cov << pos_info, 0.0, 0.0,  0.0, pos_info, 0.0,  0.0, 0.0, pos_info;
          node.reloc_ref->quat_cov << quat_info, 0.0, 0.0,  0.0, quat_info, 0.0,  0.0, 0.0, quat_info;

          has_reloc_cc = true;

          auto rpy = GetEulerRPY(node.pose.quat);
          auto reloc_rpy = GetEulerRPY(node.reloc_ref->quat);
          droslog(LogLevel::INFO, "LocTracker::TrackerThread() 查找到对齐的reloc(1), ts=%.3f, ppose=(%.3f,%.3f,%.3f; %.3f,%.3f,%.3f), reloc=(%.3f,%.3f,%.3f; %.3f,%.3f,%.3f)",
              node.timestamp, node.pose.pos[0], node.pose.pos[1], node.pose.pos[2], rpy[0], rpy[1], rpy[2],
              node.reloc_ref->pos[0], node.reloc_ref->pos[1], node.reloc_ref->pos[2], reloc_rpy[0], reloc_rpy[1], reloc_rpy[2]);
        } else if (std::abs(node.timestamp - pre_reloc.timestamp < 0.05)) {
          double pos_info = params_.reloc_info_pos_sigma * params_.reloc_info_pos_sigma;
          double quat_info = params_.reloc_info_quat_sigma * params_.reloc_info_quat_sigma;

          node.reloc_ref = std::make_shared<common::ProbPose>();
          node.reloc_ref->pos = pre_reloc.ppose.pos;
          node.reloc_ref->quat = pre_reloc.ppose.quat;

          node.reloc_ref->pos_cov << pos_info, 0.0, 0.0,  0.0, pos_info, 0.0,  0.0, 0.0, pos_info;
          node.reloc_ref->quat_cov << quat_info, 0.0, 0.0,  0.0, quat_info, 0.0,  0.0, 0.0, quat_info;

          has_reloc_cc = true;

          auto rpy = GetEulerRPY(node.pose.quat);
          auto reloc_rpy = GetEulerRPY(node.reloc_ref->quat);
          droslog(LogLevel::INFO, "LocTracker::TrackerThread() 查找到对齐的reloc(2), ts=%.3f, ppose=(%.3f,%.3f,%.3f; %.3f,%.3f,%.3f), reloc=(%.3f,%.3f,%.3f; %.3f,%.3f,%.3f)",
              node.timestamp, node.pose.pos[0], node.pose.pos[1], node.pose.pos[2], rpy[0], rpy[1], rpy[2],
              node.reloc_ref->pos[0], node.reloc_ref->pos[1], node.reloc_ref->pos[2], reloc_rpy[0], reloc_rpy[1], reloc_rpy[2]);
        } else if (std::abs(node.timestamp - next_reloc.timestamp < 0.05)) {
          double pos_info = params_.reloc_info_pos_sigma * params_.reloc_info_pos_sigma;
          double quat_info = params_.reloc_info_quat_sigma * params_.reloc_info_quat_sigma;

          node.reloc_ref = std::make_shared<common::ProbPose>();
          node.reloc_ref->pos = next_reloc.ppose.pos;
          node.reloc_ref->quat = next_reloc.ppose.quat;

          node.reloc_ref->pos_cov << pos_info, 0.0, 0.0,  0.0, pos_info, 0.0,  0.0, 0.0, pos_info;
          node.reloc_ref->quat_cov << quat_info, 0.0, 0.0,  0.0, quat_info, 0.0,  0.0, 0.0, quat_info;

          has_reloc_cc = true;

          auto rpy = GetEulerRPY(node.pose.quat);
          auto reloc_rpy = GetEulerRPY(node.reloc_ref->quat);
          droslog(LogLevel::INFO, "LocTracker::TrackerThread() 查找到对齐的reloc(3), ts=%.3f, ppose=(%.3f,%.3f,%.3f; %.3f,%.3f,%.3f), reloc=(%.3f,%.3f,%.3f; %.3f,%.3f,%.3f)",
              node.timestamp, node.pose.pos[0], node.pose.pos[1], node.pose.pos[2], rpy[0], rpy[1], rpy[2],
              node.reloc_ref->pos[0], node.reloc_ref->pos[1], node.reloc_ref->pos[2], reloc_rpy[0], reloc_rpy[1], reloc_rpy[2]);
        }
      }
    }

    bool new_node = false;
    if (has_reloc_cc) {
      // 有reloc_cc时, 间隔0.05m插入一个node: 与上一个node的相对位置差大于0.05m, 与上一个cc_node的相对位置差大于0.1m
      if ((cur_pose.ppose.pos - pre_pos).norm() > 0.10 && (cur_pose.ppose.pos - pre_reloc_pos).norm() > 0.20) {
        spa_node_q_.emplace_back(node, node.timestamp);
        pre_pos = cur_pose.ppose.pos;
        pre_reloc_pos = cur_pose.ppose.pos;
        new_node = true;
      }
    } else {
      // 没有reloc_cc时, 间隔0.15m插入一个node
      if ((cur_pose.ppose.pos - pre_pos).norm() > 0.15) {
        spa_node_q_.emplace_back(node, node.timestamp);
        pre_pos = cur_pose.ppose.pos;
        new_node = true;
      }
    }

    if (!new_node || spa_node_q_.size() < 31 || (!has_gnss_cc && !has_reloc_cc))
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
          droslog(LogLevel::WARN, "LocTracker::TrackerThread() Align failed......use_time=%lld ms, cc_cnt=%d", ts2-ts1, valid_cc_cnt);
        } else {
          auto new_rpy = GetEulerRPY(tf_pose.pose.quat);
          auto pre_rpy = GetEulerRPY(tf_pose_.data.quat);
          if (std::abs(new_rpy[0]) > 0.3 || std::abs(new_rpy[1]) > 0.3 || std::abs(new_rpy[0]) + std::abs(new_rpy[1]) > 0.5) {
            droslog(LogLevel::WARN, "LocTracker::TrackerThread() -----------对齐异常---------- rpy=(%.3f, %.3f, %.3f)", new_rpy[0], new_rpy[1], new_rpy[2]);
          } else {
            auto new_ap = tf_pose.pose.pos;
            auto pre_ap = tf_pose_.data.pos;
    
            double dpos = (new_ap - pre_ap).norm();
            double dyaw = KeepAngleInPI(new_rpy[2] - pre_rpy[2]);
            
            droslog(LogLevel::INFO, "LocTracker::TrackerThread() 新的对齐结果: use_time=%lld ms, dpos=%.3f, dyaw=%.3f, old_tf->new_tf:(%.3f,%.3f,%.3f; %.3f,%.3f,%.3f)->(%.3f,%.3f,%.3f; %.3f,%.3f,%.3f)",
                ts2-ts1, dpos, dyaw, pre_ap[0], pre_ap[1], pre_ap[2], pre_rpy[0], pre_rpy[1], pre_rpy[2],
                new_ap[0], new_ap[1], new_ap[2], new_rpy[0], new_rpy[1], new_rpy[2]);
            
            if (tf_pose_.ts <= 0.0) {
              // 首次对齐
              droslog(LogLevel::INFO, "LocTracker::TrackerThread() 首次计算对齐: 更新spa_node 队列");
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
      } else {
        static SimpleLogFilter log_filter(5000);
        if (log_filter.Output(GetNow_Steady())) {
          droslog(LogLevel::INFO, "LocTracker::TrackerThread() tf_pose_.ts=%.3f, valid_cc_cnt=%d", tf_pose_.ts, valid_cc_cnt);
        }
      }
    }
  }
  stopped_.store(true);
  droslog(LogLevel::INFO, "LocTracker::TrackerThread() stop---");
}

} // namespace utils