#include "vreloc_tracker.h"

#include "common/log_filters.h"
#include "common/math_utils.h"
#include "common/sysutils.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"
#include "spa_align.h"

using namespace utils;
using namespace common;

VrelocTracker::VrelocTracker() {    // 重定位跟踪器
  droslog(LogLevel::INFO, "VrelocTracker::ctor() ++++++");
  Init();
  droslog(LogLevel::INFO, "VrelocTracker::ctor() ------");
}

VrelocTracker::~VrelocTracker() {   // 重定位跟踪器析构
  droslog(LogLevel::INFO, "VrelocTracker::dtor() ++++++");
  Quit();
  droslog(LogLevel::INFO, "VrelocTracker::dtor() ------");
}

void VrelocTracker::Reset() {        // 重定位跟踪器重置
  droslog(LogLevel::INFO, "VrelocTracker::Reset() ++++++");
  Quit();
  Init();
  droslog(LogLevel::INFO, "VrelocTracker::Reset() ------");
}

void VrelocTracker::Init() {         // 重定位跟踪器初始化
  droslog(LogLevel::INFO, "VrelocTracker::Init() ++++++");
  vio_q_.reset(512);       // 6hz, about 100s
  vreloc_q_.reset(256);
  align_window_.reset(64);

  stopped_.store(true);
  to_stop_.store(true);

  {
    std::lock_guard<std::mutex> lock(tf_pose_mutex_);
    filtered_tf_pose_.ts = -1.0;
    filtered_tf_pose_.data = Pose();
  }

  {
    std::lock_guard<std::mutex> lock(acc_odom_mutex_);
    acc_odom_.dist = 0.0;
    acc_odom_.angle = 0.0;
  }

  tracker_thread_ = std::thread(&VrelocTracker::TrackerThread, this);
  droslog(LogLevel::INFO, "VrelocTracker::Init() ------");
}

void VrelocTracker::Quit() {         // 重定位跟踪器退出
  droslog(LogLevel::INFO, "VrelocTracker::Quit() ++++++");
  to_stop_.store(true);
  if (tracker_thread_.joinable()) {
    tracker_thread_.join();
  }
  droslog(LogLevel::INFO, "VrelocTracker::Quit() ------");
}

void VrelocTracker::SetParams(const Config &config) {  // 设置重定位跟踪器参数
  config_ = config;
}

static long long s_pre_vio_ts = 0;
bool VrelocTracker::IsVioValid() {                    // 判断vio是否有效
  return acc_odom_.dist < config_.tf_valid_dist && filtered_tf_pose_.ts > 0;
}

bool VrelocTracker::IsTFValid() {                    // 判断tf是否有效
  return filtered_tf_pose_.ts > 0;
}

void VrelocTracker::DebugPrint() {                   // 打印重定位跟踪器调试信息
  auto cur_ts = GetNow_Steady();
  droslog(LogLevel::WARN, "VrelocTracker::DebugPrint() cur_ts=%lld, s_pre_vio_ts=%lld, filtered_tf_pose_.ts=%.3f, acc_odom:(%.3f,%.3f)", 
      cur_ts, s_pre_vio_ts, filtered_tf_pose_.ts, acc_odom_.dist, acc_odom_.angle);
}

void VrelocTracker::FeedData(const common::Data_ProbPose &vio) {  // 喂入vio数据
  if (vio.timestamp <= pre_vio_.timestamp) {
    static SimpleLogFilter log_filter(500);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::WARN, "VrelocTracker::FeedData(vio) 时间戳未单调递增, ts: %.3f, pre_ts: %.3f", vio.timestamp, pre_vio_.timestamp);
    }
    return;
  }
  if (vio.timestamp > pre_vio_.timestamp + 1.0 && pre_vio_.timestamp > 0.0) {
    droslog(LogLevel::WARN, "VrelocTracker::FeedData(vio) 时间戳变化大于1s, ts: %.3f, pre_ts: %.3f", vio.timestamp, pre_vio_.timestamp);
  }

  // 这里累计里程和角程
  if (pre_vio_.timestamp > 0.0) {
    double dist = (vio.ppose.pos - pre_vio_.ppose.pos).norm();
    const Eigen::Quaterniond dq = vio.ppose.quat * pre_vio_.ppose.quat.inverse();
    double delta_angle = GetEulerRPY(dq).norm();

    std::lock_guard<std::mutex> lock(acc_odom_mutex_);
    acc_odom_.dist += dist;
    acc_odom_.angle += delta_angle;
  }
  pre_vio_ = vio;
  s_pre_vio_ts = GetNow_Steady();
  
  std::lock_guard<std::mutex> lock(vio_q_mutex_);
  vio_q_.emplace_back(vio, vio.timestamp);
}

void VrelocTracker::FeedVreloc(const common::Data_ProbPose &vreloc) {  // 喂入vreloc数据
  static double pre_vreloc_ts = 0.0;
  if (vreloc.timestamp <= pre_vreloc_ts) {
    static SimpleLogFilter log_filter(500);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::WARN, "VrelocTracker::FeedData(vreloc) 时间戳未单调递增, ts: %.3f, pre_ts: %.3f", vreloc.timestamp, pre_vreloc_ts);
    }
    return;
  }

  pre_vreloc_ts = vreloc.timestamp;

  std::lock_guard<std::mutex> lock(vreloc_q_mutex_);
  vreloc_q_.emplace_back(vreloc, vreloc.timestamp);
}

common::Timed<common::Pose> VrelocTracker::GetVioTF() {  // 获取vio到tf的转换   
  std::lock_guard<std::mutex> lock(tf_pose_mutex_);
  return filtered_tf_pose_;
}

common::Data_Pose VrelocTracker::GetVioInLocalXyz(const Eigen::Vector3d &pos, const Eigen::Quaterniond &q) {  // 获取vio到局部导航坐标系的转换
  common::Data_Pose ret;
  ret.pose.pos = pos;
  ret.pose.quat = q;
  {
    std::lock_guard<std::mutex> lock(tf_pose_mutex_);
    ret.pose.quat = filtered_tf_pose_.data.quat * ret.pose.quat;
    ret.pose.pos = filtered_tf_pose_.data.pos + filtered_tf_pose_.data.quat * ret.pose.pos;
  }

  return ret;
}

void VrelocTracker::TrackerThread() {  // 重定位跟踪器线程  
  droslog(LogLevel::INFO, "VrelocTracker::TrackerThread() start+++");
  stopped_.store(false);
  to_stop_.store(false);

  double pre_vio_ts = 0.0;
  while (!to_stop_.load()) {
    Sleep(30);

    {
      static SimpleLogFilter fps_filter(5000);
      if (fps_filter.Output(GetNow_Steady())) {
        DebugPrint();
      }
    }

    common::Data_ProbPose cur_vio;
    {
      std::lock_guard<std::mutex> lock(vio_q_mutex_);
      if (vio_q_.size() > 10) {
        cur_vio = vio_q_[10];
      } else {
        continue;
      }
    }
    if (cur_vio.timestamp <= pre_vio_ts + 0.001) {
      continue;
    }
    pre_vio_ts = cur_vio.timestamp;

    VioWithVreloc vwv;
    vwv.timestamp = cur_vio.timestamp;
    vwv.vio = cur_vio.ppose;

    {
      std::lock_guard<std::mutex> lock(tf_pose_mutex_);
      vwv.align_vio.quat = filtered_tf_pose_.data.quat * cur_vio.ppose.quat;
      vwv.align_vio.pos = filtered_tf_pose_.data.pos + filtered_tf_pose_.data.quat * cur_vio.ppose.pos;
    }

    // 查找vreloc: vreloc的时间戳与vio-kf的时间戳是一致的
    bool has_new_vreloc = false;
    {
      int idx = vreloc_q_.findAfter(vwv.timestamp);
      if (idx >= 0) {
        auto pre_vreloc = vreloc_q_[idx];
        if (vwv.timestamp - pre_vreloc.timestamp < 0.05) {
          vwv.vreloc = std::make_shared<ProbPose>();
          vwv.vreloc->pos = pre_vreloc.ppose.pos;
          vwv.vreloc->quat = pre_vreloc.ppose.quat;

          has_new_vreloc = true;

          droslog(LogLevel::INFO, "VrelocTracker::TrackerThread() 查找到对齐的vreloc: vio.ts=%.3f, vreloc.ts=%.3f", 
              cur_vio.timestamp, pre_vreloc.timestamp);
        }
      }
    }

    align_window_.emplace_back(vwv, vwv.timestamp);

    if (align_window_.size() < 32)
      continue;

    // 定时估计转换, 两秒一次
    int valid_vreloc_cnt = 0;
    static SimpleLogFilter fps_filter(2000);
    if (fps_filter.Output(GetNow_Steady())) {
      AlignConfig align_config;
      align_config.vio_factor = config_.vio_factor;
      align_config.vio_align_factor = config_.vio_align_factor;
      align_config.vio_vreloc_factor = config_.vio_vreloc_factor;

      std::vector<VioWithVreloc> align_vv_vec;
      int window_size = align_window_.size();
      for (int i = 0; i < window_size; i++) {
        align_vv_vec.push_back(align_window_[i]);
        if (align_window_[i].vreloc.get()) {
          valid_vreloc_cnt++;
        }
      }
      droslog(LogLevel::INFO, "VrelocTracker::TrackerThread() 对齐窗口大小: %d, 有效vreloc数量: %d", window_size, valid_vreloc_cnt);

      if (valid_vreloc_cnt > 0) {
        auto ts1 = GetNow_Steady();
        auto tf_pose = spa_align(align_vv_vec, align_config);
        auto ts2 = GetNow_Steady();
        droslog(LogLevel::INFO, "VrelocTracker::TrackerThread() spa_align: use_time=%lld ms", ts2-ts1);
        // 计算对齐
        if (tf_pose.timestamp <= 0.0) {
          droslog(LogLevel::WARN, "VrelocTracker::TrackerThread() spa_align failed, use_time=%lld ms", ts2-ts1);
        } else {
          auto old_pos = filtered_tf_pose_.data.pos;  //  原始TF的位姿
          auto old_quat = filtered_tf_pose_.data.quat;  //  原始TF的旋转
          auto new_pos = tf_pose.pose.pos;  //  新的TF的位姿
          auto old_rpy = GetEulerRPY(old_quat);  //  原始TF的欧拉角
          auto new_rpy = GetEulerRPY(tf_pose.pose.quat);  //  新的TF的欧拉角

          // 2026-01-14: 移除TF跳变检测
          // 原因：
          // 1. findConnection() 已有多层验证：PnP RANSAC、GPS验证、relative_t验证
          // 2. spa_align() 使用 Huber 鲁棒核函数，能自动抑制异常值
          // 3. TF跳变检测会导致死锁：TF长时间未更新 -> VIO漂移累积 -> 跳变过大 -> 拒绝更新
          // 现在直接信任 spa_align 的结果
          
          // 计算TF变化（仅用于日志）
          double pos_jump = (new_pos - old_pos).norm();
          double yaw_jump = std::abs(new_rpy[2] - old_rpy[2]);
          if (yaw_jump > M_PI) {
            yaw_jump = 2.0 * M_PI - yaw_jump;
          }
          double tf_stale_time = (filtered_tf_pose_.ts > 0.0) ? (cur_vio.timestamp - filtered_tf_pose_.ts) : 0.0;
          
          droslog(LogLevel::INFO, "VrelocTracker::TrackerThread() spa_align success, use_time=%lld ms, pos_jump=%.3f, yaw_jump=%.3f, stale=%.1fs, old_tf->new_tf:(%.3f,%.3f,%.3f; %.3f,%.3f,%.3f)->(%.3f,%.3f,%.3f; %.3f,%.3f,%.3f)", 
              ts2-ts1, pos_jump, yaw_jump, tf_stale_time,
              old_pos[0], old_pos[1], old_pos[2], old_rpy[0], old_rpy[1], old_rpy[2], 
              new_pos[0], new_pos[1], new_pos[2], new_rpy[0], new_rpy[1], new_rpy[2]);
          
          // ========== 渐进式校正 ==========
          Eigen::Vector3d final_pos;
          Eigen::Quaterniond final_quat;
          
          if (config_.tf_filter_enable && filtered_tf_pose_.ts > 0.0) {
            // 使用低通滤波进行渐进式校正
            double alpha = config_.tf_filter_alpha;
            final_pos = old_pos * (1.0 - alpha) + new_pos * alpha;
            final_quat = old_quat.slerp(alpha, Eigen::Quaterniond(tf_pose.pose.quat));
            
            auto final_rpy = GetEulerRPY(final_quat);
            droslog(LogLevel::INFO, "VrelocTracker::TrackerThread() 渐进式校正: alpha=%.2f, final_tf=(%.3f,%.3f,%.3f; %.3f,%.3f,%.3f)", 
                alpha, final_pos[0], final_pos[1], final_pos[2], final_rpy[0], final_rpy[1], final_rpy[2]);
          } else {
            // 直接使用新的TF
            final_pos = new_pos;
            final_quat = tf_pose.pose.quat;
            droslog(LogLevel::INFO, "VrelocTracker::TrackerThread() 直接更新TF(首次或禁用滤波)");
          }
          
          {
            std::lock_guard<std::mutex> lock(tf_pose_mutex_);
            filtered_tf_pose_.ts = cur_vio.timestamp;
            filtered_tf_pose_.data.pos = final_pos;
            filtered_tf_pose_.data.quat = final_quat;
          }
          {
            std::lock_guard<std::mutex> lock(acc_odom_mutex_);  
            acc_odom_.dist = 0.0;
            acc_odom_.angle = 0.0;
          }
        }
      }
    }
  }
  droslog(LogLevel::INFO, "VrelocTracker::TrackerThread() stop---");
}