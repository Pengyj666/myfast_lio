#ifndef VMAP_VRELOC_TRACKER_H
#define VMAP_VRELOC_TRACKER_H

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "common/data_utils.h"
#include "common/data_type.h"
#include "common/timed_queue.h"

#include "spa_align.h"

// 输入: vio(默认初始从桩上启动), 在桩信息, 重定位信息
// 基本逻辑: 原vio作为初始值和ref约束，在桩和重定位信息作为轻ref约束
// 重定位一致性检验: 重定位信息与vio的相对位姿一致性检验
class VrelocTracker {
 public:
  struct AccOdom {
    double dist = 0.0;      // meter
    double angle = 0.0;     // radian
  };
  struct Config {
    double tf_valid_dist = 10.0;  // meter
    double vio_factor = 1.0;
    double vio_align_factor = 0.01;
    double vio_vreloc_factor = 0.1;
    
    // TF变化检查参数  1118
    double tf_max_pos_jump = 3;      // 最大位置跳变阈值(m)
    double tf_max_yaw_jump = 0.3;      // 最大航向跳变阈值(rad, ~17度)
    
    // 渐进式校正参数 1118
    bool tf_filter_enable = true;      // 是否启用渐进式校正
    double tf_filter_alpha = 0.3;      // 滤波系数(0-1, 越小越平滑)
  };

  static VrelocTracker* Instance() {
    static VrelocTracker ins;
    return &ins;
  }
  ~VrelocTracker();

  void Reset();

  void SetParams(const Config &config);

  bool IsVioValid();
  bool IsTFValid();

  void DebugPrint();

  // vio-kf
  void FeedData(const common::Data_ProbPose &vio_kf);
  // 成功重定位的vio-kf
  void FeedVreloc(const common::Data_ProbPose &vreloc);

  // 获取vio到局部导航坐标系的转换
  // return.ts为最后一个gnss/vreloc-vio对齐的时间, < 0.0 表示未进行过对齐
  common::Timed<common::Pose> GetVioTF();

  // vio转到局部导航坐标系下, 以gps天线中心
  // return.ts < 0.0, 表示转换无效, 有效时return.ts == vio_result.ts且 > 0.0
  common::Data_Pose GetVioInLocalXyz(const Eigen::Vector3d &pos, const Eigen::Quaterniond &q);

 private:
  VrelocTracker();
  VrelocTracker(const VrelocTracker&) = delete;
  VrelocTracker& operator=(const VrelocTracker&) = delete;

  void Init();
  void Quit();

  void TrackerThread();

  std::atomic_bool stopped_;
  std::atomic_bool to_stop_;
  std::thread tracker_thread_;

  common::Data_ProbPose pre_vio_;

  std::mutex tf_pose_mutex_;
  common::Timed<common::Pose> filtered_tf_pose_;    // 过滤后的tf_pose_

  Config config_;

  std::mutex acc_odom_mutex_;
  AccOdom acc_odom_;
  
  std::mutex vio_q_mutex_;
  utils::TimedQueue<common::Data_ProbPose> vio_q_;
  std::mutex vreloc_q_mutex_;
  utils::TimedQueue<common::Data_ProbPose> vreloc_q_;

  utils::TimedQueue<VioWithVreloc> align_window_;
};

#endif //VMAP_VRELOC_TRACKER_H