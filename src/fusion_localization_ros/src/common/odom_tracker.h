#ifndef COMMON_ODOM_TRACKER_H_
#define COMMON_ODOM_TRACKER_H_

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include "common/data_utils.h"
#include "common/data_type.h"

#include "common/timed_queue.h"
#include "common/vio_gnss_align.h"

namespace utils {

class OdomTracker {
 public:
  struct OdomTrackerParams {
    double rtk_fix_ll_sigma = 0.02;
    double rtk_float_ll_sigma = 0.03;

    double pose_adj_factor = 1.0;
    double pose_align_factor = 0.49;

    double rtk_fix_info_sigma = 0.3;
    double rtk_float_info_sigma = 0.03;
    double reloc_info_pos_sigma = 0.3;
    double reloc_info_quat_sigma = 0.1;
  };

  static OdomTracker* Instance() {
    static OdomTracker ins;
    return &ins;
  }
  ~OdomTracker();
  void Reset();
  bool IsOdomValid();

  void Hello();
  void DebugPrint();

  void SetParams(const OdomTrackerParams &params);

  void InitAtStation(double ts);

  // gnss.enu转到了local_gps_xyz
  void FeedData(const common::Data_Gnss &gnss);
  // vio_result的位姿已经转换到gps天线中心
  void FeedData(const common::Data_VioResult &vio_result);
  // vreloc的位姿已经转换到gps天线中心
  void FeedVreloc(const common::Data_VioResult &vreloc);

  // 获取vio到局部导航坐标系的转换
  // return.ts为最后一个gnss/vreloc-vio对齐的时间, < 0.0 表示未进行过对齐
  common::Timed<common::Pose> GetVioTF();

  // vio转到局部导航坐标系下, 以gps天线中心
  // return.ts < 0.0, 表示转换无效, 有效时return.ts == vio_result.ts且 > 0.0
  common::Data_VioResult GetVioInLocalXyz(const common::Data_VioResult &_vio_result);

 private:
  OdomTracker();
  OdomTracker(const OdomTracker&) = delete;
  OdomTracker& operator=(const OdomTracker&) = delete;

  void Init();
  void Quit();

  void TrackerThread();

  OdomTrackerParams params_;

  std::mutex reset_mutex_;

  std::atomic_bool stopped_;
  std::atomic_bool to_stop_;
  std::thread tracker_thread_;

  common::Data_ProbPose pre_odom_;

  std::mutex tf_pose_mutex_;
  common::Timed<common::Pose> tf_pose_;

  common::Data_Pose vio_local_pose_;                       // vio的local-xyz坐标

  std::mutex gnss_q_mutex_;
  TimedQueue<common::Data_Gnss> gnss_q_;    // 存的都是固定解
  std::mutex odom_q_mutex_;
  TimedQueue<common::Data_ProbPose> odom_q_;    // 存的都是高置信度结果, 已转换到gps中心
  std::mutex reloc_q_mutex_;
  TimedQueue<common::Data_ProbPose> vreloc_q_; // 视觉重定位结果, 已转换到gps中心, 时间戳对应vio的时间戳

  TimedQueue<SpaNode> spa_node_q_;
};
 
} // namespace utils
#endif  // COMMON_ODOM_TRACKER_H_