#ifndef COMMON_LOC_TRACKER_H_
#define COMMON_LOC_TRACKER_H_

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include "common/data_utils.h"
#include "common/data_type.h"

#include "common/timed_queue.h"
#include "common/vio_gnss_align.h"

namespace utils {

class LocTracker {
 public:
  struct LocTrackerParams {
    double rtk_fix_ll_sigma = 0.02;
    double rtk_float_ll_sigma = 0.03;

    double pose_adj_factor = 1.0;
    double pose_align_factor = 0.16;
    double pose_rp_factor = 0.04;

    double rtk_fix_info_sigma = 0.3;
    double rtk_float_info_sigma = 0.03;
    double reloc_info_pos_sigma = 0.3;
    double reloc_info_quat_sigma = 0.1;
  };

  LocTracker();
  ~LocTracker();
  void Reset();

  // ck_dts: sec
  bool IsLocValid(long long ck_dts = 100000);

  void Hello();
  void DebugPrint();

  void SetParams(const LocTrackerParams &params);

  void InitAtStation(double ts);

  // gnss.enu转到了local_gps_xyz
  void FeedGnss(const common::Data_Gnss &gnss);
  // pose的位姿已经转换到gps天线中心
  void FeedPose(const common::Data_ProbPose &pose);
  // reloc的位姿已经转换到gps天线中心
  void FeedReloc(const common::Data_ProbPose &reloc);

  // 获取局部导航坐标系的转换
  // return.ts为最后一个gnss/reloc-pose对齐的时间, < 0.0 表示未进行过对齐
  common::Timed<common::Pose> GetTF();

  // pose转到局部导航坐标系下, 以gps天线中心
  // return.ts < 0.0, 表示转换无效, 有效时return.ts == pose.ts且 > 0.0
  common::Data_ProbPose GetPoseInLocalXyz(const common::Data_ProbPose &_pose);

  // for debug, gps 天线中心
  common::Data_Pose GetLastPoseLocalXyz() const { return local_pose_; }

  // 当长距离丢失rtk或reloc对齐后, 修正类别为vio
  // 仅长距离丢失reloc对齐后, 修正类别为reloc
  // 未长距离丢失rtk对齐, 修正类别为rtk
  float GetOffRtkDist() const { return off_rtk_dist_.load(); }
  float GetOffRelocDist() const { return off_reloc_dist_.load(); } 

 private:
  void Init();
  void Quit();

  void TrackerThread();

  LocTrackerParams params_;

  std::mutex reset_mutex_;

  std::atomic_bool stopped_;
  std::atomic_bool to_stop_;
  std::thread tracker_thread_;

  std::atomic<double> off_rtk_dist_;
  std::atomic<double> off_reloc_dist_;

  common::Data_ProbPose pre_pose_;
  long long pre_pose_ts_;

  std::mutex tf_pose_mutex_;
  common::Timed<common::Pose> tf_pose_;

  common::Data_Pose local_pose_;                       // pose的local-xyz坐标, gps中心

  std::mutex gnss_q_mutex_;
  TimedQueue<common::Data_Gnss> gnss_q_;        // 存的都是固定解
  std::mutex pose_q_mutex_;
  TimedQueue<common::Data_ProbPose> pose_q_;    // 已转换到gps中心
  std::mutex reloc_q_mutex_;
  TimedQueue<common::Data_ProbPose> reloc_q_;   // 重定位结果, 已转换到gps中心

  TimedQueue<SpaNode> spa_node_q_;
};
 
} // namespace utils
#endif  // COMMON_LOC_TRACKER_H_