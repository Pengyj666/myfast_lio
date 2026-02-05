#ifndef COMMON_VIO_GNSS_TRACKER_H_
#define COMMON_VIO_GNSS_TRACKER_H_

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include "common/data_utils.h"
#include "common/data_type.h"

#include "common/timed_queue.h"
#include "common/vio_gnss_align.h"

namespace utils {

// vio 定位器
// vio作为基础前端, vio跟踪有效 <=> 转换位姿有效
// 1. 在桩作为约束, 对齐vio有效, 用于无RTK下桩
// 2. 融合位姿作为约束
// 3. 参考约束: RTK约束, 重定位约束
// 4. 转换位姿有效时, 启用简易边缘化
//
// 对外输出对齐后vio前端, 当rtk掉固定解后切入
// 注意: vio要检测跳变过滤
// 注意: vio跑飞检测提高实时性
class VioTracker {
 public:
  struct VioTrackerParams {
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

  static VioTracker* Instance() {
    static VioTracker ins;
    return &ins;
  }
  ~VioTracker();
  void Reset();

  // ck_dts: sec
  bool IsVioValid(long long ck_dts = 100000);

  void Hello();
  void DebugPrint();

  void SetParams(const VioTrackerParams &params);

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

  // for debug, gps 天线中心
  common::Data_Pose GetLastVioLocalXyz() const { return vio_local_pose_; }

  // 当长距离丢失rtk或reloc对齐后, 修正类别为vio
  // 仅长距离丢失reloc对齐后, 修正类别为reloc
  // 未长距离丢失rtk对齐, 修正类别为rtk
  float GetOffRtkDist() const { return off_rtk_dist_.load(); }
  float GetOffRelocDist() const { return off_reloc_dist_.load(); } 

 private:
  VioTracker();
  VioTracker(const VioTracker&) = delete;
  VioTracker& operator=(const VioTracker&) = delete;

  void Init();
  void Quit();

  void TrackerThread();

  VioTrackerParams params_;

  std::mutex reset_mutex_;

  std::atomic_bool stopped_;
  std::atomic_bool to_stop_;
  std::thread tracker_thread_;

  std::atomic<double> off_rtk_dist_;
  std::atomic<double> off_reloc_dist_;

  common::Data_VioResult pre_vio_;
  long long pre_vio_ts_;

  std::mutex tf_pose_mutex_;
  common::Timed<common::Pose> tf_pose_;

  common::Data_Pose vio_local_pose_;                       // vio的local-xyz坐标, gps中心

  std::mutex gnss_q_mutex_;
  TimedQueue<common::Data_Gnss> gnss_q_;    // 存的都是固定解
  std::mutex vio_q_mutex_;
  TimedQueue<common::Data_VioResult> vio_q_;    // 存的都是高置信度结果, 已转换到gps中心
  std::mutex vreloc_q_mutex_;
  TimedQueue<common::Data_VioResult> vreloc_q_; // 视觉重定位结果, 已转换到gps中心, 时间戳对应vio的时间戳

  TimedQueue<SpaNode> spa_node_q_;
};
 
} // namespace utils
#endif  // COMMON_VIO_GNSS_TRACKER_H_