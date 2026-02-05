#ifndef COMMON_VIO_INIT_TRACKER_H_
#define COMMON_VIO_INIT_TRACKER_H_

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include "common/data_utils.h"
#include "common/data_type.h"

#include "common/timed_queue.h"
#include "common/vio_gnss_align.h"

namespace utils {

// 用于建图无RTK下桩
// 必须从桩上下来
class VioGnssInitor {
 public:
  static VioGnssInitor* Instance() {
    static VioGnssInitor ins;
    return &ins;
  }
  ~VioGnssInitor();

  void Reset();
  void Hello();

  void InitAtStation(double ts);
  void StopInit();

  // gnss已经转到局部地理坐标系
  void FeedGnss(const common::Data_Gnss &gnss);
  // vio_result的位姿已经转换到内部坐标系(gps)
  void FeedVio(const common::Data_VioResult &vio_result);

  // 已经计算出Gnss局部地理坐标系到局部地图坐标系的转换
  bool IsGnssMapOffsetValid() const {
    return gnss_map_offset_.timestamp > 0.0;
  }

  // 返回局部地图坐标系到局部地理坐标系的转换
  // return.ts > 0.0, 表示有效结果
  // return.pos: 局部地图坐标系原点在局部地理坐标系下的位置
  // return.quat: 局部地图坐标系在局部地理坐标系下的旋转(目前仅考虑Z轴旋转)
  common::Data_Pose GetGnssMapOffset();

 private:
  VioGnssInitor();
  VioGnssInitor(const VioGnssInitor&) = delete;
  VioGnssInitor& operator=(const VioGnssInitor&) = delete;

  void Init();
  void Quit();

  void TrackerThread();

  // return.ts < 0.0, 表示无效结果
  common::Data_Gnss GetGnssByTime(const double &ts);

  common::Data_Pose Vio2LocalPose(const common::Data_VioResult &vio_result);

  std::mutex reset_mutex_;

  std::atomic_bool stopped_;
  std::atomic_bool to_stop_;
  std::thread tracker_thread_;

  std::atomic_bool is_init_at_station_;
  std::atomic_bool is_need_init_;

  common::Data_Pose gnss_map_offset_;   // 时间戳为首次计算结果, 不可更新, 除非Reset()

  std::mutex gnss_q_mutex_;
  TimedQueue<common::Data_Gnss> gnss_q_;    // 存的都是固定解
  std::mutex pose_q_mutex_;
  TimedQueue<common::Data_Pose> pose_q_;    // 已经转换为局部地图坐标系的坐标

  TimedQueue<SpaNode> spa_node_q_;
};
 
} // namespace utils
#endif  // COMMON_VIO_INIT_TRACKER_H_