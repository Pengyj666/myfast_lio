#pragma once

#include <list>
#include <mutex>
#include <memory>
#include <vector>
#include <set>

#include "keyframe.h"

#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
#include "ThirdParty/DBoW/TemplatedDatabase.h"
#include "ThirdParty/DBoW/TemplatedVocabulary.h"

#include <sensor_msgs/PointCloud2.h>

class SimplePoseGraph
{
public:
  SimplePoseGraph(int type = 0);
  ~SimplePoseGraph();

  // void Reset();

  // online build map
  void addKeyFrame(std::shared_ptr<KeyFrame> cur_kf, bool flag_detect_loop);
  // load map
	void loadKeyFrame(std::shared_ptr<KeyFrame> cur_kf, bool flag_detect_loop);
	
	// 加载词汇表（只需在节点启动时调用一次）
	void loadVocabulary(std::string voc_path);
	
	// 检查词汇表是否已加载
	bool isVocabularyLoaded() const { return voc_ != nullptr; }
	
	// 清空数据库但保留词汇表（用于重置/切换地图）
	void clearDatabase();

  std::shared_ptr<KeyFrame> getKeyFrame(int index);
  
  // 获取所有关键帧（用于 loopCorrection 后重建空间索引）
  std::vector<std::shared_ptr<KeyFrame>> getAllKeyFrames();
  
  // 获取关键帧数量
  int getKeyFrameCount();
  
  // 清理未被空间索引的关键帧，释放内存
  // validKeyFrames: 空间索引中的有效关键帧集合
  // 返回值：清理的关键帧数量
  int cleanupUnindexedKeyFrames(const std::vector<std::shared_ptr<KeyFrame>>& validKeyFrames);

  void saveMap(std::string map_path);
  // return: 1-OK, 0-Failed
	int loadMap(std::string map_path);

  int detectLoop(const std::shared_ptr<KeyFrame> &cur_kf, int frame_index);

  // type: 0-reloc, 1-det_loop
  // is_first_reloc: 是否是首次重定位（VIO 坐标系还未对齐到地图坐标系）
  // return >=0: OK, <0: Failed
  int relocalization(const std::shared_ptr<KeyFrame> &cur_kf, Eigen::Vector3d &pos, Eigen::Quaterniond &quat, int type=0, bool is_first_reloc=false);
  
  // ========== 空间索引重定位 - 2025-12-25 ==========
  // 使用空间索引筛选候选帧 + 方向过滤，比全局 DBoW2 搜索更高效
  // cur_kf: 当前关键帧
  // cur_position: 当前位置估计（VIO 位置）
  // 设置空间索引管理器（工作模式时调用，用于冷热数据管理）
  void setSpatialMapManager(class SpatialMapManager* manager);

  // return 0-failed, 1-success
  int loopCorrection();
  
  // 2026-01-11: 将关键帧添加到 DBoW2 词袋（建图模式使用）
  // 注意：定位模式下不再添加新帧到 DBoW2，回环检测改用空间索引 + BRIEF 匹配
  void addKeyFrameIntoVoc(std::shared_ptr<KeyFrame> keyframe);

private:
  // 空间索引管理器（工作模式使用）
  class SpatialMapManager* spatial_manager_ = nullptr;
  
  // 2026-01-11: 辅助函数，尝试与候选帧进行重定位
  // is_first_reloc: 是否是首次重定位（VIO 坐标系还未对齐）
  int tryRelocWithCandidate(
      const std::shared_ptr<KeyFrame>& cur_kf,
      const std::shared_ptr<KeyFrame>& loop_kf,
      float score,
      Eigen::Vector3d& pos,
      Eigen::Quaterniond& quat,
      int type,
      bool is_first_reloc = false);

  int type_ = 0;   // 0: reloc, 1: build_map
  int global_index = 0;
  
  std::atomic<int> map_stage_;  // 0-idle, 1-building, 2-loop_correction, 3-saving, 4-finish

  std::string voc_path_;

  std::list<std::shared_ptr<KeyFrame>> keyframelist_;
  std::mutex keyframelist_mutex_;

  std::shared_ptr<BriefDatabase> db_;
  std::shared_ptr<BriefVocabulary> voc_;
  
  // 2026-01-11: DBoW2 EntryId 到 KeyFrame index 的映射
  // DBoW2 的 EntryId 是按添加顺序递增的，与 KeyFrame index 可能不一致
  std::vector<int> entry_to_kf_index_;
  std::mutex entry_map_mutex_;
  
  sensor_msgs::PointCloud2 map_point_cloud_;
};