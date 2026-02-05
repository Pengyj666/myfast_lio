/*******************************************************
 * SpatialMapManager 类定义
 * 
 * 空间地图管理器，负责子图的创建、管理和查询
 * 提供坐标转换、关键帧插入、空间查询等功能
 * 
 * 2026-01-07 更新：
 * - 合并 SubMapCache 功能，成为唯一的子图管理模块
 * - 支持后台异步预加载
 * - 统一的淘汰策略
 * 
 * 创建日期: 2025-12-10
 *******************************************************/

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <mutex>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <condition_variable>

#include <eigen3/Eigen/Dense>

#include "spatial_index.h"
#include "submap.h"
#include "spatial_meta.h"

// 预加载任务
struct PreloadTask {
    SubMapID id;
    int priority = 0;  // 优先级（越大越优先）
    
    bool operator<(const PreloadTask& other) const {
        return priority < other.priority;  // 小于 = 优先级低
    }
};

// 缓存统计信息
struct SubMapCacheStats {
    int hit_count = 0;
    int miss_count = 0;
    int load_count = 0;
    int evict_count = 0;
};

class KeyFrame;

class SpatialMapManager {
public:
    SpatialMapManager();
    ~SpatialMapManager();
    
    // ========== 关键帧管理 ==========
    
    // 插入关键帧（自动计算空间索引并插入到对应子图）
    // 返回值：true 表示插入成功，false 表示被拒绝（已有更好的关键帧）
    bool insertKeyFrame(std::shared_ptr<KeyFrame> kf);
    
    // ========== 增量索引更新 - 2025-12-15 ==========
    
    // 标记关键帧为脏（位姿已更新，需要检查索引）
    void markDirty(std::shared_ptr<KeyFrame> kf);
    
    // 批量标记脏帧
    void markDirtyBatch(const std::vector<std::shared_ptr<KeyFrame>>& kfs);
    
    // 检查关键帧是否需要重新索引（位姿变化导致 Cell/Slot 变化）
    bool needsReindex(std::shared_ptr<KeyFrame> kf) const;
    
    // 重建所有脏帧的索引
    // 返回值：实际移动的帧数
    int rebuildDirtyIndices();
    
    // 获取所有脏 SubMap（用于增量刷盘）
    std::vector<std::shared_ptr<SubMap>> getDirtySubMaps() const;
    
    // 增量保存脏 SubMap 到目录
    // 返回值：成功保存的子图数量
    int saveDirtySubMaps(const std::string& dir_path);
    
    // 清除所有 SubMap 的脏标记
    void clearAllDirtyFlags();
    
    // 根据位置查询关键帧（返回半径范围内的所有关键帧）
    std::vector<std::shared_ptr<KeyFrame>> queryKeyFrames(
        const Eigen::Vector3d& position, double radius) const;
    
    // 根据位置和方向查询关键帧（用于重定位，只返回方向相近的关键帧）
    std::vector<std::shared_ptr<KeyFrame>> queryKeyFramesByPose(
        const Eigen::Vector3d& position, double yaw, double radius) const;
    
    // ========== 子图管理 ==========
    
    // 获取或创建子图
    std::shared_ptr<SubMap> getOrCreateSubMap(const SubMapID& id);
    
    // 获取子图（不存在返回 nullptr）
    std::shared_ptr<SubMap> getSubMap(const SubMapID& id) const;
    
    // 检查子图是否已加载
    bool isSubMapLoaded(const SubMapID& id) const;
    
    // 获取所有子图ID
    std::vector<SubMapID> getAllSubMapIDs() const;
    
    // 清空所有子图
    void clear();
    
    // 获取所有被索引的关键帧（用于内存清理时标记有效帧）
    std::vector<std::shared_ptr<KeyFrame>> getAllIndexedKeyFrames() const;
    
    // ========== 坐标转换工具（静态方法）==========
    
    // 位置转子图ID
    static SubMapID positionToSubMapID(const Eigen::Vector3d& position);
    
    // 位置转 Cell ID
    static CellID positionToCellID(const Eigen::Vector3d& position);
    
    // 航向角转方向槽位 (0-5)
    // yaw: 弧度，范围任意（会自动归一化）
    static int yawToDirectionSlot(double yaw);
    
    // 从旋转矩阵提取 yaw 角
    static double getYawFromRotation(const Eigen::Matrix3d& R);
    
    // ========== 统计信息 ==========
    
    int getSubMapCount() const;
    int getTotalKeyFrameCount() const;
    int getTotalCellCount() const;
    void printStatistics() const;
    
    // ========== 序列化（保存/加载）==========
    
    // 保存所有子图到目录
    // dir_path: 存储目录路径
    // origin_lat/lon/alt: GPS/RTK 坐标系原点（可选）
    // 返回值：成功保存的子图数量
    int saveToDirectory(const std::string& dir_path, 
                        double origin_lat = 0.0, 
                        double origin_lon = 0.0, 
                        double origin_alt = 0.0);
    
    // 从目录加载所有子图
    // dir_path: 存储目录路径
    // 返回值：成功加载的子图数量
    int loadFromDirectory(const std::string& dir_path);
    
    // 加载单个子图（用于动态加载）
    bool loadSubMap(const std::string& file_path);
    
    // 卸载子图（用于动态卸载，释放内存）
    bool unloadSubMap(const SubMapID& id);
    
    // 添加已加载的子图（用于外部加载后插入）
    void addSubMap(std::shared_ptr<SubMap> submap);
    
    // 获取所有子图（用于序列化）
    std::vector<std::shared_ptr<SubMap>> getAllSubMaps() const;
    
    // 生成元信息
    SpatialMapMeta generateMeta(double origin_lat = 0.0, 
                                 double origin_lon = 0.0, 
                                 double origin_alt = 0.0) const;
    
    // ========== 工作模式支持 - 2025-12-24 ==========
    
    // 重建空间索引（位姿优化后调用）
    void rebuildSpatialIndex();
    
    // 合并 work 目录到 submaps 目录
    // 返回合并的子图数量
    int mergeWorkToSubmaps(const std::string& work_dir, const std::string& submaps_dir);
    
    // ========== 分层缓存：轻量元数据索引 - 2025-12-30 ==========
    
    // 添加轻量元数据（用于空间筛选，不存储完整关键帧）
    void addKeyFrameMetadata(const KeyFrameMetadata& meta);
    
    // 根据位置查询元数据（轻量查询，不加载完整数据）
    std::vector<KeyFrameMetadata> queryMetadataByPosition(
        const Eigen::Vector3d& position, double radius) const;
    
    // 根据位置和方向查询元数据
    std::vector<KeyFrameMetadata> queryMetadataByPose(
        const Eigen::Vector3d& position, double yaw, double radius) const;
    
    // 获取所有元数据
    const std::unordered_map<int, KeyFrameMetadata>& getAllMetadata() const { return kf_metadata_; }
    
    // 清除元数据索引
    void clearMetadata() { std::lock_guard<std::mutex> lock(meta_mutex_); kf_metadata_.clear(); }
    
    // 获取元数据数量
    int getMetadataCount() const { std::lock_guard<std::mutex> lock(meta_mutex_); return static_cast<int>(kf_metadata_.size()); }
    
    // 设置工作模式（启用 SubMap 淘汰）
    void setWorkMode(bool enabled) { work_mode_ = enabled; }
    
    // 淘汰远离当前位置的 SubMap（释放完整关键帧内存）
    // max_range: 保留范围（子图格数），默认 3 格 = ±15m
    // evicted_ids: 输出参数，返回被淘汰的子图 ID（用于同步淘汰 SubMapCache）
    // 返回值：淘汰的 SubMap 数量
    // 注意：内存紧张时应减小此值
    int evictDistantSubMaps(const Eigen::Vector3d& current_pos, int max_range, 
                            std::vector<SubMapID>* evicted_ids = nullptr);
    
    // 简化版本（不返回淘汰列表）
    int evictDistantSubMaps(const Eigen::Vector3d& current_pos) {
        return evictDistantSubMaps(current_pos, 3, nullptr);
    }
    
    // ========== 统一子图管理（原 SubMapCache 功能）- 2026-01-07 ==========
    
    // 初始化预加载系统
    // map_dir: 地图目录路径（包含 spatial/ 子目录）
    // max_cached_submaps: 最大缓存子图数量
    void initializeCache(const std::string& map_dir, int max_cached_submaps = 25);
    
    // 关闭预加载系统（停止后台线程）
    void shutdownCache();
    
    // 更新当前位置（触发淘汰和预加载）
    // 每帧调用，内部会检测子图切换
    void updatePosition(const Eigen::Vector3d& position);
    
    // GPS 辅助初始加载（加载指定位置周围的子图）
    void initialLoadByPosition(const Eigen::Vector3d& position, double radius = 15.0);
    
    // 检查子图是否已被淘汰（有元数据但无完整数据）
    bool isSubMapEvicted(const SubMapID& id) const;
    
    // 获取当前加载的子图数量
    int getLoadedSubMapCount() const { std::lock_guard<std::mutex> lock(mutex_); return static_cast<int>(submaps_.size()); }
    
    // 获取淘汰的子图数量
    int getEvictedSubMapCount() const { std::lock_guard<std::mutex> lock(mutex_); return static_cast<int>(evicted_submaps_.size()); }
    
    // 获取缓存统计信息
    SubMapCacheStats getCacheStats() const;
    
    // 获取地图路径
    std::string getMapPath() const { return map_dir_; }
    
    // 刷新脏数据到磁盘
    void flushAllDirty();
    
    // ========== 滑窗淘汰机制 - 2026-01-08 ==========
    
    // 设置起点/充电桩位置（特殊保留区域）
    void setOriginPosition(const Eigen::Vector3d& position);
    
    // 检查是否已设置起点
    bool hasOriginPosition() const { return has_origin_position_; }
    
    // 获取起点位置
    Eigen::Vector3d getOriginPosition() const { return origin_position_; }
    
    // 设置滑窗大小
    void setSlidingWindowSize(int size) { sliding_window_size_ = size; }
    
    // 获取滑窗大小
    int getSlidingWindowSize() const { return sliding_window_size_; }
    
    // 更新最新关键帧索引（每次添加关键帧时调用）
    void updateLatestKeyFrameIndex(int index);
    
    // 执行滑窗淘汰（空间 + 滑窗 + 特殊保留）
    // 返回淘汰的关键帧数量
    int evictBySlidingWindow();
    
    // ========== 按索引查询关键帧 - 2026-01-11 ==========
    // 用于 DBoW2 查询后获取候选帧，支持冷数据按需加载
    
    // 根据关键帧索引获取关键帧
    // 如果是冷数据（Layer 3 已清空），会自动从磁盘加载
    // 返回值：关键帧指针，如果不存在返回 nullptr
    std::shared_ptr<KeyFrame> getKeyFrameByIndex(int index);
    
    // 检查关键帧是否是热数据（Layer 3 数据完整）
    bool isHotData(int index) const;
    
    // 检查关键帧是否存在（不管是热数据还是冷数据）
    bool hasKeyFrame(int index) const;
    
    // 注册关键帧索引（插入时调用）
    void registerKeyFrameIndex(std::shared_ptr<KeyFrame> kf);
    
    // 注销关键帧索引（替换时调用）
    void unregisterKeyFrameIndex(int index);
    
    // 检查子图是否在原始目录存在（用于判断是新增还是覆盖）
    bool subMapExistsInOriginal(const SubMapID& id) const;

private:
    std::unordered_map<SubMapID, std::shared_ptr<SubMap>> submaps_;
    mutable std::mutex mutex_;  // 线程安全保护
    
    // 脏帧集合（位姿已更新，等待索引重建）
    std::unordered_set<std::shared_ptr<KeyFrame>> dirty_keyframes_;
    mutable std::mutex dirty_mutex_;
    
    // ========== 分层缓存：轻量元数据 - 2025-12-30 ==========
    // 所有关键帧的轻量元数据（永不淘汰，用于空间筛选）
    std::unordered_map<int, KeyFrameMetadata> kf_metadata_;  // key: kf_index
    mutable std::mutex meta_mutex_;
    
    // 工作模式标志
    bool work_mode_ = false;
    
    // 已淘汰的子图 ID 集合（有元数据但无完整数据）
    std::unordered_set<SubMapID> evicted_submaps_;
    
    // ========== 预加载系统（原 SubMapCache）- 2026-01-07 ==========
    
    // 目录路径
    std::string map_dir_;           // 基础地图目录
    std::string submaps_dir_;       // 原始子图目录 (spatial/submaps/)
    std::string submaps_work_dir_;  // 工作子图目录 (spatial/submaps_work/)
    
    // 当前位置所在的子图
    SubMapID current_submap_;
    std::mutex position_mutex_;
    
    // 预加载任务队列
    std::priority_queue<PreloadTask> preload_queue_;
    std::unordered_set<SubMapID> pending_loads_;  // 正在等待加载的子图
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // 后台加载线程
    std::thread preload_thread_;
    std::atomic<bool> preload_running_;
    
    // ========== 缓存配置（针对 600m² 庭院优化）- 2026-01-08 ==========
    // 
    // 三重保留机制：
    // 1. 空间保留：中心 + 周围8格 = 3×3 = 9格 = 225m²
    // 2. 起点保留：起点 + 周围24格 = 5×5 = 25格 = 625m²（确保回充电桩）
    // 3. 滑窗保留：最近400帧（覆盖约8行割草路径）
    //
    int max_cached_submaps_ = 25;
    int sliding_window_size_ = 400;        // 滑窗大小（帧数）
    
    // 起点/充电桩位置
    Eigen::Vector3d origin_position_;      // 起点位置
    bool has_origin_position_ = false;     // 是否已设置起点
    
    // 关键帧索引追踪（原子变量，多线程安全）
    std::atomic<int> latest_keyframe_index_{0};  // 最新关键帧索引
    
    // ========== 按索引快速查找 - 2026-01-11 ==========
    // 用于 DBoW2 查询后快速定位关键帧
    std::unordered_map<int, std::shared_ptr<KeyFrame>> index_to_keyframe_;
    mutable std::mutex index_mutex_;
    
    // 从磁盘加载关键帧的 Layer 3 数据
    bool loadKeyFrameLayer3FromDisk(std::shared_ptr<KeyFrame> kf);
    
    // 统计信息
    mutable std::mutex stats_mutex_;
    SubMapCacheStats cache_stats_;
    
    // 从旧索引位置移除关键帧
    void removeFromOldIndex(std::shared_ptr<KeyFrame> kf);
    
    // 插入到新索引位置
    void insertToNewIndex(std::shared_ptr<KeyFrame> kf);
    
    // 后台预加载线程函数
    void preloadThreadFunc();
    
    // 预加载子图（异步）
    void prefetch(const SubMapID& id, int priority = 0);
    
    // 淘汰超出范围的子图
    int evictOutOfRange();
    
    // 同步加载子图
    std::shared_ptr<SubMap> loadSubMapSync(const SubMapID& id);
    
    // 获取子图文件路径（支持双目录搜索）
    std::string getSubMapFilePath(const SubMapID& id) const;
    
    // 获取子图写入路径
    std::string getSubMapWritePath(const SubMapID& id) const;
    
    // 检查子图文件是否存在
    bool subMapFileExists(const SubMapID& id) const;
};

