/*******************************************************
 * SubMapCache 类定义
 * 
 * 子图缓存管理器，支持基于位置的加载/淘汰策略
 * 配合 AsyncSubMapLoader 实现异步预加载
 * 
 * 2025-12-22 更新：
 * - 支持双目录搜索（work 目录优先于原始目录）
 * - 分层缓存：元数据（层级2）+ 完整数据（层级3）
 * - 标记脏子图支持增量存盘
 * 
 * 创建日期: 2025-12-15
 *******************************************************/

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <mutex>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <functional>

#include <eigen3/Eigen/Dense>

#include "spatial_index.h"
#include "submap.h"

// 加载任务
struct LoadTask {
    SubMapID id;
    std::string file_path;
    int priority = 0;  // 优先级（越大越优先）
    
    bool operator<(const LoadTask& other) const {
        return priority < other.priority;  // 小于 = 优先级低
    }
};

// 缓存统计信息
struct CacheStats {
    int hit_count = 0;
    int miss_count = 0;
    int load_count = 0;
    int evict_count = 0;
};

class SubMapCache {
public:
    SubMapCache();
    ~SubMapCache();
    
    // ========== 初始化 ==========
    
    // 初始化缓存
    // map_dir: 地图目录路径（包含 spatial/ 子目录）
    // max_cached_submaps: 最大缓存子图数量
    void initialize(const std::string& map_dir, int max_cached_submaps = 16);
    
    // 设置工作模式（启用双目录搜索）
    // 工作模式下：优先从 submaps_work/ 加载，写入也到 submaps_work/
    void setWorkMode(bool enabled);
    
    // 关闭缓存（停止加载线程，清空缓存）
    void shutdown();
    
    // ========== 缓存访问 ==========
    
    // 获取子图（如果已缓存则直接返回，否则返回 nullptr 并触发异步加载）
    std::shared_ptr<SubMap> getSubMap(const SubMapID& id);
    
    // 同步获取子图（阻塞直到加载完成）
    std::shared_ptr<SubMap> getSubMapSync(const SubMapID& id);
    
    // 检查子图是否已缓存
    bool isCached(const SubMapID& id) const;
    
    // 预加载子图（异步）
    void prefetch(const SubMapID& id, int priority = 0);
    
    // 批量预加载
    void prefetchBatch(const std::vector<SubMapID>& ids, int priority = 0);
    
    // ========== 位置感知缓存管理 ==========
    
    // 更新当前位置（触发智能淘汰和预加载周围 8 个子图）
    // 只保留一层预加载，移除速度预测预加载
    void updatePosition(const Eigen::Vector3d& position);
    
    // GPS 辅助初始加载（加载指定位置周围的子图）
    // position: GPS 位置
    // radius: 加载半径（米）
    void initialLoadByPosition(const Eigen::Vector3d& position, double radius = 15.0);
    
    // ========== 缓存控制 ==========
    
    // 强制淘汰指定子图
    void evict(const SubMapID& id);
    
    // 清空缓存
    void clear();
    
    // 设置最大缓存数量
    void setMaxCachedSubmaps(int max_count);
    
    // 设置最大缓存范围（子图格数，默认2表示 ±2 格）
    void setMaxCacheRange(int range);
    
    // ========== 缓存内容访问 ==========
    
    // 获取所有缓存的子图
    std::vector<std::shared_ptr<SubMap>> getAllCachedSubMaps();
    
    // ========== 统计信息 ==========
    
    int getCachedCount() const;
    CacheStats getStats() const;
    void resetStats();
    void printStats() const;
    
    // ========== 工作模式支持 - 2025-12-24 ==========
    
    // 获取地图路径
    std::string getMapPath() const { return map_dir_; }
    
    // 刷新所有脏数据到磁盘
    void flushAllDirty();

private:
    // 异步加载线程函数
    void loadThreadFunc();
    
    // 淘汰策略：淘汰超出范围的子图（基于子图格数）
    // 返回值：淘汰的子图数量
    int evictOutOfRange();
    
    // 获取子图文件路径（支持双目录搜索）
    // 返回存在的文件路径，优先 work 目录
    std::string getSubMapFilePath(const SubMapID& id) const;
    
    // 获取子图写入路径（工作模式写入 work 目录）
    std::string getSubMapWritePath(const SubMapID& id) const;
    
    // 检查子图文件是否存在
    bool subMapFileExists(const SubMapID& id) const;
    
    // 缓存数据
    std::unordered_map<SubMapID, std::shared_ptr<SubMap>> cache_;
    mutable std::mutex cache_mutex_;
    
    // 当前位置所在的子图（用于淘汰判断）
    SubMapID current_submap_;
    std::mutex position_mutex_;
    
    // 加载任务队列
    std::priority_queue<LoadTask> load_queue_;
    std::unordered_set<SubMapID> pending_loads_;  // 正在等待加载的子图
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // 加载线程
    std::thread load_thread_;
    std::atomic<bool> running_;
    
    // 配置
    std::string map_dir_;           // 基础地图目录
    std::string submaps_dir_;       // 原始子图目录 (submaps/)
    std::string submaps_work_dir_;  // 工作子图目录 (submaps_work/)
    bool work_mode_ = false;        // 是否工作模式
    int max_cached_submaps_;
    int max_cache_range_ = 3;       // 最大缓存范围（子图格数），增大以减少频繁淘汰
    int evict_hysteresis_ = 2;      // 淘汰滞后（格数），只有超出 max_cache_range_ + hysteresis 才淘汰
    
    // 统计
    mutable std::mutex stats_mutex_;
    CacheStats stats_;
};


