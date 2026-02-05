/*******************************************************
 * SubMapCache 实现
 * 
 * 子图缓存管理和异步加载
 * 
 * 2025-12-22 更新：
 * - 支持双目录搜索（work 目录优先于原始目录）
 * - 分层缓存支持
 * - 脏数据管理
 * 
 * 创建日期: 2025-12-15
 *******************************************************/

#include "submap_cache.h"
#include "submap_serializer.h"
#include "spatial_map_manager.h"
#include "droslog/log.h"
#include "common/log_filters.h"
#include "common/sysutils.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <sys/stat.h>

using namespace utils;

// 辅助函数：检查目录是否存在
static bool DirExists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

// 辅助函数：创建目录
static void MakeDir(const char* path) {
    mkdir(path, 0755);
}

SubMapCache::SubMapCache()
    : current_submap_{0, 0}
    , running_(false)
    , work_mode_(false)
    , max_cached_submaps_(16) {
}

SubMapCache::~SubMapCache() {
    shutdown();
}

// ========== 初始化 ==========

void SubMapCache::initialize(const std::string& map_dir, int max_cached_submaps) {
    map_dir_ = map_dir;
    if (map_dir_.back() != '/') map_dir_ += '/';
    
    // 设置目录路径
    submaps_dir_ = map_dir_ + "spatial/submaps/";
    submaps_work_dir_ = map_dir_ + "spatial/submaps_work/";
    
    max_cached_submaps_ = max_cached_submaps;
    
    // 启动加载线程
    running_ = true;
    load_thread_ = std::thread(&SubMapCache::loadThreadFunc, this);
}

void SubMapCache::setWorkMode(bool enabled) {
    work_mode_ = enabled;
    
    if (work_mode_) {
        // 确保 work 目录存在
        std::string spatial_dir = map_dir_ + "spatial/";
        if (!DirExists(spatial_dir.c_str())) {
            MakeDir(spatial_dir.c_str());
        }
        if (!DirExists(submaps_work_dir_.c_str())) {
            MakeDir(submaps_work_dir_.c_str());
        }
    }
}

void SubMapCache::shutdown() {
    if (!running_) return;
    
    running_ = false;
    queue_cv_.notify_all();
    
    if (load_thread_.joinable()) {
        load_thread_.join();
    }
    
    clear();
}

// ========== 缓存访问 ==========

std::shared_ptr<SubMap> SubMapCache::getSubMap(const SubMapID& id) {
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = cache_.find(id);
        if (it != cache_.end()) {
            std::lock_guard<std::mutex> slock(stats_mutex_);
            stats_.hit_count++;
            return it->second;
        }
    }
    
    // 缓存未命中，触发异步加载
    {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        stats_.miss_count++;
    }
    
    prefetch(id, 10);  // 高优先级加载
    return nullptr;
}

std::shared_ptr<SubMap> SubMapCache::getSubMapSync(const SubMapID& id) {
    // 先检查缓存
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = cache_.find(id);
        if (it != cache_.end()) {
            return it->second;
        }
    }
    
    // 同步加载
    std::string file_path = getSubMapFilePath(id);
    struct stat st;
    if (stat(file_path.c_str(), &st) != 0) {
        return nullptr;
    }
    
    SubMapSerializer serializer;
    auto submap = serializer.deserialize(file_path);
    
    if (submap) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        // 检查是否需要淘汰（基于子图格数）
        evictOutOfRange();
        
        // 如果仍超出数量限制，淘汰最远的（使用切比雪夫距离）
        SubMapID cur_submap;
        {
            std::lock_guard<std::mutex> plock(position_mutex_);
            cur_submap = current_submap_;
        }
        
        while (static_cast<int>(cache_.size()) >= max_cached_submaps_) {
            if (cache_.empty()) break;
            int max_dist = -1;
            SubMapID farthest_id;
            for (const auto& p : cache_) {
                int dx = std::abs(p.first.x - cur_submap.x);
                int dy = std::abs(p.first.y - cur_submap.y);
                int d = std::max(dx, dy);  // 切比雪夫距离
                if (d > max_dist) { max_dist = d; farthest_id = p.first; }
            }
            if (max_dist >= 0) {
                cache_.erase(farthest_id);
                std::lock_guard<std::mutex> slock(stats_mutex_);
                stats_.evict_count++;
            } else break;
        }
        
        cache_[id] = submap;
        
        std::lock_guard<std::mutex> slock(stats_mutex_);
        stats_.load_count++;
    }
    
    return submap;
}

bool SubMapCache::isCached(const SubMapID& id) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return cache_.find(id) != cache_.end();
}

void SubMapCache::prefetch(const SubMapID& id, int priority) {
    // 检查是否已缓存
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (cache_.find(id) != cache_.end()) {
            // 已在缓存中，无需重复加载
            return;
        }
    }
    
    // 检查文件是否存在
    if (!subMapFileExists(id)) {
        return;
    }
    
    // 添加到加载队列
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // 检查是否已在等待队列中（去重）
        if (pending_loads_.find(id) != pending_loads_.end()) {
            return;
        }
        
        LoadTask task;
        task.id = id;
        task.file_path = getSubMapFilePath(id);
        task.priority = priority;
        
        load_queue_.push(task);
        pending_loads_.insert(id);
    }
    
    queue_cv_.notify_one();
}

void SubMapCache::prefetchBatch(const std::vector<SubMapID>& ids, int priority) {
    for (const auto& id : ids) {
        prefetch(id, priority);
    }
}

// ========== 位置感知缓存管理 ==========

void SubMapCache::updatePosition(const Eigen::Vector3d& position) {
    // 计算当前位置所在的子图
    SubMapID new_submap = SpatialMapManager::positionToSubMapID(position);
    
    // ========== 优化 - 2026-01-07 ==========
    // 只有当子图 ID 发生变化时才执行后续操作，避免频繁触发淘汰和预加载
    SubMapID old_submap;
    {
        std::lock_guard<std::mutex> lock(position_mutex_);
        old_submap = current_submap_;
        current_submap_ = new_submap;
    }
    
    // 子图未变化，跳过后续操作
    if (new_submap.x == old_submap.x && new_submap.y == old_submap.y) {
        return;
    }
    
    // 子图变化时才触发淘汰和预加载
    // 触发基于子图格数的淘汰
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        evictOutOfRange();
    }
    
    // 预加载当前子图及相邻子图（±1 格，共 9 个）
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            SubMapID neighbor{new_submap.x + dx, new_submap.y + dy};
            prefetch(neighbor, 5 - std::abs(dx) - std::abs(dy));  // 中心优先级最高
        }
    }
    
    // 输出子图切换日志（降频）
    static SimpleLogFilter submap_filter(10000);  // 10 秒一次
    if (submap_filter.Output(GetNow_Steady())) {
        droslog(LogLevel::INFO, "SubMapCache: 子图切换 (%d,%d)->(%d,%d), 当前缓存 %d 个", 
                old_submap.x, old_submap.y, new_submap.x, new_submap.y, getCachedCount());
    }
}

// predictivePreload 已移除，只保留 updatePosition 的单层预加载

void SubMapCache::initialLoadByPosition(const Eigen::Vector3d& position, double radius) {
    // 计算当前位置所在的子图
    SubMapID center = SpatialMapManager::positionToSubMapID(position);
    
    // 计算需要加载的格数范围
    int range = static_cast<int>(std::ceil(radius / SubMap::SUBMAP_SIZE));
    
    // 按距离（曼哈顿距离）排序，优先加载近的
    std::vector<std::pair<int, SubMapID>> sorted_submaps;
    
    for (int dx = -range; dx <= range; dx++) {
        for (int dy = -range; dy <= range; dy++) {
            SubMapID id{center.x + dx, center.y + dy};
            int dist = std::abs(dx) + std::abs(dy);  // 曼哈顿距离
            sorted_submaps.push_back({dist, id});
        }
    }
    
    // 按距离排序
    std::sort(sorted_submaps.begin(), sorted_submaps.end(),
        [](const std::pair<int, SubMapID>& a, const std::pair<int, SubMapID>& b) { 
            return a.first < b.first; 
        });
    
    // 同步加载最近的几个子图，其余异步加载
    int sync_count = std::min(4, static_cast<int>(sorted_submaps.size()));
    
    for (int i = 0; i < sync_count; i++) {
        getSubMapSync(sorted_submaps[i].second);
    }
    
    for (size_t i = sync_count; i < sorted_submaps.size(); i++) {
        prefetch(sorted_submaps[i].second, 10 - static_cast<int>(i));
    }
}

// ========== 缓存控制 ==========

void SubMapCache::evict(const SubMapID& id) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto it = cache_.find(id);
    if (it != cache_.end()) {
        cache_.erase(it);
        
        std::lock_guard<std::mutex> slock(stats_mutex_);
        stats_.evict_count++;
    }
}

void SubMapCache::clear() {
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_.clear();
    }
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!load_queue_.empty()) load_queue_.pop();
        pending_loads_.clear();
    }
}

void SubMapCache::setMaxCachedSubmaps(int max_count) {
    max_cached_submaps_ = max_count;
    
    // 如果超出限制，淘汰超出半径的
    std::lock_guard<std::mutex> lock(cache_mutex_);
    evictOutOfRange();
}

void SubMapCache::setMaxCacheRange(int range) {
    max_cache_range_ = range;
    
    // 立即淘汰超出新范围的子图
    std::lock_guard<std::mutex> lock(cache_mutex_);
    evictOutOfRange();
}

// ========== 缓存内容访问 ==========

std::vector<std::shared_ptr<SubMap>> SubMapCache::getAllCachedSubMaps() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    std::vector<std::shared_ptr<SubMap>> result;
    result.reserve(cache_.size());
    for (const auto& pair : cache_) {
        if (pair.second) {
            result.push_back(pair.second);
        }
    }
    return result;
}

// ========== 统计信息 ==========

int SubMapCache::getCachedCount() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return static_cast<int>(cache_.size());
}

CacheStats SubMapCache::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void SubMapCache::resetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = CacheStats();
}

void SubMapCache::printStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    double hit_rate = (stats_.hit_count + stats_.miss_count > 0) 
        ? 100.0 * stats_.hit_count / (stats_.hit_count + stats_.miss_count) 
        : 0.0;
    
    printf("========== SubMapCache Statistics ==========\n");
    printf("  Cached: %d / %d\n", getCachedCount(), max_cached_submaps_);
    printf("  Hits: %d, Misses: %d (%.1f%% hit rate)\n", 
        stats_.hit_count, stats_.miss_count, hit_rate);
    printf("  Loads: %d, Evicts: %d\n", stats_.load_count, stats_.evict_count);
    printf("=============================================\n");
}

// ========== 私有方法 ==========

void SubMapCache::loadThreadFunc() {
    while (running_) {
        LoadTask task;
        
        // 等待任务
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { 
                return !running_ || !load_queue_.empty(); 
            });
            
            if (!running_) break;
            if (load_queue_.empty()) continue;
            
            task = load_queue_.top();
            load_queue_.pop();
            pending_loads_.erase(task.id);
        }
        
        // 再次检查是否已缓存
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            if (cache_.find(task.id) != cache_.end()) {
                continue;
            }
        }
        
        // 加载子图
        SubMapSerializer serializer;
        auto submap = serializer.deserialize(task.file_path);
        
        if (submap) {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            
            // 淘汰超出半径的子图
            evictOutOfRange();
            
            cache_[task.id] = submap;
            
            {
                std::lock_guard<std::mutex> slock(stats_mutex_);
                stats_.load_count++;
            }
        }
    }
}

int SubMapCache::evictOutOfRange() {
    // 注意：调用者已持有 cache_mutex_
    
    if (cache_.empty()) return 0;
    
    SubMapID cur_submap;
    {
        std::lock_guard<std::mutex> lock(position_mutex_);
        cur_submap = current_submap_;
    }
    
    // ========== 优化 - 2026-01-07 ==========
    // 使用滞后淘汰策略，避免在边界来回移动时频繁淘汰和重新加载
    // 只有超出 max_cache_range_ + evict_hysteresis_ 才淘汰
    int evict_threshold = max_cache_range_ + evict_hysteresis_;
    
    // 收集超出范围的子图（使用切比雪夫距离：max(|dx|, |dy|)）
    std::vector<SubMapID> to_evict;
    
    for (const auto& pair : cache_) {
        int dx = std::abs(pair.first.x - cur_submap.x);
        int dy = std::abs(pair.first.y - cur_submap.y);
        int dist = std::max(dx, dy);  // 切比雪夫距离
        
        // 超出淘汰阈值才淘汰（有滞后）
        if (dist > evict_threshold) {
            to_evict.push_back(pair.first);
        }
    }
    
    // 执行淘汰
    for (const auto& id : to_evict) {
        cache_.erase(id);
    }
    
    if (!to_evict.empty()) {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        stats_.evict_count += static_cast<int>(to_evict.size());
        
        // 输出日志：淘汰了多少子图，释放了多少内存
        // 每个子图大约 500KB-2MB，取 1MB 估计
        int freed_mb = static_cast<int>(to_evict.size());
        droslog(LogLevel::INFO, "SubMapCache: 淘汰 %d 个子图(>%d格)，释放约 %dMB 内存，当前缓存 %d 个",
                (int)to_evict.size(), evict_threshold, freed_mb, (int)cache_.size());
    }
    
    return static_cast<int>(to_evict.size());
}

std::string SubMapCache::getSubMapFilePath(const SubMapID& id) const {
    std::string filename = SubMapSerializer::getSubMapFileName(id);
    
    // 工作模式：优先从 work 目录加载
    if (work_mode_) {
        std::string work_path = submaps_work_dir_ + filename;
        struct stat st;
        if (stat(work_path.c_str(), &st) == 0) {
            return work_path;
        }
    }
    
    // 回退到原始目录
    return submaps_dir_ + filename;
}

std::string SubMapCache::getSubMapWritePath(const SubMapID& id) const {
    std::string filename = SubMapSerializer::getSubMapFileName(id);
    
    // 工作模式：写入 work 目录
    if (work_mode_) {
        return submaps_work_dir_ + filename;
    }
    
    // 非工作模式：写入原始目录
    return submaps_dir_ + filename;
}

bool SubMapCache::subMapFileExists(const SubMapID& id) const {
    std::string filename = SubMapSerializer::getSubMapFileName(id);
    struct stat st;
    
    // 检查 work 目录
    if (work_mode_) {
        std::string work_path = submaps_work_dir_ + filename;
        if (stat(work_path.c_str(), &st) == 0) {
            return true;
        }
    }
    
    // 检查原始目录
    std::string orig_path = submaps_dir_ + filename;
    return stat(orig_path.c_str(), &st) == 0;
}

void SubMapCache::flushAllDirty() {
    // 将缓存中所有修改过的子图写入磁盘
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    SubMapSerializer serializer;
    int flushed = 0;
    
    for (auto& pair : cache_) {
        auto& submap = pair.second;
        if (submap && submap->isDirty()) {
            std::string path = getSubMapWritePath(pair.first);
            if (serializer.serialize(submap, path)) {
                submap->clearDirty();
                flushed++;
            }
        }
    }
    
    if (flushed > 0) {
        droslog(LogLevel::INFO, "SubMapCache::flushAllDirty() 刷盘完成: %d 个子图", flushed);
    }
}

