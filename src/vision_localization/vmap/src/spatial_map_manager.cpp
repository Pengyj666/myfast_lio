/*******************************************************
 * SpatialMapManager 类实现
 * 
 * 创建日期: 2025-12-10
 *******************************************************/

#include "spatial_map_manager.h"
#include "keyframe.h"
#include "submap_serializer.h"
#include "droslog/log.h"
#include "common/log_filters.h"
#include "common/sysutils.h"

#include <cmath>
#include <cstdio>
#include <ctime>
#include <dirent.h>
#include <algorithm>
#include <fstream>
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

// 辅助函数：检查文件是否存在
static bool FileExists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

// 辅助函数：获取当前时间戳
static double GetTimestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// 辅助函数：获取当前时间字符串
static std::string GetTimeStr() {
    time_t now = time(nullptr);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&now));
    return std::string(buf);
}

SpatialMapManager::SpatialMapManager() 
    : current_submap_{0, 0}
    , preload_running_(false)
    , max_cached_submaps_(25)
    , sliding_window_size_(400)     // 滑窗保留最近400帧
    , origin_position_(0, 0, 0)
    , has_origin_position_(false)
    , latest_keyframe_index_(0) {
}

SpatialMapManager::~SpatialMapManager() {
    shutdownCache();
}

// ========== 坐标转换工具函数 ==========

SubMapID SpatialMapManager::positionToSubMapID(const Eigen::Vector3d& position) {
    return SubMapID{
        static_cast<int>(std::floor(position.x() / SubMap::SUBMAP_SIZE)),
        static_cast<int>(std::floor(position.y() / SubMap::SUBMAP_SIZE))
    };
}

CellID SpatialMapManager::positionToCellID(const Eigen::Vector3d& position) {
    return CellID{
        static_cast<int>(std::floor(position.x() / Cell::CELL_SIZE)),
        static_cast<int>(std::floor(position.y() / Cell::CELL_SIZE))
    };
}

int SpatialMapManager::yawToDirectionSlot(double yaw) {
    // 将 yaw 归一化到 [0, 2π)
    double normalized_yaw = yaw;
    while (normalized_yaw < 0) normalized_yaw += 2 * M_PI;
    while (normalized_yaw >= 2 * M_PI) normalized_yaw -= 2 * M_PI;
    
    // 转换为度数并计算槽位（每60度一个槽位）
    double yaw_deg = normalized_yaw * 180.0 / M_PI;
    return static_cast<int>(yaw_deg / 60.0) % 6;
}

double SpatialMapManager::getYawFromRotation(const Eigen::Matrix3d& R) {
    // 从旋转矩阵提取 yaw 角（ZYX 欧拉角顺序）
    // yaw = atan2(R(1,0), R(0,0))
    return std::atan2(R(1, 0), R(0, 0));
}

// ========== 子图管理 ==========

std::shared_ptr<SubMap> SpatialMapManager::getOrCreateSubMap(const SubMapID& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = submaps_.find(id);
    if (it != submaps_.end()) {
        return it->second;
    }
    
    // 创建新子图
    auto submap = std::make_shared<SubMap>(id);
    submaps_[id] = submap;
    return submap;
}

std::shared_ptr<SubMap> SpatialMapManager::getSubMap(const SubMapID& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = submaps_.find(id);
    if (it != submaps_.end()) {
        return it->second;
    }
    return nullptr;
}

bool SpatialMapManager::isSubMapLoaded(const SubMapID& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return submaps_.find(id) != submaps_.end();
}

std::vector<SubMapID> SpatialMapManager::getAllSubMapIDs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<SubMapID> ids;
    ids.reserve(submaps_.size());
    for (const auto& pair : submaps_) {
        ids.push_back(pair.first);
    }
    return ids;
}

void SpatialMapManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    submaps_.clear();
}

std::vector<std::shared_ptr<KeyFrame>> SpatialMapManager::getAllIndexedKeyFrames() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<KeyFrame>> result;
    
    for (const auto& submap_pair : submaps_) {
        auto kfs = submap_pair.second->getAllKeyFrames();
        result.insert(result.end(), kfs.begin(), kfs.end());
    }
    
    return result;
}

// ========== 关键帧管理 ==========

bool SpatialMapManager::insertKeyFrame(std::shared_ptr<KeyFrame> kf) {
    if (!kf) return false;
    
    // ========== 自动计算空间索引 ==========
    // 使用 T_w_i（世界坐标系下的位置）计算
    SubMapID submap_id = positionToSubMapID(kf->T_w_i);
    CellID cell_id = positionToCellID(kf->T_w_i);
    
    kf->submap_x = submap_id.x;
    kf->submap_y = submap_id.y;
    kf->cell_x = cell_id.x;
    kf->cell_y = cell_id.y;
    
    // 从旋转矩阵提取 yaw 角并计算方向槽位
    double yaw = getYawFromRotation(kf->R_w_i);
    kf->direction_slot = yawToDirectionSlot(yaw);
    
    // 缓存当前索引位置（用于后续增量更新）
    kf->cached_submap_x = submap_id.x;
    kf->cached_submap_y = submap_id.y;
    kf->cached_cell_x = cell_id.x;
    kf->cached_cell_y = cell_id.y;
    kf->cached_direction_slot = kf->direction_slot;
    kf->index_dirty = false;
    
    // 插入到对应子图
    auto submap = getOrCreateSubMap(submap_id);
    std::shared_ptr<KeyFrame> replaced_kf;
    bool inserted = submap->tryInsertKeyFrame(kf, &replaced_kf);
    
    // 2026-01-11: 处理索引映射
    if (inserted) {
        // 如果替换了旧帧，先注销旧帧的索引
        if (replaced_kf) {
            unregisterKeyFrameIndex(replaced_kf->index);
        }
        // 注册新帧到索引映射（用于 DBoW2 查询后快速定位）
        registerKeyFrameIndex(kf);
    }
    
    return inserted;
}

// ========== 增量索引更新实现 - 2025-12-15 ==========

void SpatialMapManager::markDirty(std::shared_ptr<KeyFrame> kf) {
    if (!kf) return;
    
    kf->index_dirty = true;
    
    std::lock_guard<std::mutex> lock(dirty_mutex_);
    dirty_keyframes_.insert(kf);
}

void SpatialMapManager::markDirtyBatch(const std::vector<std::shared_ptr<KeyFrame>>& kfs) {
    std::lock_guard<std::mutex> lock(dirty_mutex_);
    for (auto& kf : kfs) {
        if (kf) {
            kf->index_dirty = true;
            dirty_keyframes_.insert(kf);
        }
    }
}

bool SpatialMapManager::needsReindex(std::shared_ptr<KeyFrame> kf) const {
    if (!kf) return false;
    
    // 计算优化后应该属于的 Cell 和方向槽位
    CellID new_cell = positionToCellID(kf->T_w_i);
    double yaw = getYawFromRotation(kf->R_w_i);
    int new_slot = yawToDirectionSlot(yaw);
    
    // 检查是否与缓存的索引位置不同
    return (new_cell.x != kf->cached_cell_x) || 
           (new_cell.y != kf->cached_cell_y) ||
           (new_slot != kf->cached_direction_slot);
}

void SpatialMapManager::removeFromOldIndex(std::shared_ptr<KeyFrame> kf) {
    if (!kf) return;
    
    // 使用缓存的索引位置查找并移除
    SubMapID old_submap_id{kf->cached_submap_x, kf->cached_submap_y};
    
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = submaps_.find(old_submap_id);
    if (it != submaps_.end()) {
        it->second->removeKeyFrame(kf);
    }
}

void SpatialMapManager::insertToNewIndex(std::shared_ptr<KeyFrame> kf) {
    if (!kf) return;
    
    // 计算新的索引位置
    SubMapID new_submap_id = positionToSubMapID(kf->T_w_i);
    CellID new_cell_id = positionToCellID(kf->T_w_i);
    double yaw = getYawFromRotation(kf->R_w_i);
    int new_slot = yawToDirectionSlot(yaw);
    
    // 更新关键帧的索引信息
    kf->submap_x = new_submap_id.x;
    kf->submap_y = new_submap_id.y;
    kf->cell_x = new_cell_id.x;
    kf->cell_y = new_cell_id.y;
    kf->direction_slot = new_slot;
    
    // 更新缓存
    kf->cached_submap_x = new_submap_id.x;
    kf->cached_submap_y = new_submap_id.y;
    kf->cached_cell_x = new_cell_id.x;
    kf->cached_cell_y = new_cell_id.y;
    kf->cached_direction_slot = new_slot;
    
    // 插入到新位置（使用 forceInsert 避免被拒绝）
    auto submap = getOrCreateSubMap(new_submap_id);
    submap->forceInsertKeyFrame(kf);
}

int SpatialMapManager::rebuildDirtyIndices() {
    std::vector<std::shared_ptr<KeyFrame>> to_reindex;
    
    // 收集需要重新索引的帧
    {
        std::lock_guard<std::mutex> lock(dirty_mutex_);
        for (auto& kf : dirty_keyframes_) {
            if (kf && kf->index_dirty && needsReindex(kf)) {
                to_reindex.push_back(kf);
            } else if (kf) {
                // 不需要移动，只清除脏标记
                kf->index_dirty = false;
            }
        }
        dirty_keyframes_.clear();
    }
    
    // 执行重新索引
    int moved_count = 0;
    for (auto& kf : to_reindex) {
        // 从旧位置移除
        removeFromOldIndex(kf);
        
        // 插入到新位置
        insertToNewIndex(kf);
        
        kf->index_dirty = false;
        moved_count++;
    }
    
    return moved_count;
}

std::vector<std::shared_ptr<SubMap>> SpatialMapManager::getDirtySubMaps() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::shared_ptr<SubMap>> result;
    for (const auto& pair : submaps_) {
        if (pair.second->isDirty()) {
            result.push_back(pair.second);
        }
    }
    return result;
}

int SpatialMapManager::saveDirtySubMaps(const std::string& dir_path) {
    // 确保目录存在
    std::string spatial_dir = dir_path;
    if (spatial_dir.back() != '/') spatial_dir += '/';
    spatial_dir += "spatial/";
    
    if (!DirExists(spatial_dir.c_str())) {
        MakeDir(spatial_dir.c_str());
    }
    
    std::string submaps_dir = spatial_dir + "submaps/";
    if (!DirExists(submaps_dir.c_str())) {
        MakeDir(submaps_dir.c_str());
    }
    
    // 只保存脏的子图
    SubMapSerializer serializer;
    int success_count = 0;
    
    auto dirty_submaps = getDirtySubMaps();
    
    for (auto& submap : dirty_submaps) {
        std::string file_path = submaps_dir + SubMapSerializer::getSubMapFileName(submap->id());
        if (serializer.serialize(submap, file_path)) {
            submap->clearDirty();
            success_count++;
        }
    }
    
    return success_count;
}

void SpatialMapManager::clearAllDirtyFlags() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& pair : submaps_) {
        pair.second->clearDirty();
    }
    
    {
        std::lock_guard<std::mutex> dlock(dirty_mutex_);
        for (auto& kf : dirty_keyframes_) {
            if (kf) kf->index_dirty = false;
        }
        dirty_keyframes_.clear();
    }
}

std::vector<std::shared_ptr<KeyFrame>> SpatialMapManager::queryKeyFrames(
    const Eigen::Vector3d& position, double radius) const {
    
    std::vector<std::shared_ptr<KeyFrame>> result;
    std::vector<SubMapID> need_reload;  // 需要重新加载的子图
    
    // 计算搜索范围覆盖的子图
    int min_sx = static_cast<int>(std::floor((position.x() - radius) / SubMap::SUBMAP_SIZE));
    int max_sx = static_cast<int>(std::floor((position.x() + radius) / SubMap::SUBMAP_SIZE));
    int min_sy = static_cast<int>(std::floor((position.y() - radius) / SubMap::SUBMAP_SIZE));
    int max_sy = static_cast<int>(std::floor((position.y() + radius) / SubMap::SUBMAP_SIZE));
    
    double radius_sq = radius * radius;
    
    // 第一遍：查询已加载的子图，收集需要重新加载的子图
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (int sx = min_sx; sx <= max_sx; sx++) {
            for (int sy = min_sy; sy <= max_sy; sy++) {
                SubMapID sid{sx, sy};
                auto it = submaps_.find(sid);
                if (it != submaps_.end()) {
                    // 子图已加载，获取关键帧
                    auto kfs = it->second->getAllKeyFrames();
                    for (auto& kf : kfs) {
                        double dx = kf->T_w_i.x() - position.x();
                        double dy = kf->T_w_i.y() - position.y();
                        if (dx * dx + dy * dy <= radius_sq) {
                            result.push_back(kf);
                        }
                    }
                } else if (evicted_submaps_.find(sid) != evicted_submaps_.end()) {
                    // 子图曾被淘汰，需要重新加载
                    need_reload.push_back(sid);
                }
            }
        }
    }
    
    // 第二遍：同步重新加载被淘汰的子图（回环/重定位场景）
    if (!need_reload.empty()) {
        static SimpleLogFilter reload_filter(5000);  // 5秒一次日志
        if (reload_filter.Output(GetNow_Steady())) {
            droslog(LogLevel::INFO, "SpatialMapManager::queryKeyFrames() 按需重新加载 %d 个被淘汰子图",
                    (int)need_reload.size());
        }
        
        for (const auto& sid : need_reload) {
            // 调用非 const 版本的加载函数（需要 const_cast）
            auto* self = const_cast<SpatialMapManager*>(this);
            auto submap = self->loadSubMapSync(sid);
            
            if (submap) {
                auto kfs = submap->getAllKeyFrames();
                for (auto& kf : kfs) {
                    double dx = kf->T_w_i.x() - position.x();
                    double dy = kf->T_w_i.y() - position.y();
                    if (dx * dx + dy * dy <= radius_sq) {
                        result.push_back(kf);
                    }
                }
            }
        }
    }
    
    return result;
}

std::vector<std::shared_ptr<KeyFrame>> SpatialMapManager::queryKeyFramesByPose(
    const Eigen::Vector3d& position, double yaw, double radius) const {
    
    std::vector<std::shared_ptr<KeyFrame>> result;
    std::vector<SubMapID> need_reload;  // 需要重新加载的子图
    
    // 计算目标方向槽位及相邻槽位（允许 ±1 方向）
    int target_slot = yawToDirectionSlot(yaw);
    int slot_prev = (target_slot + 5) % 6;  // 前一个槽位
    int slot_next = (target_slot + 1) % 6;  // 后一个槽位
    
    // 计算搜索范围覆盖的子图
    int min_sx = static_cast<int>(std::floor((position.x() - radius) / SubMap::SUBMAP_SIZE));
    int max_sx = static_cast<int>(std::floor((position.x() + radius) / SubMap::SUBMAP_SIZE));
    int min_sy = static_cast<int>(std::floor((position.y() - radius) / SubMap::SUBMAP_SIZE));
    int max_sy = static_cast<int>(std::floor((position.y() + radius) / SubMap::SUBMAP_SIZE));
    
    double radius_sq = radius * radius;
    
    // 第一遍：查询已加载的子图，收集需要重新加载的子图
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (int sx = min_sx; sx <= max_sx; sx++) {
            for (int sy = min_sy; sy <= max_sy; sy++) {
                SubMapID sid{sx, sy};
                auto it = submaps_.find(sid);
                if (it != submaps_.end()) {
                    // 子图已加载，获取关键帧
                    for (int slot : {target_slot, slot_prev, slot_next}) {
                        auto kfs = it->second->getKeyFramesByDirection(slot);
                        for (auto& kf : kfs) {
                            double dx = kf->T_w_i.x() - position.x();
                            double dy = kf->T_w_i.y() - position.y();
                            if (dx * dx + dy * dy <= radius_sq) {
                                result.push_back(kf);
                            }
                        }
                    }
                } else if (evicted_submaps_.find(sid) != evicted_submaps_.end()) {
                    // 子图曾被淘汰，需要重新加载
                    need_reload.push_back(sid);
                }
            }
        }
    }
    
    // 第二遍：同步重新加载被淘汰的子图（回环/重定位场景）
    if (!need_reload.empty()) {
        static SimpleLogFilter reload_filter2(5000);  // 5秒一次日志
        if (reload_filter2.Output(GetNow_Steady())) {
            droslog(LogLevel::INFO, "SpatialMapManager::queryKeyFramesByPose() 按需重新加载 %d 个被淘汰子图",
                    (int)need_reload.size());
        }
        
        for (const auto& sid : need_reload) {
            auto* self = const_cast<SpatialMapManager*>(this);
            auto submap = self->loadSubMapSync(sid);
            
            if (submap) {
                for (int slot : {target_slot, slot_prev, slot_next}) {
                    auto kfs = submap->getKeyFramesByDirection(slot);
                    for (auto& kf : kfs) {
                        double dx = kf->T_w_i.x() - position.x();
                        double dy = kf->T_w_i.y() - position.y();
                        if (dx * dx + dy * dy <= radius_sq) {
                            result.push_back(kf);
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

// ========== 统计信息 ==========

int SpatialMapManager::getSubMapCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(submaps_.size());
}

int SpatialMapManager::getTotalKeyFrameCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    int count = 0;
    for (const auto& pair : submaps_) {
        count += pair.second->getKeyFrameCount();
    }
    return count;
}

int SpatialMapManager::getTotalCellCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    int count = 0;
    for (const auto& pair : submaps_) {
        count += pair.second->getCellCount();
    }
    return count;
}

void SpatialMapManager::printStatistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    int total_cells = 0;
    int total_kfs = 0;
    for (const auto& pair : submaps_) {
        total_cells += pair.second->getCellCount();
        total_kfs += pair.second->getKeyFrameCount();
    }
    
    printf("========== SpatialMapManager Statistics ==========\n");
    printf("  SubMaps: %d\n", static_cast<int>(submaps_.size()));
    printf("  Cells: %d\n", total_cells);
    printf("  KeyFrames: %d\n", total_kfs);
    printf("==================================================\n");
}

// ========== 序列化（保存/加载）==========

int SpatialMapManager::saveToDirectory(const std::string& dir_path,
                                        double origin_lat,
                                        double origin_lon,
                                        double origin_alt) {
    // 确保目录存在
    std::string spatial_dir = dir_path;
    if (spatial_dir.back() != '/') spatial_dir += '/';
    spatial_dir += "spatial/";
    
    if (!DirExists(spatial_dir.c_str())) {
        MakeDir(spatial_dir.c_str());
    }
    
    std::string submaps_dir = spatial_dir + "submaps/";
    if (!DirExists(submaps_dir.c_str())) {
        MakeDir(submaps_dir.c_str());
    }
    
    // 保存所有子图
    SubMapSerializer serializer;
    int success_count = 0;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (const auto& pair : submaps_) {
        std::string file_path = submaps_dir + SubMapSerializer::getSubMapFileName(pair.first);
        if (serializer.serialize(pair.second, file_path)) {
            success_count++;
        }
    }
    
    // 生成并保存元信息
    SpatialMapMeta meta = generateMeta(origin_lat, origin_lon, origin_alt);
    std::string meta_path = spatial_dir + "spatial_meta.txt";
    SpatialMetaIO::save(meta, meta_path);
    
    return success_count;
}

int SpatialMapManager::loadFromDirectory(const std::string& dir_path) {
    std::string spatial_dir = dir_path;
    if (spatial_dir.back() != '/') spatial_dir += '/';
    spatial_dir += "spatial/";
    
    std::string submaps_dir = spatial_dir + "submaps/";
    
    // ========== 修复 - 2026-01-07 ==========
    // 保存子图目录路径，用于按需重新加载
    submaps_dir_ = submaps_dir;
    
    if (!DirExists(submaps_dir.c_str())) {
        return 0;
    }
    
    // 加载元信息（可选）
    std::string meta_path = spatial_dir + "spatial_meta.txt";
    SpatialMapMeta meta;
    if (FileExists(meta_path.c_str())) {
        SpatialMetaIO::load(meta_path, meta);
    }
    
    // 扫描目录中的 .smap 文件
    DIR* dir = opendir(submaps_dir.c_str());
    if (!dir) {
        return 0;
    }
    
    std::vector<std::string> smap_files;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.size() > 5 && filename.substr(filename.size() - 5) == ".smap") {
            smap_files.push_back(submaps_dir + filename);
        }
    }
    closedir(dir);
    
    // 清空现有数据
    {
        std::lock_guard<std::mutex> lock(mutex_);
        submaps_.clear();
        evicted_submaps_.clear();  // 同时清空淘汰集合
    }
    
    // 2026-01-11: 清空索引映射
    {
        std::lock_guard<std::mutex> lock(index_mutex_);
        index_to_keyframe_.clear();
    }
    
    // 加载每个子图
    SubMapSerializer serializer;
    int success_count = 0;
    int kf_count = 0;
    
    for (const auto& file_path : smap_files) {
        auto submap = serializer.deserialize(file_path);
        if (submap) {
            std::lock_guard<std::mutex> lock(mutex_);
            submaps_[submap->id()] = submap;
            success_count++;
            
            // 2026-01-11: 注册所有关键帧到索引映射
            auto all_kfs = submap->getAllKeyFrames();
            for (auto& kf : all_kfs) {
                if (kf) {
                    registerKeyFrameIndex(kf);
                    kf_count++;
                }
            }
        }
    }
    
    droslog(LogLevel::INFO, "SpatialMapManager::loadFromDirectory() 加载完成: %d 子图, %d 关键帧已注册到索引", 
            success_count, kf_count);
    
    return success_count;
}

bool SpatialMapManager::loadSubMap(const std::string& file_path) {
    SubMapSerializer serializer;
    auto submap = serializer.deserialize(file_path);
    if (!submap) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    submaps_[submap->id()] = submap;
    return true;
}

bool SpatialMapManager::unloadSubMap(const SubMapID& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = submaps_.find(id);
    if (it == submaps_.end()) {
        return false;
    }
    
    submaps_.erase(it);
    return true;
}

void SpatialMapManager::addSubMap(std::shared_ptr<SubMap> submap) {
    if (!submap) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    submaps_[submap->id()] = submap;
}

std::vector<std::shared_ptr<SubMap>> SpatialMapManager::getAllSubMaps() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::shared_ptr<SubMap>> result;
    result.reserve(submaps_.size());
    for (const auto& pair : submaps_) {
        result.push_back(pair.second);
    }
    return result;
}

SpatialMapMeta SpatialMapManager::generateMeta(double origin_lat,
                                                double origin_lon,
                                                double origin_alt) const {
    // 注意：调用者已持有锁，或者这是 const 方法自己加锁
    SpatialMapMeta meta;
    
    meta.origin_lat = origin_lat;
    meta.origin_lon = origin_lon;
    meta.origin_alt = origin_alt;
    
    meta.cell_size = Cell::CELL_SIZE;
    meta.submap_size = SubMap::SUBMAP_SIZE;
    meta.num_directions = Cell::NUM_DIRECTIONS;
    
    meta.create_timestamp = GetTimestamp();
    meta.create_time_str = GetTimeStr();
    
    // 统计并收集子图元信息
    meta.total_submaps = static_cast<int>(submaps_.size());
    meta.total_keyframes = 0;
    meta.total_cells = 0;
    
    bool first = true;
    
    for (const auto& pair : submaps_) {
        SubMapMeta sm;
        sm.id = pair.first;
        sm.keyframe_count = pair.second->getKeyFrameCount();
        sm.cell_count = pair.second->getCellCount();
        sm.file_name = SubMapSerializer::getSubMapFileName(pair.first);
        
        // 计算子图边界
        sm.min_x = pair.first.x * SubMap::SUBMAP_SIZE;
        sm.max_x = sm.min_x + SubMap::SUBMAP_SIZE;
        sm.min_y = pair.first.y * SubMap::SUBMAP_SIZE;
        sm.max_y = sm.min_y + SubMap::SUBMAP_SIZE;
        
        meta.submap_metas.push_back(sm);
        
        meta.total_keyframes += sm.keyframe_count;
        meta.total_cells += sm.cell_count;
        
        // 更新全局边界
        if (first) {
            meta.map_min_x = sm.min_x;
            meta.map_max_x = sm.max_x;
            meta.map_min_y = sm.min_y;
            meta.map_max_y = sm.max_y;
            first = false;
        } else {
            meta.map_min_x = std::min(meta.map_min_x, sm.min_x);
            meta.map_max_x = std::max(meta.map_max_x, sm.max_x);
            meta.map_min_y = std::min(meta.map_min_y, sm.min_y);
            meta.map_max_y = std::max(meta.map_max_y, sm.max_y);
        }
    }
    
    return meta;
}

void SpatialMapManager::rebuildSpatialIndex() {
    // 全局优化后，关键帧位姿已改变，需要重建空间索引
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 收集所有关键帧
    std::vector<std::shared_ptr<KeyFrame>> all_keyframes;
    for (auto& pair : submaps_) {
        auto kfs = pair.second->getAllKeyFrames();
        all_keyframes.insert(all_keyframes.end(), kfs.begin(), kfs.end());
    }
    
    // 清空现有索引
    submaps_.clear();
    
    // 重新插入所有关键帧
    for (auto& kf : all_keyframes) {
        // 重新计算空间索引
        SubMapID new_submap_id = positionToSubMapID(kf->T_w_i);
        CellID new_cell_id = positionToCellID(kf->T_w_i);
        int new_direction = yawToDirectionSlot(getYawFromRotation(kf->R_w_i));
        
        // 更新关键帧的空间索引信息
        kf->submap_x = new_submap_id.x;
        kf->submap_y = new_submap_id.y;
        kf->cell_x = new_cell_id.x;
        kf->cell_y = new_cell_id.y;
        kf->direction_slot = new_direction;
        
        // 获取或创建子图
        auto& submap = submaps_[new_submap_id];
        if (!submap) {
            submap = std::make_shared<SubMap>(new_submap_id);
        }
        
        // 插入到子图
        submap->forceInsertKeyFrame(kf);
    }
    
    droslog(LogLevel::INFO, "SpatialMapManager::rebuildSpatialIndex() 重建完成: %zu 关键帧, %zu 子图",
            all_keyframes.size(), submaps_.size());
}

int SpatialMapManager::mergeWorkToSubmaps(const std::string& work_dir, const std::string& submaps_dir) {
    // 将 work 目录中的子图复制/覆盖到 submaps 目录
    // 这实现了"增量更新合并到原始地图"的功能
    
    if (!DirExists(work_dir.c_str())) {
        return 0;
    }
    
    if (!DirExists(submaps_dir.c_str())) {
        MakeDir(submaps_dir.c_str());
    }
    
    DIR* dir = opendir(work_dir.c_str());
    if (!dir) {
        return 0;
    }
    
    int merged_count = 0;
    struct dirent* entry;
    
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        // 只处理 .smap 文件
        if (filename.size() < 5 || filename.substr(filename.size() - 5) != ".smap") {
            continue;
        }
        
        std::string src_path = work_dir + filename;
        std::string dst_path = submaps_dir + filename;
        
        // 复制文件（覆盖已有）
        std::ifstream src(src_path, std::ios::binary);
        std::ofstream dst(dst_path, std::ios::binary);
        
        if (src && dst) {
            dst << src.rdbuf();
            merged_count++;
        }
    }
    
    closedir(dir);
    
    droslog(LogLevel::INFO, "SpatialMapManager::mergeWorkToSubmaps() 合并完成: %d 个子图", merged_count);
    
    return merged_count;
}

// ========== 分层缓存：轻量元数据索引 - 2025-12-30 ==========

void SpatialMapManager::addKeyFrameMetadata(const KeyFrameMetadata& meta) {
    std::lock_guard<std::mutex> lock(meta_mutex_);
    kf_metadata_[meta.index] = meta;
}

std::vector<KeyFrameMetadata> SpatialMapManager::queryMetadataByPosition(
    const Eigen::Vector3d& position, double radius) const {
    
    std::lock_guard<std::mutex> lock(meta_mutex_);
    std::vector<KeyFrameMetadata> result;
    
    double radius_sq = radius * radius;
    for (const auto& pair : kf_metadata_) {
        const auto& meta = pair.second;
        double dx = meta.pos_x - position.x();
        double dy = meta.pos_y - position.y();
        double dist_sq = dx * dx + dy * dy;
        
        if (dist_sq <= radius_sq) {
            result.push_back(meta);
        }
    }
    
    return result;
}

std::vector<KeyFrameMetadata> SpatialMapManager::queryMetadataByPose(
    const Eigen::Vector3d& position, double yaw, double radius) const {
    
    std::lock_guard<std::mutex> lock(meta_mutex_);
    std::vector<KeyFrameMetadata> result;
    
    // 计算目标方向槽位及相邻槽位（允许 ±1 方向）
    int target_slot = yawToDirectionSlot(yaw);
    int slot_prev = (target_slot + 5) % 6;
    int slot_next = (target_slot + 1) % 6;
    
    // 同时考虑相反方向（180度）
    int opposite_slot = (target_slot + 3) % 6;
    int opp_prev = (opposite_slot + 5) % 6;
    int opp_next = (opposite_slot + 1) % 6;
    
    double radius_sq = radius * radius;
    for (const auto& pair : kf_metadata_) {
        const auto& meta = pair.second;
        double dx = meta.pos_x - position.x();
        double dy = meta.pos_y - position.y();
        double dist_sq = dx * dx + dy * dy;
        
        if (dist_sq > radius_sq) continue;
        
        // 检查方向是否匹配（包含正方向和反方向）
        int slot = meta.direction_slot;
        if (slot == target_slot || slot == slot_prev || slot == slot_next ||
            slot == opposite_slot || slot == opp_prev || slot == opp_next) {
            result.push_back(meta);
        }
    }
    
    return result;
}

int SpatialMapManager::evictDistantSubMaps(const Eigen::Vector3d& current_pos, int max_range,
                                           std::vector<SubMapID>* evicted_ids) {
    if (!work_mode_) return 0;
    
    SubMapID current_submap = positionToSubMapID(current_pos);
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<SubMapID> to_evict;
    for (const auto& pair : submaps_) {
        int dx = std::abs(pair.first.x - current_submap.x);
        int dy = std::abs(pair.first.y - current_submap.y);
        int chebyshev_dist = std::max(dx, dy);
        
        if (chebyshev_dist > max_range) {
            to_evict.push_back(pair.first);
        }
    }
    
    for (const auto& id : to_evict) {
        // ========== 修复 - 2026-01-07 ==========
        // 不再清空关键帧数据，直接删除子图
        // 原因：如果 SubMapCache 持有同一个 shared_ptr，清空数据会污染缓存
        // 调用者需要同步淘汰 SubMapCache 中的对应子图
        
        // 标记为已淘汰，用于后续按需重新加载
        evicted_submaps_.insert(id);
        submaps_.erase(id);
    }
    
    // 返回被淘汰的子图 ID（用于同步淘汰 SubMapCache）
    if (evicted_ids) {
        *evicted_ids = to_evict;
    }
    
    if (!to_evict.empty()) {
        droslog(LogLevel::INFO, "SpatialMapManager::evictDistantSubMaps() 淘汰 %d 个远离子图 (当前:%d,%d, 范围:%d)",
                (int)to_evict.size(), current_submap.x, current_submap.y, max_range);
    }
    
    return static_cast<int>(to_evict.size());
}

// ========== 统一子图管理（原 SubMapCache 功能）- 2026-01-07 ==========

void SpatialMapManager::initializeCache(const std::string& map_dir, int max_cached_submaps) {
    map_dir_ = map_dir;
    if (!map_dir_.empty() && map_dir_.back() != '/') map_dir_ += '/';
    
    // 设置目录路径
    submaps_dir_ = map_dir_ + "spatial/submaps/";
    submaps_work_dir_ = map_dir_ + "spatial/submaps_work/";
    
    max_cached_submaps_ = max_cached_submaps;
    
    // 确保工作目录存在
    if (work_mode_) {
        std::string spatial_dir = map_dir_ + "spatial/";
        if (!DirExists(spatial_dir.c_str())) {
            MakeDir(spatial_dir.c_str());
        }
        if (!DirExists(submaps_work_dir_.c_str())) {
            MakeDir(submaps_work_dir_.c_str());
        }
    }
    
    // 启动后台预加载线程
    preload_running_ = true;
    preload_thread_ = std::thread(&SpatialMapManager::preloadThreadFunc, this);
    
    droslog(LogLevel::INFO, "SpatialMapManager::initializeCache() 初始化完成, map_dir=%s, max_cached=%d",
            map_dir_.c_str(), max_cached_submaps_);
}

void SpatialMapManager::shutdownCache() {
    if (!preload_running_) return;
    
    preload_running_ = false;
    queue_cv_.notify_all();
    
    if (preload_thread_.joinable()) {
        preload_thread_.join();
    }
    
    // 清空预加载队列
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!preload_queue_.empty()) preload_queue_.pop();
        pending_loads_.clear();
    }
    
    droslog(LogLevel::INFO, "SpatialMapManager::shutdownCache() 预加载系统已关闭");
}

void SpatialMapManager::updatePosition(const Eigen::Vector3d& position) {
    // 计算当前位置所在的子图
    SubMapID new_submap = positionToSubMapID(position);
    
    // 只有当子图 ID 发生变化时才执行后续操作
    SubMapID old_submap;
    {
        std::lock_guard<std::mutex> lock(position_mutex_);
        old_submap = current_submap_;
        current_submap_ = new_submap;
    }
    
    // 子图未变化，跳过
    if (new_submap.x == old_submap.x && new_submap.y == old_submap.y) {
        return;
    }
    
    // 子图变化时触发淘汰和预加载
    {
        std::lock_guard<std::mutex> lock(mutex_);
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
        droslog(LogLevel::INFO, "SpatialMapManager: 子图切换 (%d,%d)->(%d,%d), 当前缓存 %d 个",
                old_submap.x, old_submap.y, new_submap.x, new_submap.y, getLoadedSubMapCount());
    }
}

void SpatialMapManager::initialLoadByPosition(const Eigen::Vector3d& position, double radius) {
    // 计算当前位置所在的子图
    SubMapID center = positionToSubMapID(position);
    
    // 计算需要加载的格数范围
    int range = static_cast<int>(std::ceil(radius / SubMap::SUBMAP_SIZE));
    
    // 按距离排序，优先加载近的
    std::vector<std::pair<int, SubMapID>> sorted_submaps;
    
    for (int dx = -range; dx <= range; dx++) {
        for (int dy = -range; dy <= range; dy++) {
            SubMapID id{center.x + dx, center.y + dy};
            int dist = std::abs(dx) + std::abs(dy);  // 曼哈顿距离
            sorted_submaps.push_back({dist, id});
        }
    }
    
    std::sort(sorted_submaps.begin(), sorted_submaps.end(),
        [](const std::pair<int, SubMapID>& a, const std::pair<int, SubMapID>& b) {
            return a.first < b.first;
        });
    
    // 同步加载最近的几个子图，其余异步加载
    int sync_count = std::min(4, static_cast<int>(sorted_submaps.size()));
    
    for (int i = 0; i < sync_count; i++) {
        loadSubMapSync(sorted_submaps[i].second);
    }
    
    for (size_t i = sync_count; i < sorted_submaps.size(); i++) {
        prefetch(sorted_submaps[i].second, 10 - static_cast<int>(i));
    }
    
    droslog(LogLevel::INFO, "SpatialMapManager::initialLoadByPosition() 初始加载: 同步=%d, 异步=%d",
            sync_count, (int)sorted_submaps.size() - sync_count);
}

bool SpatialMapManager::isSubMapEvicted(const SubMapID& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return evicted_submaps_.find(id) != evicted_submaps_.end();
}

SubMapCacheStats SpatialMapManager::getCacheStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return cache_stats_;
}

void SpatialMapManager::flushAllDirty() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    SubMapSerializer serializer;
    int flushed = 0;
    
    for (auto& pair : submaps_) {
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
        droslog(LogLevel::INFO, "SpatialMapManager::flushAllDirty() 刷盘完成: %d 个子图", flushed);
    }
}

// ========== 预加载系统私有方法 ==========

void SpatialMapManager::prefetch(const SubMapID& id, int priority) {
    // 检查是否已缓存
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (submaps_.find(id) != submaps_.end()) {
            return;  // 已在缓存中
        }
    }
    
    // 检查文件是否存在
    if (!subMapFileExists(id)) {
        return;
    }
    
    // 添加到预加载队列
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // 检查是否已在等待队列中（去重）
        if (pending_loads_.find(id) != pending_loads_.end()) {
            return;
        }
        
        PreloadTask task;
        task.id = id;
        task.priority = priority;
        
        preload_queue_.push(task);
        pending_loads_.insert(id);
    }
    
    queue_cv_.notify_one();
}

void SpatialMapManager::preloadThreadFunc() {
    while (preload_running_) {
        PreloadTask task;
        
        // 等待任务
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !preload_running_ || !preload_queue_.empty();
            });
            
            if (!preload_running_) break;
            if (preload_queue_.empty()) continue;
            
            task = preload_queue_.top();
            preload_queue_.pop();
            pending_loads_.erase(task.id);
        }
        
        // 再次检查是否已缓存
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (submaps_.find(task.id) != submaps_.end()) {
                continue;
            }
        }
        
        // 加载子图
        std::string file_path = getSubMapFilePath(task.id);
        SubMapSerializer serializer;
        auto submap = serializer.deserialize(file_path);
        
        if (submap) {
            std::lock_guard<std::mutex> lock(mutex_);
            
            // 淘汰超出范围的子图
            evictOutOfRange();
            
            submaps_[task.id] = submap;
            evicted_submaps_.erase(task.id);  // 如果之前被淘汰过，现在重新加载
            
            {
                std::lock_guard<std::mutex> slock(stats_mutex_);
                cache_stats_.load_count++;
            }
        }
    }
}

int SpatialMapManager::evictOutOfRange() {
    // 注意：调用者已持有 mutex_
    // 2026-01-08: 空间+滑窗+起点 三重保留机制
    
    if (submaps_.empty()) return 0;
    
    SubMapID cur_submap;
    {
        std::lock_guard<std::mutex> lock(position_mutex_);
        cur_submap = current_submap_;
    }
    
    // 计算起点所在的子图（如果已设置）
    SubMapID origin_submap{0, 0};
    if (has_origin_position_) {
        origin_submap = positionToSubMapID(origin_position_);
    }
    
    // 滑窗保留的最小索引（原子读取）
    int latest_idx = latest_keyframe_index_.load();
    int min_sliding_index = std::max(0, latest_idx - sliding_window_size_);
    
    std::vector<SubMapID> to_evict;
    
    for (const auto& pair : submaps_) {
        const SubMapID& sid = pair.first;
        
        // 1. 空间保留：中心 + 周围8格（3×3范围）
        bool in_spatial_range = (std::abs(sid.x - cur_submap.x) <= 1 && 
                                 std::abs(sid.y - cur_submap.y) <= 1);
        if (in_spatial_range) continue;
        
        // 2. 起点保留：起点 + 周围24格（5×5范围，±2格）
        if (has_origin_position_) {
            bool in_origin_range = (std::abs(sid.x - origin_submap.x) <= 2 && 
                                    std::abs(sid.y - origin_submap.y) <= 2);
            if (in_origin_range) continue;
        }
        
        // 3. 滑窗保留：检查子图内是否有最近400帧内的关键帧
        bool has_sliding_window_kf = false;
        auto all_kfs = pair.second->getAllKeyFrames();
        for (const auto& kf : all_kfs) {
            if (kf && kf->index >= min_sliding_index) {
                has_sliding_window_kf = true;
                break;
            }
        }
        if (has_sliding_window_kf) continue;
        
        // 不在任何保留范围内，淘汰
        to_evict.push_back(sid);
    }
    
    // 执行淘汰
    for (const auto& id : to_evict) {
        evicted_submaps_.insert(id);
        submaps_.erase(id);
    }
    
    if (!to_evict.empty()) {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        cache_stats_.evict_count += static_cast<int>(to_evict.size());
        
        droslog(LogLevel::INFO, "SpatialMapManager: 淘汰 %d 个子图, 当前缓存 %d 个 "
                "(空间3×3, 起点5×5, 滑窗%d帧)",
                (int)to_evict.size(), (int)submaps_.size(), sliding_window_size_);
    }
    
    return static_cast<int>(to_evict.size());
}

std::shared_ptr<SubMap> SpatialMapManager::loadSubMapSync(const SubMapID& id) {
    // 先检查缓存
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = submaps_.find(id);
        if (it != submaps_.end()) {
            std::lock_guard<std::mutex> slock(stats_mutex_);
            cache_stats_.hit_count++;
            return it->second;
        }
    }
    
    // 缓存未命中
    {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        cache_stats_.miss_count++;
    }
    
    // 同步加载
    std::string file_path = getSubMapFilePath(id);
    struct stat st;
    if (stat(file_path.c_str(), &st) != 0) {
        return nullptr;  // 文件不存在
    }
    
    SubMapSerializer serializer;
    auto submap = serializer.deserialize(file_path);
    
    if (submap) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 淘汰超出范围的子图
        evictOutOfRange();
        
        // 如果仍超出数量限制，淘汰最远的
        SubMapID cur_submap;
        {
            std::lock_guard<std::mutex> plock(position_mutex_);
            cur_submap = current_submap_;
        }
        
        while (static_cast<int>(submaps_.size()) >= max_cached_submaps_) {
            if (submaps_.empty()) break;
            int max_dist = -1;
            SubMapID farthest_id;
            for (const auto& p : submaps_) {
                int dx = std::abs(p.first.x - cur_submap.x);
                int dy = std::abs(p.first.y - cur_submap.y);
                int d = std::max(dx, dy);
                if (d > max_dist) { max_dist = d; farthest_id = p.first; }
            }
            if (max_dist >= 0) {
                evicted_submaps_.insert(farthest_id);
                submaps_.erase(farthest_id);
                std::lock_guard<std::mutex> slock(stats_mutex_);
                cache_stats_.evict_count++;
            } else break;
        }
        
        submaps_[id] = submap;
        evicted_submaps_.erase(id);
        
        {
            std::lock_guard<std::mutex> slock(stats_mutex_);
            cache_stats_.load_count++;
        }
    }
    
    return submap;
}

std::string SpatialMapManager::getSubMapFilePath(const SubMapID& id) const {
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

std::string SpatialMapManager::getSubMapWritePath(const SubMapID& id) const {
    std::string filename = SubMapSerializer::getSubMapFileName(id);
    
    // 工作模式：写入 work 目录
    if (work_mode_) {
        return submaps_work_dir_ + filename;
    }
    
    return submaps_dir_ + filename;
}

bool SpatialMapManager::subMapFileExists(const SubMapID& id) const {
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

bool SpatialMapManager::subMapExistsInOriginal(const SubMapID& id) const {
    std::string filename = SubMapSerializer::getSubMapFileName(id);
    std::string orig_path = submaps_dir_ + filename;
    struct stat st;
    return stat(orig_path.c_str(), &st) == 0;
}

// ========== 滑窗淘汰机制 - 2026-01-08 ==========

void SpatialMapManager::setOriginPosition(const Eigen::Vector3d& position) {
    origin_position_ = position;
    has_origin_position_ = true;
    
    droslog(LogLevel::INFO, "SpatialMapManager: 设置起点位置 (%.2f, %.2f, %.2f), 保留范围 5×5 格",
            position.x(), position.y(), position.z());
}

void SpatialMapManager::updateLatestKeyFrameIndex(int index) {
    // 原子操作：只有当新索引更大时才更新
    int current = latest_keyframe_index_.load();
    while (index > current) {
        if (latest_keyframe_index_.compare_exchange_weak(current, index)) {
            break;
        }
    }
}

int SpatialMapManager::evictBySlidingWindow() {
    if (!work_mode_) return 0;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 获取当前位置所在的子图
    SubMapID cur_submap;
    {
        std::lock_guard<std::mutex> plock(position_mutex_);
        cur_submap = current_submap_;
    }
    
    // 计算起点所在的子图（如果已设置）
    SubMapID origin_submap{0, 0};
    if (has_origin_position_) {
        origin_submap = positionToSubMapID(origin_position_);
    }
    
    // 滑窗保留的最小索引（原子读取）
    int latest_idx = latest_keyframe_index_.load();
    int min_sliding_index = std::max(0, latest_idx - sliding_window_size_);
    
    int total_evicted = 0;
    std::vector<SubMapID> submaps_to_remove;
    
    // 遍历所有子图
    for (auto& pair : submaps_) {
        const SubMapID& sid = pair.first;
        auto& submap = pair.second;
        
        // 1. 空间保留：中心 + 周围8格（3×3范围）
        bool in_spatial_range = (std::abs(sid.x - cur_submap.x) <= 1 && 
                                 std::abs(sid.y - cur_submap.y) <= 1);
        
        // 2. 起点保留：起点 + 周围24格（5×5范围）
        bool in_origin_range = false;
        if (has_origin_position_) {
            in_origin_range = (std::abs(sid.x - origin_submap.x) <= 2 && 
                               std::abs(sid.y - origin_submap.y) <= 2);
        }
        
        // 2026-01-11: 修改淘汰逻辑
        // 热数据保留条件（满足任一即保留 Layer 3）：
        // 1. 在滑窗内（最近 400 帧）
        // 2. 在空间保留范围内（当前位置 3×3 子图）
        // 3. 在起点保留范围内（充电桩 5×5 子图）
        
        // 遍历子图内的所有关键帧
        auto all_kfs = submap->getAllKeyFrames();
        int evicted_in_submap = 0;
        bool has_hot_kf = false;  // 是否有热数据帧
        
        for (auto& kf : all_kfs) {
            if (!kf) continue;
            
            bool in_sliding_window = (kf->index >= min_sliding_index);
            
            // 热数据条件：在滑窗内 OR 在空间保留范围内 OR 在起点保留范围内
            bool should_keep_hot = in_sliding_window || in_spatial_range || in_origin_range;
            
            if (should_keep_hot) {
                // 保留完整数据（热数据）
                has_hot_kf = true;
            } else {
                // 不满足任何保留条件，清空 Layer 3 数据（变成冷数据）
                if (!kf->brief_descriptors.empty()) {
                    kf->point_3d.clear();
                    kf->point_3d.shrink_to_fit();
                    kf->point_2d_uv.clear();
                    kf->point_2d_uv.shrink_to_fit();
                    kf->point_2d_norm.clear();
                    kf->point_2d_norm.shrink_to_fit();
                    kf->point_id.clear();
                    kf->point_id.shrink_to_fit();
                    kf->brief_descriptors.clear();
                    kf->brief_descriptors.shrink_to_fit();
                    kf->keypoints.clear();
                    kf->keypoints.shrink_to_fit();
                    kf->keypoints_norm.clear();
                    kf->keypoints_norm.shrink_to_fit();
                    kf->window_keypoints.clear();
                    kf->window_keypoints.shrink_to_fit();
                    kf->window_brief_descriptors.clear();
                    kf->window_brief_descriptors.shrink_to_fit();
                    kf->image.release();
                    kf->thumbnail.release();
                    
                    evicted_in_submap++;
                }
            }
        }
        
        total_evicted += evicted_in_submap;
        
        // 子图移除条件：不在空间/起点保留范围内，且没有热数据帧
        if (!in_spatial_range && !in_origin_range && !has_hot_kf) {
            submaps_to_remove.push_back(sid);
        }
    }
    
    // 移除空子图
    for (const auto& sid : submaps_to_remove) {
        evicted_submaps_.insert(sid);
        submaps_.erase(sid);
    }
    
    if (total_evicted > 0 || !submaps_to_remove.empty()) {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        cache_stats_.evict_count += static_cast<int>(submaps_to_remove.size());
        
        static SimpleLogFilter evict_filter(10000);  // 10秒一次日志
        if (evict_filter.Output(GetNow_Steady())) {
            droslog(LogLevel::INFO, "SpatialMapManager: 滑窗淘汰 %d 帧Layer3, 移除 %d 个子图, "
                    "滑窗[%d,%d], 缓存 %d 个子图",
                    total_evicted, (int)submaps_to_remove.size(),
                    min_sliding_index, latest_idx,
                    (int)submaps_.size());
        }
    }
    
    return total_evicted;
}

// ========== 按索引查询关键帧 - 2026-01-11 ==========

void SpatialMapManager::registerKeyFrameIndex(std::shared_ptr<KeyFrame> kf) {
    if (!kf) return;
    
    std::lock_guard<std::mutex> lock(index_mutex_);
    index_to_keyframe_[kf->index] = kf;
}

void SpatialMapManager::unregisterKeyFrameIndex(int index) {
    std::lock_guard<std::mutex> lock(index_mutex_);
    index_to_keyframe_.erase(index);
}

bool SpatialMapManager::hasKeyFrame(int index) const {
    std::lock_guard<std::mutex> lock(index_mutex_);
    return index_to_keyframe_.find(index) != index_to_keyframe_.end();
}

bool SpatialMapManager::isHotData(int index) const {
    std::lock_guard<std::mutex> lock(index_mutex_);
    auto it = index_to_keyframe_.find(index);
    if (it == index_to_keyframe_.end() || !it->second) {
        return false;
    }
    // 热数据：描述子不为空
    return !it->second->brief_descriptors.empty();
}

std::shared_ptr<KeyFrame> SpatialMapManager::getKeyFrameByIndex(int index) {
    std::shared_ptr<KeyFrame> kf;
    
    {
        std::lock_guard<std::mutex> lock(index_mutex_);
        auto it = index_to_keyframe_.find(index);
        if (it == index_to_keyframe_.end() || !it->second) {
            return nullptr;
        }
        kf = it->second;
    }
    
    // 检查是否是冷数据
    if (kf->brief_descriptors.empty()) {
        // 冷数据，尝试从磁盘加载 Layer 3
        if (!loadKeyFrameLayer3FromDisk(kf)) {
            // 加载失败，返回 nullptr（调用者会尝试其他候选帧）
            static SimpleLogFilter load_fail_filter(5000);
            if (load_fail_filter.Output(GetNow_Steady())) {
                droslog(LogLevel::WARN, "SpatialMapManager::getKeyFrameByIndex() 冷数据加载失败: index=%d", index);
            }
            return nullptr;
        }
        
        static SimpleLogFilter load_ok_filter(5000);
        if (load_ok_filter.Output(GetNow_Steady())) {
            droslog(LogLevel::INFO, "SpatialMapManager::getKeyFrameByIndex() 冷数据加载成功: index=%d", index);
        }
    }
    
    return kf;
}

bool SpatialMapManager::loadKeyFrameLayer3FromDisk(std::shared_ptr<KeyFrame> kf) {
    if (!kf) return false;
    
    // 构建文件路径
    // 关键帧数据存储在子图文件中（.smap 二进制格式）
    SubMapID submap_id{kf->submap_x, kf->submap_y};
    
    std::string file_path = getSubMapFilePath(submap_id);
    if (file_path.empty()) {
        droslog(LogLevel::WARN, "SpatialMapManager::loadKeyFrameLayer3FromDisk() 子图文件不存在: (%d,%d)", 
                submap_id.x, submap_id.y);
        return false;
    }
    
    // 打开二进制子图文件
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) {
        droslog(LogLevel::WARN, "SpatialMapManager::loadKeyFrameLayer3FromDisk() 无法打开文件: %s", file_path.c_str());
        return false;
    }
    
    // 读取文件头
    SubMapFileHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(SubMapFileHeader));
    
    // 验证魔数
    if (header.magic[0] != 'S' || header.magic[1] != 'M' || 
        header.magic[2] != 'A' || header.magic[3] != 'P') {
        droslog(LogLevel::WARN, "SpatialMapManager::loadKeyFrameLayer3FromDisk() 文件格式错误: %s", file_path.c_str());
        in.close();
        return false;
    }
    
    bool found = false;
    
    // 遍历查找目标关键帧
    for (int i = 0; i < header.keyframe_count; i++) {
        // 读取关键帧头
        KeyFrameBlockHeader kf_header;
        in.read(reinterpret_cast<char*>(&kf_header), sizeof(KeyFrameBlockHeader));
        
        if (!in.good()) {
            break;
        }
        
        // 计算需要跳过的数据大小
        const size_t keypoint_data_size = kf_header.keypoint_count * 4 * sizeof(float);  // [x, y, norm_x, norm_y]
        const size_t desc_bits = static_cast<size_t>(kf_header.brief_size);
        const size_t bytes_per_desc = (desc_bits + 7) / 8;
        const size_t desc_data_size = kf_header.keypoint_count * bytes_per_desc;
        
        if (kf_header.kf_index == kf->index) {
            // 找到目标关键帧，加载数据
            
            // 读取关键点数据
            kf->keypoints.resize(kf_header.keypoint_count);
            kf->keypoints_norm.resize(kf_header.keypoint_count);
            
            for (int j = 0; j < kf_header.keypoint_count; j++) {
                float data[4];
                in.read(reinterpret_cast<char*>(data), sizeof(data));
                
                kf->keypoints[j].pt.x = data[0];
                kf->keypoints[j].pt.y = data[1];
                kf->keypoints_norm[j].pt.x = data[2];
                kf->keypoints_norm[j].pt.y = data[3];
            }
            
            // 读取 BRIEF 描述子
            kf->brief_descriptors.resize(kf_header.keypoint_count);
            std::vector<uint8_t> desc_bytes(bytes_per_desc);
            
            for (int j = 0; j < kf_header.keypoint_count; j++) {
                in.read(reinterpret_cast<char*>(desc_bytes.data()), bytes_per_desc);
                
                BRIEF::bitset desc(desc_bits);
                for (size_t bit = 0; bit < desc_bits; bit++) {
                    desc[bit] = (desc_bytes[bit / 8] >> (bit % 8)) & 1;
                }
                kf->brief_descriptors[j] = desc;
            }
            
            found = true;
            break;
        } else {
            // 跳过这个关键帧的数据
            in.seekg(keypoint_data_size + desc_data_size, std::ios::cur);
        }
    }
    
    in.close();
    return found;
}

