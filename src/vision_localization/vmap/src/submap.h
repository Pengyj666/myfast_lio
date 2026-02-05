/*******************************************************
 * SubMap 类定义
 * 
 * 子图（5×5m 区域），动态加载的基本单元
 * 每个子图包含 20×20 个 Cell（稀疏存储，Cell大小为0.25m）
 * 
 * 创建日期: 2025-12-10
 *******************************************************/

#pragma once

#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>

#include "spatial_index.h"
#include "cell.h"

class SubMap {
public:
    static constexpr double SUBMAP_SIZE = 5.0;    // 子图边长（米）
    static constexpr int CELLS_PER_SIDE = 20;     // 每边 Cell 数量（5.0 / 0.25 = 20）
    
    SubMap(const SubMapID& id);
    ~SubMap();
    
    // 获取或创建 Cell（如果不存在则创建）
    std::shared_ptr<Cell> getOrCreateCell(const CellID& cell_id);
    
    // 获取 Cell（不存在返回 nullptr）
    std::shared_ptr<Cell> getCell(const CellID& cell_id) const;
    
    // 尝试插入关键帧（会自动找到对应的 Cell）
    // replaced_kf: 输出参数，如果替换了冷数据帧，返回被替换的帧
    bool tryInsertKeyFrame(std::shared_ptr<KeyFrame> kf,
                           std::shared_ptr<KeyFrame>* replaced_kf = nullptr);
    
    // 强制插入关键帧（用于索引更新，会替换已有帧）
    bool forceInsertKeyFrame(std::shared_ptr<KeyFrame> kf);
    
    // 移除关键帧（通过指针匹配）
    bool removeKeyFrame(std::shared_ptr<KeyFrame> kf);
    
    // 获取所有关键帧
    std::vector<std::shared_ptr<KeyFrame>> getAllKeyFrames() const;
    
    // 获取指定方向的所有关键帧（用于重定位时方向过滤）
    std::vector<std::shared_ptr<KeyFrame>> getKeyFramesByDirection(int direction_slot) const;
    
    // 统计信息
    int getCellCount() const;
    int getKeyFrameCount() const;
    
    // 清空所有 Cell
    void clear();
    
    const SubMapID& id() const { return id_; }
    
    // ========== 增量更新支持 - 2025-12-15 ==========
    
    // 标记为脏（需要刷盘）
    void markDirty() { is_dirty_ = true; }
    
    // 清除脏标记
    void clearDirty() { is_dirty_ = false; }
    
    // 检查是否为脏
    bool isDirty() const { return is_dirty_; }

private:
    SubMapID id_;
    std::unordered_map<CellID, std::shared_ptr<Cell>> cells_;
    mutable std::mutex mutex_;  // 线程安全保护
    bool is_dirty_ = false;     // 是否有变更需要刷盘
};

