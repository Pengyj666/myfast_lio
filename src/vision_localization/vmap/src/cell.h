/*******************************************************
 * Cell 类定义
 * 
 * 空间划分的最小单元（0.25×0.25m），每个 Cell 包含 6 个方向槽位
 * 每个方向槽位保留第一帧进入该槽位的关键帧（先入先占策略）
 * 
 * 创建日期: 2025-12-10
 *******************************************************/

#pragma once

#include <array>
#include <vector>
#include <memory>

#include "spatial_index.h"

// 前向声明，避免循环引用
class KeyFrame;

class Cell {
public:
    static constexpr int NUM_DIRECTIONS = 6;      // 方向槽位数量（每60度一个）
    static constexpr double CELL_SIZE = 0.25;     // Cell 边长（米）
    
    Cell(const CellID& id);
    ~Cell();
    
    // 尝试插入关键帧（先入先占策略：只有槽位为空或冷数据才插入）
    // 返回值：true 表示插入成功，false 表示槽位已被有效帧占用
    // replaced_kf: 输出参数，如果替换了冷数据帧，返回被替换的帧
    bool tryInsertKeyFrame(std::shared_ptr<KeyFrame> kf, 
                           std::shared_ptr<KeyFrame>* replaced_kf = nullptr);
    
    // 强制插入关键帧（替换已有帧，用于索引更新）
    // 返回值：true 成功，false 失败
    bool forceInsertKeyFrame(std::shared_ptr<KeyFrame> kf);
    
    // 移除指定关键帧（通过指针匹配）
    // 返回值：true 成功移除，false 未找到
    bool removeKeyFrame(std::shared_ptr<KeyFrame> kf);
    
    // 移除指定方向槽位的关键帧
    // 返回值：移除的关键帧（可能为空）
    std::shared_ptr<KeyFrame> removeKeyFrame(int direction_slot);
    
    // 获取指定方向的关键帧（可能为空）
    std::shared_ptr<KeyFrame> getKeyFrame(int direction_slot) const;
    
    // 获取所有非空关键帧
    std::vector<std::shared_ptr<KeyFrame>> getAllKeyFrames() const;
    
    // 获取关键帧数量
    int getKeyFrameCount() const;
    
    // 判断是否为空
    bool isEmpty() const;
    
    // 清空所有关键帧
    void clear();
    
    const CellID& id() const { return id_; }

private:
    CellID id_;
    std::array<std::shared_ptr<KeyFrame>, NUM_DIRECTIONS> direction_slots_;
};

