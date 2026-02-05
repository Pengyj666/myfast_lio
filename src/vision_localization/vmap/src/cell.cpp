/*******************************************************
 * Cell 类实现
 * 
 * 创建日期: 2025-12-10
 * 策略：先入先占 - 每个方向槽位保留第一帧进入的关键帧
 *******************************************************/

#include "cell.h"
#include "keyframe.h"

Cell::Cell(const CellID& id) : id_(id) {
    // 初始化所有槽位为空
    direction_slots_.fill(nullptr);
}

Cell::~Cell() {
    // shared_ptr 会自动管理内存
}

bool Cell::tryInsertKeyFrame(std::shared_ptr<KeyFrame> kf, 
                              std::shared_ptr<KeyFrame>* replaced_kf) {
    if (!kf) return false;
    
    int slot = kf->direction_slot;
    if (slot < 0 || slot >= NUM_DIRECTIONS) return false;
    
    auto& existing = direction_slots_[slot];
    
    if (!existing) {
        // 槽位为空，直接插入（先入先占）
        direction_slots_[slot] = kf;
        if (replaced_kf) *replaced_kf = nullptr;
        return true;
    }
    
    // 2026-01-11: 检查现有帧是否是冷数据（Layer 3 已清空）
    // 如果是冷数据，允许新帧替换，实现"季节更新"功能
    if (existing->brief_descriptors.empty()) {
        // 旧帧数据已清空，允许新帧替换
        if (replaced_kf) *replaced_kf = existing;  // 返回被替换的帧
        direction_slots_[slot] = kf;
        return true;
    }
    
    // 槽位被有效帧占用，拒绝新帧
    if (replaced_kf) *replaced_kf = nullptr;
    return false;
}

bool Cell::forceInsertKeyFrame(std::shared_ptr<KeyFrame> kf) {
    if (!kf) return false;
    
    int slot = kf->direction_slot;
    if (slot < 0 || slot >= NUM_DIRECTIONS) return false;
    
    // 强制替换（用于索引更新场景）
    direction_slots_[slot] = kf;
    return true;
}

bool Cell::removeKeyFrame(std::shared_ptr<KeyFrame> kf) {
    if (!kf) return false;
    
    for (int i = 0; i < NUM_DIRECTIONS; i++) {
        if (direction_slots_[i] == kf) {
            direction_slots_[i] = nullptr;
            return true;
        }
    }
    return false;
}

std::shared_ptr<KeyFrame> Cell::removeKeyFrame(int direction_slot) {
    if (direction_slot < 0 || direction_slot >= NUM_DIRECTIONS) {
        return nullptr;
    }
    
    auto kf = direction_slots_[direction_slot];
    direction_slots_[direction_slot] = nullptr;
    return kf;
}

std::shared_ptr<KeyFrame> Cell::getKeyFrame(int direction_slot) const {
    if (direction_slot < 0 || direction_slot >= NUM_DIRECTIONS) {
        return nullptr;
    }
    return direction_slots_[direction_slot];
}

std::vector<std::shared_ptr<KeyFrame>> Cell::getAllKeyFrames() const {
    std::vector<std::shared_ptr<KeyFrame>> result;
    result.reserve(NUM_DIRECTIONS);
    
    for (const auto& kf : direction_slots_) {
        if (kf) {
            result.push_back(kf);
        }
    }
    return result;
}

int Cell::getKeyFrameCount() const {
    int count = 0;
    for (const auto& kf : direction_slots_) {
        if (kf) count++;
    }
    return count;
}

bool Cell::isEmpty() const {
    for (const auto& kf : direction_slots_) {
        if (kf) return false;
    }
    return true;
}

void Cell::clear() {
    for (auto& kf : direction_slots_) {
        kf.reset();
    }
}

