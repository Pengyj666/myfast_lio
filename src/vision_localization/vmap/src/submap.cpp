/*******************************************************
 * SubMap 类实现
 * 
 * 创建日期: 2025-12-10
 *******************************************************/

#include "submap.h"
#include "keyframe.h"

SubMap::SubMap(const SubMapID& id) : id_(id) {
}

SubMap::~SubMap() {
}

std::shared_ptr<Cell> SubMap::getOrCreateCell(const CellID& cell_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cells_.find(cell_id);
    if (it != cells_.end()) {
        return it->second;
    }
    
    // 创建新 Cell
    auto cell = std::make_shared<Cell>(cell_id);
    cells_[cell_id] = cell;
    return cell;
}

std::shared_ptr<Cell> SubMap::getCell(const CellID& cell_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cells_.find(cell_id);
    if (it != cells_.end()) {
        return it->second;
    }
    return nullptr;
}

bool SubMap::tryInsertKeyFrame(std::shared_ptr<KeyFrame> kf,
                                std::shared_ptr<KeyFrame>* replaced_kf) {
    if (!kf) return false;
    
    // 根据关键帧的 cell 坐标获取或创建对应的 Cell
    CellID cell_id{kf->cell_x, kf->cell_y};
    auto cell = getOrCreateCell(cell_id);
    
    // 尝试插入到 Cell 中
    bool inserted = cell->tryInsertKeyFrame(kf, replaced_kf);
    if (inserted) {
        is_dirty_ = true;  // 标记为脏
    }
    return inserted;
}

bool SubMap::forceInsertKeyFrame(std::shared_ptr<KeyFrame> kf) {
    if (!kf) return false;
    
    CellID cell_id{kf->cell_x, kf->cell_y};
    auto cell = getOrCreateCell(cell_id);
    
    bool inserted = cell->forceInsertKeyFrame(kf);
    if (inserted) {
        is_dirty_ = true;
    }
    return inserted;
}

bool SubMap::removeKeyFrame(std::shared_ptr<KeyFrame> kf) {
    if (!kf) return false;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 根据关键帧记录的 cell 坐标查找
    CellID cell_id{kf->cell_x, kf->cell_y};
    auto it = cells_.find(cell_id);
    if (it != cells_.end()) {
        bool removed = it->second->removeKeyFrame(kf);
        if (removed) {
            is_dirty_ = true;
            return true;
        }
    }
    
    // 如果按记录的坐标没找到，遍历所有 Cell 查找（兼容索引不一致情况）
    for (auto& pair : cells_) {
        if (pair.second->removeKeyFrame(kf)) {
            is_dirty_ = true;
            return true;
        }
    }
    
    return false;
}

std::vector<std::shared_ptr<KeyFrame>> SubMap::getAllKeyFrames() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::shared_ptr<KeyFrame>> result;
    
    for (const auto& pair : cells_) {
        auto kfs = pair.second->getAllKeyFrames();
        result.insert(result.end(), kfs.begin(), kfs.end());
    }
    
    return result;
}

std::vector<std::shared_ptr<KeyFrame>> SubMap::getKeyFramesByDirection(int direction_slot) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::shared_ptr<KeyFrame>> result;
    
    for (const auto& pair : cells_) {
        auto kf = pair.second->getKeyFrame(direction_slot);
        if (kf) {
            result.push_back(kf);
        }
    }
    
    return result;
}

int SubMap::getCellCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(cells_.size());
}

int SubMap::getKeyFrameCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    int count = 0;
    for (const auto& pair : cells_) {
        count += pair.second->getKeyFrameCount();
    }
    return count;
}

void SubMap::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cells_.clear();
}

