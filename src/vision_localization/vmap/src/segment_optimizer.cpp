/*******************************************************
 * SegmentOptimizer 类实现
 * 
 * 段优化器：建图过程中的增量位姿优化
 * 
 * 创建日期: 2025-12-15
 *******************************************************/

#include "segment_optimizer.h"
#include "spatial_map_manager.h"
#include "geo_utils/tf_helper.h"
#include "geo_utils/geo_utils.h"
#include "droslog/log.h"

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/slam3d/edge_se3.h"

using namespace utils;

typedef g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType> SlamLinearSolver;
typedef g2o::OptimizationAlgorithmLevenberg OptimizationAlgo;

SegmentOptimizer::SegmentOptimizer() {
    last_segment_end_pos_ = Eigen::Vector3d::Zero();
}

SegmentOptimizer::~SegmentOptimizer() {
    shutdown();
}

void SegmentOptimizer::shutdown() {
    if (disk_writer_) {
        disk_writer_->shutdown();
        disk_writer_.reset();
    }
}

void SegmentOptimizer::flushDiskSync() {
    if (disk_writer_) {
        disk_writer_->flushSync();
    }
}

DiskWriterStats SegmentOptimizer::getDiskWriterStats() const {
    if (disk_writer_) {
        return disk_writer_->getStats();
    }
    return DiskWriterStats();
}

void SegmentOptimizer::initDiskWriter() {
    if (!config_.auto_save_to_disk || config_.map_dir.empty()) {
        return;
    }
    
    if (!disk_writer_) {
        disk_writer_.reset(new DiskWriter());
    }
    
    // 配置磁盘写入器
    DiskWriterConfig dw_config;
    dw_config.batch_threshold = config_.disk_batch_threshold;
    dw_config.max_pending_time_sec = config_.disk_max_pending_sec;
    
    // 根据模式选择存储目录
    // - 建图模式 (use_temp_dir=true): submaps_temp/（未经全局优化的增量数据）
    // - 工作模式 (work_mode=true): submaps_work/（增量更新）
    // - 其他: submaps/（最终目录）
    std::string base_dir = config_.map_dir;
    // 移除末尾的斜杠，避免双斜杠问题
    while (!base_dir.empty() && base_dir.back() == '/') {
        base_dir.pop_back();
    }
    
    std::string submaps_dir;
    if (config_.work_mode) {
        submaps_dir = base_dir + "/spatial/submaps_work";
    } else if (config_.use_temp_dir) {
        submaps_dir = base_dir + "/spatial/submaps_temp";
    } else {
        submaps_dir = base_dir + "/spatial/submaps";
    }
    disk_writer_->initialize(submaps_dir, dw_config);
}

void SegmentOptimizer::setWorkMode(bool enabled) {
    config_.work_mode = enabled;
    if (enabled) {
        // 工作模式参数调整
        config_.disk_batch_threshold = 10;   // 累积更多才刷盘
        config_.disk_max_pending_sec = 60;   // 等待时间更长
        config_.use_temp_dir = false;        // 工作模式不用 temp 目录
    }
    
    // 重新初始化磁盘写入器
    if (disk_writer_) {
        disk_writer_->shutdown();
        disk_writer_.reset();
    }
    initDiskWriter();
}

int SegmentOptimizer::getCurrentOptimizationQuality() const {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    
    // 根据约束类型判断质量
    if (!rtk_frame_indices_.empty()) {
        return 3;  // QUALITY_VIO_RTK
    } else if (!loop_constraints_.empty()) {
        return 2;  // QUALITY_VIO_LOOP
    } else if (!reloc_constraints_.empty()) {
        return 1;  // QUALITY_VIO_RELOC
    } else {
        return 0;  // QUALITY_VIO_ONLY
    }
}

bool SegmentOptimizer::isVioOnlyTimeout() const {
    // 已在 pending_mutex_ 保护下调用
    if (pending_keyframes_.empty()) return false;
    
    // 检查是否为纯 VIO 段
    bool is_vio_only = rtk_frame_indices_.empty() && 
                       loop_constraints_.empty() && 
                       reloc_constraints_.empty();
    
    if (!is_vio_only) return false;
    
    // 检查时间
    double first_ts = pending_keyframes_.front()->time_stamp;
    double last_ts = pending_keyframes_.back()->time_stamp;
    
    return (last_ts - first_ts) > config_.vio_only_max_time;
}

bool SegmentOptimizer::shouldBacktrackOptimize() const {
    // 已在 pending_mutex_ 保护下调用
    if (!config_.work_mode) return false;
    
    // 检查是否有未优化的历史段
    bool has_unoptimized_history = false;
    for (const auto& seg : history_segments_) {
        if (!seg.optimized && seg.best_quality == 0) {  // 纯 VIO 段
            has_unoptimized_history = true;
            break;
        }
    }
    
    if (!has_unoptimized_history) return false;
    
    // 当前段有约束时才触发回溯
    return !rtk_frame_indices_.empty() || 
           !loop_constraints_.empty() || 
           !reloc_constraints_.empty();
}

void SegmentOptimizer::setConfig(const SegmentOptimizerConfig& config) {
    config_ = config;
    
    // 如果配置了自动存盘，初始化磁盘写入器
    if (config_.auto_save_to_disk && !config_.map_dir.empty()) {
        initDiskWriter();
    }
}

void SegmentOptimizer::setSpatialMapManager(SpatialMapManager* manager) {
    spatial_manager_ = manager;
}

bool SegmentOptimizer::addKeyFrame(std::shared_ptr<KeyFrame> kf) {
    if (!kf) return false;
    
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        
        // 记录关键帧
        pending_keyframes_.push_back(kf);
        
        // 检查是否是 RTK Fix 帧
        if (kf->ref_loc_info_.type == 1) {  // RTK_NARROW_INT
            rtk_frame_indices_.push_back(static_cast<int>(pending_keyframes_.size()) - 1);
            last_rtk_timestamp_ = kf->time_stamp;
            last_constraint_timestamp_ = kf->time_stamp;
        }
        
        // 检查是否有回环信息
        if (kf->has_loop && kf->loop_index >= 0) {
            LoopConstraint lc;
            lc.from_index = kf->index;
            lc.to_index = kf->loop_index;
            lc.relative_t = kf->getLoopRelativeT();
            lc.relative_q = kf->getLoopRelativeQ();
            lc.timestamp = kf->time_stamp;
            loop_constraints_.push_back(lc);
            last_constraint_timestamp_ = kf->time_stamp;
        }
    }
    
    // ========== 拆锁版：在同一个锁内完成检查和执行 - 2025-12-24 ==========
    // "谁加锁，谁负责锁的生命周期"
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        
        // 检查是否满足触发条件（不加锁版本）
        if (!shouldTriggerOptimizationUnlocked()) {
            return false;
        }
        
        // 工作模式特殊处理
        bool current_is_vio_only = rtk_frame_indices_.empty() && 
                                   loop_constraints_.empty() && 
                                   reloc_constraints_.empty();
        
        if (config_.work_mode && current_is_vio_only && !pending_keyframes_.empty()) {
            // 检查是否是超时强制触发（2分钟或帧数过多）
            double elapsed = pending_keyframes_.back()->time_stamp - pending_keyframes_.front()->time_stamp;
            bool is_timeout = elapsed > config_.max_time_gap || pending_keyframes_.size() >= 200;
            // 2026-01-11: 限制最多 2 段（2 × 200 帧 × 5KB = 2MB）
            bool is_history_overflow = history_segments_.size() >= 2;
            
            if (is_timeout || is_history_overflow) {
                // 超时或历史段过多，必须立即优化存盘，不能再延迟
                droslog(LogLevel::WARN, "SegmentOptimizer: 纯VIO段强制刷盘 (%.1fs, %d帧, %d历史段)",
                        elapsed, (int)pending_keyframes_.size(), (int)history_segments_.size());
                // 继续执行 doSegmentOptimization()
            } else {
                // 正常情况：纯 VIO 段暂存到历史，等待约束恢复
                // 2026-01-11: 保留完整帧，限制最多 2 段（2MB 内存）
                PendingSegment seg;
                seg.keyframes = pending_keyframes_;  // 保留完整帧，避免回溯时磁盘 I/O
                seg.rtk_indices = rtk_frame_indices_;
                seg.loops = loop_constraints_;
                seg.relocs = reloc_constraints_;
                seg.best_quality = 0;  // QUALITY_VIO_ONLY
                seg.first_timestamp = pending_keyframes_.front()->time_stamp;
                seg.last_timestamp = pending_keyframes_.back()->time_stamp;
                seg.optimized = false;
                seg.flushed = false;
                history_segments_.push_back(seg);
                
                // 清空当前段（等待约束恢复后回溯优化）
                pending_keyframes_.clear();
                rtk_frame_indices_.clear();
                loop_constraints_.clear();
                reloc_constraints_.clear();
                
                droslog(LogLevel::INFO, "SegmentOptimizer: 纯VIO段暂存历史 (共%d段，等待约束)",
                        (int)history_segments_.size());
                return false;
            }
        }
        
        // 执行段优化（调用者已持有锁）
        int optimized = doSegmentOptimization();
        return optimized > 0;
    }
}

void SegmentOptimizer::addLoopConstraint(int from_index, int to_index,
                                          const Eigen::Matrix<double, 8, 1>& loop_info) {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    
    LoopConstraint lc;
    lc.from_index = from_index;
    lc.to_index = to_index;
    lc.relative_t = Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
    lc.relative_q = Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
    
    // 获取时间戳
    for (const auto& kf : pending_keyframes_) {
        if (kf->index == from_index) {
            lc.timestamp = kf->time_stamp;
            break;
        }
    }
    
    loop_constraints_.push_back(lc);
    last_constraint_timestamp_ = lc.timestamp;
}

void SegmentOptimizer::addRelocConstraint(int frame_index, int match_index,
                                           const Eigen::Vector3d& reloc_pos,
                                           const Eigen::Quaterniond& reloc_quat) {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    
    RelocConstraint rc;
    rc.frame_index = frame_index;
    rc.match_index = match_index;
    rc.reloc_pos = reloc_pos;
    rc.reloc_quat = reloc_quat;
    
    // 获取时间戳
    for (const auto& kf : pending_keyframes_) {
        if (kf->index == frame_index) {
            rc.timestamp = kf->time_stamp;
            break;
        }
    }
    
    reloc_constraints_.push_back(rc);
    last_constraint_timestamp_ = rc.timestamp;
}

int SegmentOptimizer::forceOptimize() {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    
    if (pending_keyframes_.size() < 10) {
        droslog(LogLevel::WARN, "SegmentOptimizer::forceOptimize() 帧数不足: %d", 
                (int)pending_keyframes_.size());
        return 0;
    }
    
    // 即使没有 RTK 也尝试优化（使用充电桩约束或仅 VIO 约束）
    return doSegmentOptimization();
}

void SegmentOptimizer::reset() {
    // 先关闭旧的磁盘写入器
    if (disk_writer_) {
        disk_writer_->shutdown();
        disk_writer_.reset();
    }
    
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        
        pending_keyframes_.clear();
        rtk_frame_indices_.clear();
        loop_constraints_.clear();
        reloc_constraints_.clear();
        history_segments_.clear();  // 清理历史段
        last_segment_end_pos_ = Eigen::Vector3d::Zero();
        has_last_segment_ = false;
        last_rtk_timestamp_ = 0.0;
        last_constraint_timestamp_ = 0.0;
    }
    
    {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        stats_ = SegmentStats();
    }
}

int SegmentOptimizer::getPendingCount() const {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    return static_cast<int>(pending_keyframes_.size());
}

int SegmentOptimizer::getCurrentRtkCount() const {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    return static_cast<int>(rtk_frame_indices_.size());
}

int SegmentOptimizer::getCurrentLoopCount() const {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    return static_cast<int>(loop_constraints_.size());
}

int SegmentOptimizer::getCurrentRelocCount() const {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    return static_cast<int>(reloc_constraints_.size());
}

SegmentStats SegmentOptimizer::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

bool SegmentOptimizer::hasPendingSegment() const {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    return !pending_keyframes_.empty();
}

bool SegmentOptimizer::shouldTriggerOptimization() const {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    return shouldTriggerOptimizationUnlocked();
}

// 不加锁版本，调用方必须持有 pending_mutex_
bool SegmentOptimizer::shouldTriggerOptimizationUnlocked() const {
    // 条件1：帧数不足，不触发
    if (pending_keyframes_.size() < static_cast<size_t>(config_.min_keyframe_count)) {
        return false;
    }
    
    int current_idx = static_cast<int>(pending_keyframes_.size()) - 1;
    
    // 检查各类约束是否足够
    bool has_rtk_constraint = rtk_frame_indices_.size() >= static_cast<size_t>(config_.min_rtk_count);
    bool has_loop_constraint = loop_constraints_.size() >= static_cast<size_t>(config_.min_loop_count);
    bool has_reloc_constraint = reloc_constraints_.size() >= static_cast<size_t>(config_.min_reloc_count);
    bool has_any_constraint = has_rtk_constraint || has_loop_constraint || has_reloc_constraint;
    
    // ========== 优先级1：RTK 约束（最可靠）==========
    // RTK 数量足够 且 最后 3 帧内有 RTK
    if (has_rtk_constraint && !rtk_frame_indices_.empty()) {
        int last_rtk_idx = rtk_frame_indices_.back();
        if (current_idx - last_rtk_idx <= 3) {
            return true;  // RTK 条件满足，立即触发
        }
        // 有 RTK 但不在末尾，继续等待 RTK（不立即用次优约束）
        // 除非超时（见下方条件6）
    }
    
    // ========== 优先级2：没有 RTK 时，使用回环约束 ==========
    
    if (!has_rtk_constraint && has_loop_constraint) {
        return true;
    }
    
    // ========== 优先级3：没有 RTK 和回环时，使用重定位约束 ==========
    if (!has_rtk_constraint && !has_loop_constraint && has_reloc_constraint) {
        return true;
    }
    
    // ========== 条件5：行驶距离足够 + 有任意约束 ==========
    if (calculateSegmentDistance() >= config_.min_distance && has_any_constraint) {
        // 需要末尾有约束才触发
        if (has_rtk_constraint) {
            int last_rtk_idx = rtk_frame_indices_.back();
            if (current_idx - last_rtk_idx <= 3) {
                return true;
            }
        } else if (has_loop_constraint || has_reloc_constraint) {
            return true;
        }
    }
    
    // ========== 工作模式：检查回溯优化 ==========
    if (config_.work_mode && shouldBacktrackOptimize()) {
        return true;  // 有历史纯 VIO 段需要回溯优化
    }
    
    // ========== 条件6：超时强制触发 ==========
    // 工作模式：max_time_gap (默认60秒) 未优化就强制触发
    // 这是防止内存无限增长的关键保护机制
    if (!pending_keyframes_.empty()) {
        double current_ts = pending_keyframes_.back()->time_stamp;
        double first_ts = pending_keyframes_.front()->time_stamp;
        double elapsed = current_ts - first_ts;
        
        // 超时强制触发（无论有无约束）
        if (elapsed > config_.max_time_gap) {
            droslog(LogLevel::WARN, "SegmentOptimizer: 超时强制触发 (%.1fs > %.1fs), frames=%d, rtk=%d, reloc=%d",
                    elapsed, config_.max_time_gap, (int)pending_keyframes_.size(),
                    (int)rtk_frame_indices_.size(), (int)reloc_constraints_.size());
            return true;
        }
        
        // 帧数过多也强制触发（防止内存溢出）
        if (pending_keyframes_.size() >= 200) {
            droslog(LogLevel::WARN, "SegmentOptimizer: 帧数过多强制触发 (%d frames)", (int)pending_keyframes_.size());
            return true;
        }
    }
    
    // ========== 历史段数量限制 ==========
    // 如果历史段太多，也触发优化防止内存溢出
    // 2026-01-11: 限制最多 2 段（2 × 200 帧 × 5KB = 2MB）
    if (config_.work_mode && history_segments_.size() >= 2) {
        droslog(LogLevel::WARN, "SegmentOptimizer: 历史段过多强制触发 (%d segments)", (int)history_segments_.size());
        return true;
    }
    
    return false;
}

double SegmentOptimizer::calculateSegmentDistance() const {
    // 已在 pending_mutex_ 保护下调用
    if (pending_keyframes_.empty()) return 0.0;
    
    Eigen::Vector3d start_pos;
    if (has_last_segment_) {
        start_pos = last_segment_end_pos_;
    } else if (!pending_keyframes_.empty()) {
        start_pos = pending_keyframes_.front()->T_w_i;
    } else {
        return 0.0;
    }
    
    Eigen::Vector3d end_pos = pending_keyframes_.back()->T_w_i;
    
    // 计算 2D 距离（忽略 Z）
    double dx = end_pos.x() - start_pos.x();
    double dy = end_pos.y() - start_pos.y();
    return std::sqrt(dx * dx + dy * dy);
}

std::vector<TimedAlignNode> SegmentOptimizer::buildOptimizationNodes() const {
    // 已在 pending_mutex_ 保护下调用
    std::vector<TimedAlignNode> nodes;
    
    auto TF_v2g = TFHelper::Instance()->Vio2Gps_t();
    
    for (size_t i = 0; i < pending_keyframes_.size(); i++) {
        auto& kf = pending_keyframes_[i];
        
        TimedAlignNode node;
        node.id = kf->index;
        node.timestamp = kf->time_stamp;
        
        // 使用 VIO 位姿（需要坐标变换）
        Eigen::Quaterniond vio_q{kf->vio_R_w_i};
        auto pose = TFHelper::Instance()->TF_Vio2Gps(kf->vio_T_w_i, vio_q);
        node.pos = pose.pos + TF_v2g;
        node.quat = pose.quat;
        
        // RTK 或充电桩约束
        const auto& rli = kf->ref_loc_info_;
        if (rli.type == 0) {  // 充电桩
            auto sp = std::make_shared<NodeRefPose>();
            sp->ref_pos_valid = true;
            sp->ref_pos = rli.xyz;
            sp->ref_pos_cov = rli.cov;
            sp->ref_quat_valid = true;
            sp->ref_quat = Eigen::Quaterniond::Identity();
            sp->ref_quat_cov = Eigen::Matrix3d::Identity() * 0.0001;
            node.ref_pose = sp;
        } else if (rli.type == 1) {  // RTK Fix
            auto sp = std::make_shared<NodeRefPose>();
            sp->ref_pos_valid = true;
            sp->ref_pos = rli.xyz;
            sp->ref_pos_cov = rli.cov;
            node.ref_pose = sp;
        }
        
        nodes.push_back(node);
    }
    
    return nodes;
}

int SegmentOptimizer::doSegmentOptimization() {
    // 注意：调用者需持有 pending_mutex_
    
    if (pending_keyframes_.size() < 10) {
        return 0;
    }
    
    // 计算当前段的优化质量
    int current_quality = 0;  // QUALITY_VIO_ONLY
    if (!rtk_frame_indices_.empty()) {
        current_quality = 3;  // QUALITY_VIO_RTK
    } else if (!loop_constraints_.empty()) {
        current_quality = 2;  // QUALITY_VIO_LOOP
    } else if (!reloc_constraints_.empty()) {
        current_quality = 1;  // QUALITY_VIO_RELOC
    }
    
    droslog(LogLevel::INFO, "SegmentOptimizer::doSegmentOptimization() 开始段优化: frames=%d, rtk=%d, loop=%d, reloc=%d, quality=%d",
            (int)pending_keyframes_.size(), (int)rtk_frame_indices_.size(),
            (int)loop_constraints_.size(), (int)reloc_constraints_.size(), current_quality);
    
    // 工作模式：检查是否需要回溯优化
    if (config_.work_mode && current_quality > 0 && !history_segments_.empty()) {
        // 有约束时，执行回溯优化
        int backtrack_count = doBacktrackOptimization();
        if (backtrack_count > 0) {
            droslog(LogLevel::INFO, "SegmentOptimizer::doSegmentOptimization() 回溯优化完成: %d frames", backtrack_count);
        }
    }
    
    // 构建优化节点
    auto nodes = buildOptimizationNodes();
    
    if (nodes.size() < 10) {
        droslog(LogLevel::WARN, "SegmentOptimizer::doSegmentOptimization() 节点数不足: %d", (int)nodes.size());
        return 0;
    }
    
    // 调用 spa_align 执行优化
    auto aligned_poses = spa_align(nodes);
    
    if (aligned_poses.size() != nodes.size()) {
        droslog(LogLevel::ERROR, "SegmentOptimizer::doSegmentOptimization() 优化失败: aligned=%d, nodes=%d",
                (int)aligned_poses.size(), (int)nodes.size());
        return 0;
    }
    
    // 应用优化结果并标记质量
    applyOptimizationResult(aligned_poses);
    
    // 标记每个关键帧的优化质量
    for (auto& kf : pending_keyframes_) {
        kf->optimization_quality = current_quality;
    }
    
    // 更新空间索引
    updateSpatialIndex();
    
    // 增量存盘（带质量控制）
    if (config_.auto_save_to_disk && !config_.map_dir.empty()) {
        saveTosDisk(current_quality);
    }
    
    // 更新统计信息
    {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        stats_.segment_count++;
        stats_.total_keyframes += static_cast<int>(pending_keyframes_.size());
        stats_.total_rtk_frames += static_cast<int>(rtk_frame_indices_.size());
        stats_.total_distance += calculateSegmentDistance();
    }
    
    // 记录段末位置
    if (!pending_keyframes_.empty()) {
        last_segment_end_pos_ = pending_keyframes_.back()->T_w_i;
        has_last_segment_ = true;
    }
    
    int optimized_count = static_cast<int>(pending_keyframes_.size());
    
    // 清空待处理列表
    pending_keyframes_.clear();
    rtk_frame_indices_.clear();
    loop_constraints_.clear();
    reloc_constraints_.clear();
    
    // 清空历史段（已经回溯优化过了）
    history_segments_.clear();
    
    // 计算释放的内存估计（每个 KeyFrame 约 5KB）
    int freed_memory_kb = optimized_count * 5;
    droslog(LogLevel::INFO, "SegmentOptimizer: 段优化完成 optimized=%d, quality=%d, 释放待处理队列约 %dKB", 
            optimized_count, current_quality, freed_memory_kb);
    
    return optimized_count;
}

int SegmentOptimizer::doBacktrackOptimization() {
    // 已在 pending_mutex_ 保护下调用
    // 将历史纯 VIO 段与当前有约束段合并优化
    
    if (history_segments_.empty()) {
        return 0;
    }
    
    // 2026-01-11: 直接使用保留的完整帧，无需磁盘 I/O
    // 收集所有未优化的历史段关键帧
    std::vector<std::shared_ptr<KeyFrame>> all_frames;
    for (auto& seg : history_segments_) {
        if (!seg.optimized) {
            all_frames.insert(all_frames.end(), seg.keyframes.begin(), seg.keyframes.end());
        }
    }
    
    if (all_frames.empty()) {
        return 0;
    }
    
    droslog(LogLevel::INFO, "SegmentOptimizer::doBacktrackOptimization() 回溯 %d 个历史帧", 
            (int)all_frames.size());
    
    // 将历史帧添加到当前 pending 列表的开头
    std::vector<std::shared_ptr<KeyFrame>> merged_frames;
    merged_frames.reserve(all_frames.size() + pending_keyframes_.size());
    merged_frames.insert(merged_frames.end(), all_frames.begin(), all_frames.end());
    merged_frames.insert(merged_frames.end(), pending_keyframes_.begin(), pending_keyframes_.end());
    
    // 替换 pending_keyframes_
    pending_keyframes_ = std::move(merged_frames);
    
    // 合并 RTK 索引（需要重新计算索引）
    std::vector<int> new_rtk_indices;
    for (size_t i = 0; i < pending_keyframes_.size(); i++) {
        auto& kf = pending_keyframes_[i];
        if (kf->ref_loc_info_.type == 1) {  // RTK_NARROW_INT
            new_rtk_indices.push_back(static_cast<int>(i));
        }
    }
    rtk_frame_indices_ = std::move(new_rtk_indices);
    
    // 标记历史段为已优化
    for (auto& seg : history_segments_) {
        seg.optimized = true;
    }
    
    return static_cast<int>(all_frames.size());
}

void SegmentOptimizer::applyOptimizationResult(const std::vector<TimedPose>& aligned_poses) {
    // 已在 pending_mutex_ 保护下调用
    
    // auto TF_v2g = TFHelper::Instance()->Vio2Gps_t(); // 2025-12-15: 新版本建图不需要转换到 Gps 坐标系
    
    for (size_t i = 0; i < pending_keyframes_.size() && i < aligned_poses.size(); i++) {
        auto& kf = pending_keyframes_[i];
        const auto& aligned = aligned_poses[i];
        
        // 将优化后位姿转换回 VIO 坐标系
        auto aligned_vio = TFHelper::Instance()->TF_Gps2Vio(aligned.pos, aligned.quat);
        
        // 更新关键帧位姿
        kf->updatePose(aligned_vio.pos, aligned_vio.quat.toRotationMatrix());
        
        // 标记为已段优化
        kf->is_segment_optimized = true;
    }
}

void SegmentOptimizer::updateSpatialIndex() {
    if (!spatial_manager_) return;
    
    // 标记所有优化过的帧为脏
    std::vector<std::shared_ptr<KeyFrame>> to_mark;
    {
        // 已在 pending_mutex_ 保护下
        to_mark = pending_keyframes_;
    }
    
    spatial_manager_->markDirtyBatch(to_mark);
    
    // 执行增量索引重建
    int moved = spatial_manager_->rebuildDirtyIndices();
    
    {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        stats_.total_moved_indices += moved;
    }
    
    droslog(LogLevel::INFO, "SegmentOptimizer::updateSpatialIndex() 索引更新: marked=%d, moved=%d",
            (int)to_mark.size(), moved);
}

void SegmentOptimizer::saveTosDisk(int quality) {
    if (!spatial_manager_) return;
    
    // 获取所有脏 SubMap
    auto dirty_submaps = spatial_manager_->getDirtySubMaps();
    
    if (dirty_submaps.empty()) {
        return;
    }
    
    // 工作模式下的质量控制策略：
    // - 新增区域（原始目录不存在）：无论质量如何都写入（有总比没有好）
    // - 覆盖区域（原始目录已存在）：只有高质量（quality > 0，有约束）才覆盖
    std::vector<std::shared_ptr<SubMap>> submaps_to_save;
    int skipped_low_quality = 0;
    
    if (config_.work_mode && quality == 0) {
        // 纯 VIO 段，只保存新增区域
        for (auto& submap : dirty_submaps) {
            if (!spatial_manager_->subMapExistsInOriginal(submap->id())) {
                // 新增区域，保存
                submaps_to_save.push_back(submap);
            } else {
                // 覆盖区域，跳过（保留原始高质量数据）
                skipped_low_quality++;
            }
        }
        
        if (skipped_low_quality > 0) {
            droslog(LogLevel::INFO, "SegmentOptimizer: 纯VIO段跳过 %d 个已有子图的覆盖（保留原始数据）",
                    skipped_low_quality);
        }
    } else {
        // 有约束的段，全部保存
        submaps_to_save = dirty_submaps;
    }
    
    if (submaps_to_save.empty()) {
        droslog(LogLevel::INFO, "SegmentOptimizer: 无需存盘（全部跳过）");
        return;
    }
    
    if (disk_writer_ && disk_writer_->isRunning()) {
        // 使用异步写入器
        disk_writer_->queueSubMaps(submaps_to_save);
        
        droslog(LogLevel::INFO, "SegmentOptimizer: 异步存盘 queued=%d, pending=%d, quality=%d (内存将在写入后释放)",
                (int)submaps_to_save.size(), disk_writer_->getPendingCount(), quality);
    } else {
        // 降级到同步写入
        int saved = spatial_manager_->saveDirtySubMaps(config_.map_dir);
        
        droslog(LogLevel::INFO, "SegmentOptimizer::saveTosDisk() 同步存盘: saved=%d submaps, quality=%d", 
                saved, quality);
    }
}


