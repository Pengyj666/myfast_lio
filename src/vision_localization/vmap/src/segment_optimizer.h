/*******************************************************
 * SegmentOptimizer 类定义
 * 
 * 段优化器：在建图/工作过程中，执行局部 SPA 优化
 * 
 * 约束优先级（统一建图与工作模式）：
 * 1. RTK + VIO（最可靠）
 * 2. 回环 + VIO（需要几何验证通过）
 * 3. 重定位 + VIO（依赖地图质量）
 * 4. 仅 VIO（最后手段，只做相对优化）
 * 
 * 工作流程：
 * 1. 收集关键帧，记录各类约束
 * 2. 当满足触发条件时，对该段执行优化
 * 3. 更新关键帧位姿，标记为脏帧
 * 4. 触发增量索引重建和异步存盘
 * 
 * 2025-12-15: 新增异步批量刷盘，减少 IO 阻塞
 * 2025-12-22: 支持回环/重定位约束，统一建图与工作模式
 * 
 * 创建日期: 2025-12-15
 *******************************************************/

#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <string>

#include <eigen3/Eigen/Dense>

#include "keyframe.h"
#include "spa_align.h"
#include "disk_writer.h"

// 前向声明
class SpatialMapManager;

// 回环约束信息
struct LoopConstraint {
    int from_index = -1;              // 当前帧索引
    int to_index = -1;                // 回环目标帧索引
    Eigen::Vector3d relative_t;       // 相对平移
    Eigen::Quaterniond relative_q;    // 相对旋转
    double timestamp = 0.0;
};

// 重定位约束信息
struct RelocConstraint {
    int frame_index = -1;             // 关键帧索引
    int match_index = -1;             // 匹配的历史关键帧索引
    Eigen::Vector3d reloc_pos;        // 重定位位置
    Eigen::Quaterniond reloc_quat;    // 重定位姿态
    double timestamp = 0.0;
};

// 段优化配置参数
struct SegmentOptimizerConfig {
    // 触发条件（满足帧数前提下，任一约束条件即可）
    int min_keyframe_count = 30;     // 最少关键帧数
    double min_distance = 5.0;       // 最小行驶距离（米）
    double max_time_gap = 60.0;      // 最大时间间隔（秒），超过则强制触发
    
    // 约束数量要求（满足任一即可触发）
    int min_rtk_count = 2;           // 最少 RTK Fix 帧数
    int min_loop_count = 1;          // 最少回环约束数
    int min_reloc_count = 3;         // 最少重定位成功数
    
    // 优化参数
    double vio_weight = 1.0;         // VIO 相对约束权重
    double rtk_pos_weight = 0.1;     // RTK 位置约束权重
    double rtk_quat_weight = 0.16;   // RTK 姿态约束权重（仅充电桩）
    double loop_weight = 0.5;        // 回环约束权重
    double reloc_weight = 0.3;       // 重定位约束权重
    
    // 存盘参数
    bool auto_save_to_disk = true;   // 优化后自动存盘
    std::string map_dir = "";        // 地图存储目录
    bool use_temp_dir = true;        // 使用临时目录（submaps_temp/）存储增量数据
    
    // 异步刷盘参数（建图模式默认值）
    int disk_batch_threshold = 5;    // 累积多少个 SubMap 才触发写入
    int disk_max_pending_sec = 30;   // 最长等待时间（秒）
    
    // 工作模式特有参数
    bool work_mode = false;          // 是否工作模式
    double vio_only_max_time = 120.0; // 纯 VIO 段最大保留时间（秒），超时强制刷盘
};

// 段信息统计
struct SegmentStats {
    int segment_count = 0;           // 已优化段数
    int total_keyframes = 0;         // 总优化帧数
    int total_rtk_frames = 0;        // 总 RTK 帧数
    int total_moved_indices = 0;     // 总移动索引数
    double total_distance = 0.0;     // 总优化距离
};

class SegmentOptimizer {
public:
    SegmentOptimizer();
    ~SegmentOptimizer();
    
    // ========== 初始化 ==========
    
    // 设置配置参数
    void setConfig(const SegmentOptimizerConfig& config);
    
    // 设置空间索引管理器（用于增量更新）
    void setSpatialMapManager(SpatialMapManager* manager);
    
    // ========== 关键帧输入 ==========
    
    // 添加关键帧（建图/工作时调用）
    // 返回值：true 表示触发了段优化
    bool addKeyFrame(std::shared_ptr<KeyFrame> kf);
    
    // 添加回环约束（回环检测成功时调用）
    void addLoopConstraint(int from_index, int to_index, 
                           const Eigen::Matrix<double, 8, 1>& loop_info);
    
    // 添加重定位约束（重定位成功时调用）
    void addRelocConstraint(int frame_index, int match_index,
                            const Eigen::Vector3d& reloc_pos,
                            const Eigen::Quaterniond& reloc_quat);
    
    // 强制执行当前段的优化（保存地图前调用）
    // 返回值：优化的帧数
    int forceOptimize();
    
    // 重置状态（开始新建图时调用）
    void reset();
    
    // 关闭（保存地图后调用，确保所有数据落盘）
    void shutdown();
    
    // 强制同步刷盘（保存地图前调用，确保所有数据写入磁盘）
    void flushDiskSync();
    
    // 获取磁盘写入器统计信息
    DiskWriterStats getDiskWriterStats() const;
    
    // ========== 状态查询 ==========
    
    // 获取当前待优化帧数
    int getPendingCount() const;
    
    // 获取当前段的 RTK 帧数
    int getCurrentRtkCount() const;
    
    // 获取当前段的回环约束数
    int getCurrentLoopCount() const;
    
    // 获取当前段的重定位成功数
    int getCurrentRelocCount() const;
    
    // 获取统计信息
    SegmentStats getStats() const;
    
    // 是否有待处理的段
    bool hasPendingSegment() const;
    
    // ========== 工作模式特有方法 - 2025-12-23 ==========
    
    // 设置工作模式（调整刷盘参数）
    void setWorkMode(bool enabled);
    
    // 获取当前段的优化质量
    int getCurrentOptimizationQuality() const;

private:
    // ========== 拆锁版触发检查（修复死锁）- 2025-12-24 ==========
    // 不加锁版本，调用方必须持有 pending_mutex_
    bool shouldTriggerOptimizationUnlocked() const;
    // 待优化段信息（用于回溯优化）
    // 2026-01-11: 保留完整关键帧，但限制最多 2 段，避免内存爆炸
    // 2 段 × 200 帧 × 5KB = 2MB，可接受
    struct PendingSegment {
        std::vector<std::shared_ptr<KeyFrame>> keyframes;  // 保留完整帧，避免回溯时磁盘 I/O
        std::vector<int> rtk_indices;       // RTK 帧在 keyframes 中的位置
        std::vector<LoopConstraint> loops;
        std::vector<RelocConstraint> relocs;
        int best_quality = -1;  // OptimizationQuality
        double first_timestamp = 0.0;
        double last_timestamp = 0.0;
        bool optimized = false;
        bool flushed = false;
    };
    
    // 保留的历史段（用于回溯优化）
    std::vector<PendingSegment> history_segments_;
    
    // 检查是否需要回溯优化
    bool shouldBacktrackOptimize() const;
    
    // 执行回溯优化（将历史纯 VIO 段与当前有约束段一起优化）
    int doBacktrackOptimization();
    
    // 检查纯 VIO 段是否超时
    bool isVioOnlyTimeout() const;
    // 检查是否满足触发条件
    bool shouldTriggerOptimization() const;
    
    // 执行段优化
    // 返回值：优化的帧数，0 表示失败
    int doSegmentOptimization();
    
    // 构建优化节点
    std::vector<TimedAlignNode> buildOptimizationNodes() const;
    
    // 应用优化结果
    void applyOptimizationResult(const std::vector<TimedPose>& aligned_poses);
    
    // 更新空间索引
    void updateSpatialIndex();
    
    // 增量存盘（带质量控制）
    // quality: 当前段的优化质量
    // 工作模式下：
    //   - 新增区域（原始目录不存在）：无论质量如何都写入
    //   - 覆盖区域（原始目录已存在）：只有高质量（有RTK/回环/重定位约束）才覆盖
    void saveTosDisk(int quality);
    
    // 计算段行驶距离
    double calculateSegmentDistance() const;
    
    // 配置
    SegmentOptimizerConfig config_;
    
    // 空间索引管理器
    SpatialMapManager* spatial_manager_ = nullptr;
    
    // 当前段的关键帧列表
    std::vector<std::shared_ptr<KeyFrame>> pending_keyframes_;
    mutable std::mutex pending_mutex_;
    
    // 当前段的 RTK 帧索引
    std::vector<int> rtk_frame_indices_;
    
    // 当前段的回环约束
    std::vector<LoopConstraint> loop_constraints_;
    
    // 当前段的重定位约束
    std::vector<RelocConstraint> reloc_constraints_;
    
    // 上一段最后一帧的位置（用于计算距离）
    Eigen::Vector3d last_segment_end_pos_;
    bool has_last_segment_ = false;
    
    // 上一个 RTK 帧的时间戳
    double last_rtk_timestamp_ = 0.0;
    
    // 上一个约束（任意类型）的时间戳
    double last_constraint_timestamp_ = 0.0;
    
    // 统计信息
    SegmentStats stats_;
    mutable std::mutex stats_mutex_;
    
    // 异步磁盘写入器
    std::unique_ptr<DiskWriter> disk_writer_;
    
    // 初始化磁盘写入器
    void initDiskWriter();
};


