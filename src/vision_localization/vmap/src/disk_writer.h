/*******************************************************
 * DiskWriter 类定义
 * 
 * 异步磁盘写入器：将 SubMap 序列化操作放到独立线程执行，
 * 避免阻塞主建图流程
 * 
 * 特性：
 * 1. 异步写入：独立线程执行 IO 操作
 * 2. 批量延迟：累积多个 SubMap 后批量写入
 * 3. 去重合并：同一 SubMap 多次修改只写最新版本
 * 4. 优雅关闭：确保所有待写数据落盘
 * 
 * 创建日期: 2025-12-15
 *******************************************************/

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <string>
#include <chrono>

#include "spatial_index.h"
#include "submap.h"

// 写入器配置
struct DiskWriterConfig {
    int batch_threshold = 5;              // 累积多少个 SubMap 才触发写入
    int max_pending_time_sec = 30;        // 最长等待时间（秒），超过后强制写入
    int writer_thread_priority = 0;       // 写入线程优先级（0=默认）
    bool sync_on_write = false;           // 是否每次写入后 sync
    
    // 工作模式默认配置
    static DiskWriterConfig workModeDefault() {
        DiskWriterConfig cfg;
        cfg.batch_threshold = 10;         // 工作模式：累积更多才刷盘
        cfg.max_pending_time_sec = 60;    // 工作模式：等待时间更长
        return cfg;
    }
};

// 写入器统计信息
struct DiskWriterStats {
    int total_submaps_written = 0;        // 总写入 SubMap 数
    int total_batches = 0;                // 总批次数
    int64_t total_bytes_written = 0;      // 总写入字节数
    double total_write_time_ms = 0.0;     // 总写入耗时
    double avg_batch_size = 0.0;          // 平均批次大小
    double avg_write_time_ms = 0.0;       // 平均单次写入耗时
    int current_pending_count = 0;        // 当前待写数量
};

class DiskWriter {
public:
    DiskWriter();
    ~DiskWriter();
    
    // ========== 生命周期 ==========
    
    // 初始化并启动写入线程
    void initialize(const std::string& submaps_dir, const DiskWriterConfig& config = DiskWriterConfig());
    
    // 停止写入线程并清空队列（会等待所有待写数据落盘）
    void shutdown();
    
    // 是否正在运行
    bool isRunning() const { return writer_running_.load(); }
    
    // ========== 写入接口 ==========
    
    // 添加 SubMap 到写入队列（非阻塞）
    // 如果同一 SubMap 已在队列中，会被新版本覆盖
    void queueSubMap(std::shared_ptr<SubMap> submap);
    
    // 添加多个 SubMap 到写入队列
    void queueSubMaps(const std::vector<std::shared_ptr<SubMap>>& submaps);
    
    // 强制立即写入所有待处理的 SubMap（阻塞直到完成）
    void flushSync();
    
    // 触发一次异步写入（非阻塞）
    void triggerFlush();
    
    // ========== 状态查询 ==========
    
    // 获取当前待写 SubMap 数量
    int getPendingCount() const;
    
    // 获取统计信息
    DiskWriterStats getStats() const;
    
    // 重置统计信息
    void resetStats();
    
private:
    // 写入线程主循环
    void writerLoop();
    
    // 执行批量写入
    void doBatchWrite(std::unordered_map<SubMapID, std::shared_ptr<SubMap>, SubMapIDHash>& batch);
    
    // 写入单个 SubMap
    bool writeSingleSubMap(std::shared_ptr<SubMap> submap);
    
    // 清空剩余待写数据
    void flushRemaining();
    
    // 配置
    DiskWriterConfig config_;
    std::string submaps_dir_;
    
    // 写入线程
    std::thread writer_thread_;
    std::atomic<bool> writer_running_{false};
    
    // 待写队列（使用 SubMapID 作为 key 实现去重）
    std::unordered_map<SubMapID, std::shared_ptr<SubMap>, SubMapIDHash> pending_submaps_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // 强制刷新标记
    std::atomic<bool> force_flush_{false};
    
    // 上次写入时间
    std::chrono::steady_clock::time_point last_write_time_;
    
    // 统计信息
    DiskWriterStats stats_;
    mutable std::mutex stats_mutex_;
};


