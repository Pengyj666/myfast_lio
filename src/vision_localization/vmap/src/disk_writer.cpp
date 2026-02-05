/*******************************************************
 * DiskWriter 类实现
 * 
 * 异步磁盘写入器：独立线程执行 SubMap 序列化
 * 
 * 创建日期: 2025-12-15
 *******************************************************/

#include "disk_writer.h"
#include "submap_serializer.h"
#include "droslog/log.h"

#include <sys/stat.h>
#include <sys/types.h>

using namespace utils;

DiskWriter::DiskWriter() {
    last_write_time_ = std::chrono::steady_clock::now();
}

DiskWriter::~DiskWriter() {
    if (writer_running_) {
        shutdown();
    }
}

void DiskWriter::initialize(const std::string& submaps_dir, const DiskWriterConfig& config) {
    if (writer_running_) {
        droslog(LogLevel::WARN, "DiskWriter::initialize() 已经在运行");
        return;
    }
    
    submaps_dir_ = submaps_dir;
    config_ = config;
    
    // 确保目录存在（递归创建）
    if (!submaps_dir_.empty()) {
        // 简单的递归目录创建
        std::string path = submaps_dir_;
        size_t pos = 0;
        while ((pos = path.find('/', pos + 1)) != std::string::npos) {
            std::string subpath = path.substr(0, pos);
            mkdir(subpath.c_str(), 0755);
        }
        mkdir(path.c_str(), 0755);
    }
    
    // 启动写入线程
    writer_running_ = true;
    writer_thread_ = std::thread(&DiskWriter::writerLoop, this);
    
    droslog(LogLevel::INFO, "DiskWriter::initialize() 启动完成: dir=%s, batch=%d, timeout=%ds",
            submaps_dir_.c_str(), config_.batch_threshold, config_.max_pending_time_sec);
}

void DiskWriter::shutdown() {
    if (!writer_running_) {
        return;
    }
    
    droslog(LogLevel::INFO, "DiskWriter::shutdown() 开始关闭...");
    
    // 通知写入线程退出
    writer_running_ = false;
    queue_cv_.notify_one();
    
    // 等待线程结束
    if (writer_thread_.joinable()) {
        writer_thread_.join();
    }
    
    // 清空剩余数据
    flushRemaining();
    
    droslog(LogLevel::INFO, "DiskWriter::shutdown() 关闭完成, 总写入: %d submaps",
            stats_.total_submaps_written);
}

void DiskWriter::queueSubMap(std::shared_ptr<SubMap> submap) {
    if (!submap || !writer_running_) return;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        // 使用 SubMapID 作为 key，自动去重（同一 SubMap 多次修改只保留最新版本）
        pending_submaps_[submap->id()] = submap;
    }
    
    // 检查是否达到批量阈值
    if (static_cast<int>(pending_submaps_.size()) >= config_.batch_threshold) {
        queue_cv_.notify_one();
    }
}

void DiskWriter::queueSubMaps(const std::vector<std::shared_ptr<SubMap>>& submaps) {
    if (submaps.empty() || !writer_running_) return;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (auto& submap : submaps) {
            if (submap) {
                pending_submaps_[submap->id()] = submap;
            }
        }
    }
    
    // 检查是否达到批量阈值
    if (static_cast<int>(pending_submaps_.size()) >= config_.batch_threshold) {
        queue_cv_.notify_one();
    }
}

void DiskWriter::flushSync() {
    if (!writer_running_) {
        // 如果线程未运行，直接同步写入
        flushRemaining();
        return;
    }
    
    // 触发异步写入
    force_flush_ = true;
    queue_cv_.notify_one();
    
    // 等待队列清空
    while (true) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (pending_submaps_.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void DiskWriter::triggerFlush() {
    force_flush_ = true;
    queue_cv_.notify_one();
}

int DiskWriter::getPendingCount() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return static_cast<int>(pending_submaps_.size());
}

DiskWriterStats DiskWriter::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    DiskWriterStats result = stats_;
    
    // 更新当前待写数量
    {
        std::lock_guard<std::mutex> qlock(queue_mutex_);
        result.current_pending_count = static_cast<int>(pending_submaps_.size());
    }
    
    // 计算平均值
    if (result.total_batches > 0) {
        result.avg_batch_size = static_cast<double>(result.total_submaps_written) / result.total_batches;
        result.avg_write_time_ms = result.total_write_time_ms / result.total_batches;
    }
    
    return result;
}

void DiskWriter::resetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = DiskWriterStats();
}

void DiskWriter::writerLoop() {
    droslog(LogLevel::INFO, "DiskWriter::writerLoop() 写入线程启动");
    
    while (writer_running_) {
        std::unordered_map<SubMapID, std::shared_ptr<SubMap>, SubMapIDHash> batch;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // 等待条件：
            // 1. 队列达到阈值
            // 2. 超时
            // 3. 强制刷新
            // 4. 退出信号
            auto timeout = std::chrono::seconds(config_.max_pending_time_sec);
            queue_cv_.wait_for(lock, timeout, [this] {
                return static_cast<int>(pending_submaps_.size()) >= config_.batch_threshold ||
                       force_flush_.load() ||
                       !writer_running_;
            });
            
            // 检查是否超时需要写入
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_write_time_).count();
            
            bool should_write = 
                static_cast<int>(pending_submaps_.size()) >= config_.batch_threshold ||
                force_flush_.load() ||
                (elapsed >= config_.max_pending_time_sec && !pending_submaps_.empty()) ||
                !writer_running_;
            
            if (should_write && !pending_submaps_.empty()) {
                // 取出所有待写 SubMap
                std::swap(batch, pending_submaps_);
                force_flush_ = false;
            }
        }
        
        // 执行批量写入（在锁外执行，不阻塞主线程）
        if (!batch.empty()) {
            doBatchWrite(batch);
        }
    }
    
    droslog(LogLevel::INFO, "DiskWriter::writerLoop() 写入线程退出");
}

void DiskWriter::doBatchWrite(std::unordered_map<SubMapID, std::shared_ptr<SubMap>, SubMapIDHash>& batch) {
    if (batch.empty()) return;
    
    auto start_time = std::chrono::steady_clock::now();
    
    int success_count = 0;
    int64_t bytes_written = 0;
    
    // 兼容 C++11：使用迭代器遍历
    for (auto it = batch.begin(); it != batch.end(); ++it) {
        auto& submap = it->second;
        if (writeSingleSubMap(submap)) {
            success_count++;
            // 估算写入大小（实际大小需要从序列化器获取）
            bytes_written += submap->getKeyFrameCount() * 2048;  // 估算每帧约 2KB
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // 更新统计
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_submaps_written += success_count;
        stats_.total_batches++;
        stats_.total_bytes_written += bytes_written;
        stats_.total_write_time_ms += elapsed_ms;
    }
    
    last_write_time_ = end_time;
    
    droslog(LogLevel::INFO, "DiskWriter::doBatchWrite() 批量写入完成: count=%d, time=%.1fms",
            success_count, elapsed_ms);
}

bool DiskWriter::writeSingleSubMap(std::shared_ptr<SubMap> submap) {
    if (!submap || submaps_dir_.empty()) return false;
    
    try {
        SubMapSerializer serializer;
        // 2026-01-17: 修复 BUG - serialize 需要完整文件路径，不是目录
        std::string file_path = submaps_dir_ + "/" + SubMapSerializer::getSubMapFileName(submap->id());
        bool success = serializer.serialize(submap, file_path);
        if (success) {
            submap->clearDirty();
        }
        return success;
    } catch (const std::exception& e) {
        droslog(LogLevel::ERROR, "DiskWriter::writeSingleSubMap() 写入失败: submap(%d,%d), error=%s",
                submap->id().x, submap->id().y, e.what());
        return false;
    }
}

void DiskWriter::flushRemaining() {
    std::unordered_map<SubMapID, std::shared_ptr<SubMap>, SubMapIDHash> remaining;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        std::swap(remaining, pending_submaps_);
    }
    
    if (!remaining.empty()) {
        droslog(LogLevel::INFO, "DiskWriter::flushRemaining() 清空剩余数据: count=%d", (int)remaining.size());
        doBatchWrite(remaining);
    }
}


