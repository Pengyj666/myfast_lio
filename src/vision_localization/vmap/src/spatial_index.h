/*******************************************************
 * 空间索引定义
 * 
 * 用于动态地图加载系统的空间索引结构体定义
 * 包含子图ID、Cell ID及其哈希函数
 * 
 * 创建日期: 2025-12-10
 *******************************************************/

#pragma once

#include <functional>

// 子图ID（5×5m 区域）
// 用于动态加载的基本单元，每个子图包含 20×20 个 Cell（Cell为0.25m）
struct SubMapID {
    int x = 0;  // floor(position.x / 5.0)
    int y = 0;  // floor(position.y / 5.0)
    
    SubMapID() = default;
    SubMapID(int _x, int _y) : x(_x), y(_y) {}
    
    bool operator==(const SubMapID& other) const {
        return x == other.x && y == other.y;
    }
    
    bool operator!=(const SubMapID& other) const {
        return !(*this == other);
    }
};

// Cell ID（0.25×0.25m 栅格）
// 空间划分的最小单元，每个 Cell 包含 6 个方向槽位
struct CellID {
    int x = 0;  // floor(position.x / 0.25)
    int y = 0;  // floor(position.y / 0.25)
    
    CellID() = default;
    CellID(int _x, int _y) : x(_x), y(_y) {}
    
    bool operator==(const CellID& other) const {
        return x == other.x && y == other.y;
    }
    
    bool operator!=(const CellID& other) const {
        return !(*this == other);
    }
};

// 哈希函数（用于 unordered_map）
// 原理：将 (x, y) 组合成一个整数，x 和 y 错开位置避免 (1,2) 和 (2,1) 产生相同哈希值
namespace std {
    template<>
    struct hash<SubMapID> {
        size_t operator()(const SubMapID& id) const {
            // 使用异或和位移组合 x, y 的哈希值
            return hash<int>()(id.x) ^ (hash<int>()(id.y) << 16);
        }
    };
    
    template<>
    struct hash<CellID> {
        size_t operator()(const CellID& id) const {
            return hash<int>()(id.x) ^ (hash<int>()(id.y) << 16);
        }
    };
}

// 独立的哈希结构体（用于显式指定哈希函数）
struct SubMapIDHash {
    size_t operator()(const SubMapID& id) const {
        return std::hash<int>()(id.x) ^ (std::hash<int>()(id.y) << 16);
    }
};

struct CellIDHash {
    size_t operator()(const CellID& id) const {
        return std::hash<int>()(id.x) ^ (std::hash<int>()(id.y) << 16);
    }
};

// ========== 优化质量等级 - 2025-12-23 ==========
// 用于标记关键帧/子图的优化质量，后续可根据质量筛选
enum OptimizationQuality {
    QUALITY_UNKNOWN = -1,      // 未知/未优化
    QUALITY_VIO_ONLY = 0,      // 仅 VIO，质量最低
    QUALITY_VIO_RELOC = 1,     // VIO + 重定位
    QUALITY_VIO_LOOP = 2,      // VIO + 回环
    QUALITY_VIO_RTK = 3        // VIO + RTK，质量最高
};

// ========== 分层缓存：关键帧轻量元数据 - 2025-12-22 ==========
// 层级2：轻量元数据，始终保留在内存（约 200 字节/帧）
// 用于快速空间查询、回环候选筛选等，无需加载完整关键帧数据
struct KeyFrameMetadata {
    int index = -1;                   // 关键帧索引（与词袋数据库 ID 对应）
    double timestamp = 0.0;           // 时间戳
    
    // 位姿信息（世界坐标系）
    double pos_x = 0.0;
    double pos_y = 0.0;
    double pos_z = 0.0;
    double quat_w = 1.0;
    double quat_x = 0.0;
    double quat_y = 0.0;
    double quat_z = 0.0;
    
    // 空间索引信息
    int submap_x = 0;
    int submap_y = 0;
    int cell_x = 0;
    int cell_y = 0;
    int direction_slot = 0;
    
    // 回环信息
    bool has_loop = false;
    int loop_index = -1;
    
    // 重定位信息（工作模式）
    bool reloc_success = false;       // 是否重定位成功过
    int reloc_match_index = -1;       // 匹配的历史关键帧 ID
    
    // 约束信息
    int ref_loc_type = -1;            // -1:无, 0:充电桩, 1:RTK_FIX, 2:RTK_FLOAT
    
    // 优化质量
    OptimizationQuality opt_quality = QUALITY_UNKNOWN;
    
    // 数据加载状态
    bool full_data_loaded = false;    // 完整数据是否已加载到内存
    
    // 从 SubMapID/CellID 获取
    SubMapID getSubMapID() const { return SubMapID{submap_x, submap_y}; }
    CellID getCellID() const { return CellID{cell_x, cell_y}; }
    
    // ========== 从完整 KeyFrame 构造 - 2025-12-30 ==========
    // 用于将完整 KeyFrame 转换为轻量元数据
    void fromKeyFramePose(int kf_index, double kf_ts,
                          double px, double py, double pz,
                          double qw, double qx, double qy, double qz,
                          int sm_x, int sm_y, int c_x, int c_y, int dir_slot) {
        index = kf_index;
        timestamp = kf_ts;
        pos_x = px; pos_y = py; pos_z = pz;
        quat_w = qw; quat_x = qx; quat_y = qy; quat_z = qz;
        submap_x = sm_x; submap_y = sm_y;
        cell_x = c_x; cell_y = c_y;
        direction_slot = dir_slot;
        full_data_loaded = false;
    }
};

