/*******************************************************
 * SpatialMeta 类定义
 * 
 * 空间地图元信息管理
 * 存储/加载全局配置和子图索引信息
 * 
 * 创建日期: 2025-12-15
 *******************************************************/

#pragma once

#include <string>
#include <vector>
#include <map>

#include "spatial_index.h"

// 子图元信息
struct SubMapMeta {
    SubMapID id;
    int keyframe_count = 0;
    int cell_count = 0;
    double min_x = 0.0, max_x = 0.0;
    double min_y = 0.0, max_y = 0.0;
    std::string file_name;
};

// 全局地图元信息
struct SpatialMapMeta {
    // 版本信息
    std::string version = "1.0";
    
    // 坐标系原点（GPS/RTK 参考点）
    double origin_lat = 0.0;
    double origin_lon = 0.0;
    double origin_alt = 0.0;
    
    // 地图统计
    int total_submaps = 0;
    int total_keyframes = 0;
    int total_cells = 0;
    
    // 参数配置
    double cell_size = 0.25;
    double submap_size = 5.0;
    int num_directions = 6;
    
    // 地图边界
    double map_min_x = 0.0, map_max_x = 0.0;
    double map_min_y = 0.0, map_max_y = 0.0;
    
    // 创建时间
    double create_timestamp = 0.0;
    std::string create_time_str;
    
    // 子图索引
    std::vector<SubMapMeta> submap_metas;
};

class SpatialMetaIO {
public:
    // 保存元信息到 JSON 文件
    static bool save(const SpatialMapMeta& meta, const std::string& file_path);
    
    // 从 JSON 文件加载元信息
    static bool load(const std::string& file_path, SpatialMapMeta& meta);
    
    // 生成元信息（从 SpatialMapManager 收集数据）
    // 注意：需要在 SpatialMapManager 中调用
};


