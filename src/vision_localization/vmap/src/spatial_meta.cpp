/*******************************************************
 * SpatialMetaIO 实现
 * 
 * 空间地图元信息的 JSON 序列化/反序列化
 * 使用简单的文本格式，不依赖 JSON 库
 * 
 * 创建日期: 2025-12-15
 *******************************************************/

#include "spatial_meta.h"

#include <fstream>
#include <sstream>
#include <iomanip>

bool SpatialMetaIO::save(const SpatialMapMeta& meta, const std::string& file_path) {
    std::ofstream out(file_path);
    if (!out.is_open()) {
        return false;
    }
    
    // 使用简单的键值对格式，便于解析
    out << "# Spatial Map Meta File" << std::endl;
    out << "# Created: " << meta.create_time_str << std::endl;
    out << std::endl;
    
    out << "[version]" << std::endl;
    out << "format=" << meta.version << std::endl;
    out << std::endl;
    
    out << "[origin]" << std::endl;
    out << std::fixed << std::setprecision(10);
    out << "lat=" << meta.origin_lat << std::endl;
    out << "lon=" << meta.origin_lon << std::endl;
    out << "alt=" << meta.origin_alt << std::endl;
    out << std::endl;
    
    out << "[statistics]" << std::endl;
    out << "total_submaps=" << meta.total_submaps << std::endl;
    out << "total_keyframes=" << meta.total_keyframes << std::endl;
    out << "total_cells=" << meta.total_cells << std::endl;
    out << std::endl;
    
    out << "[parameters]" << std::endl;
    out << std::fixed << std::setprecision(4);
    out << "cell_size=" << meta.cell_size << std::endl;
    out << "submap_size=" << meta.submap_size << std::endl;
    out << "num_directions=" << meta.num_directions << std::endl;
    out << std::endl;
    
    out << "[bounds]" << std::endl;
    out << std::fixed << std::setprecision(4);
    out << "min_x=" << meta.map_min_x << std::endl;
    out << "max_x=" << meta.map_max_x << std::endl;
    out << "min_y=" << meta.map_min_y << std::endl;
    out << "max_y=" << meta.map_max_y << std::endl;
    out << std::endl;
    
    out << "[time]" << std::endl;
    out << std::fixed << std::setprecision(6);
    out << "create_timestamp=" << meta.create_timestamp << std::endl;
    out << "create_time=" << meta.create_time_str << std::endl;
    out << std::endl;
    
    // 子图索引
    out << "[submaps]" << std::endl;
    out << "count=" << meta.submap_metas.size() << std::endl;
    for (size_t i = 0; i < meta.submap_metas.size(); i++) {
        const auto& sm = meta.submap_metas[i];
        // 格式: id_x,id_y,keyframes,cells,min_x,max_x,min_y,max_y,filename
        out << "submap_" << i << "=" 
            << sm.id.x << "," << sm.id.y << ","
            << sm.keyframe_count << "," << sm.cell_count << ","
            << std::fixed << std::setprecision(2)
            << sm.min_x << "," << sm.max_x << ","
            << sm.min_y << "," << sm.max_y << ","
            << sm.file_name << std::endl;
    }
    
    out.close();
    return true;
}

bool SpatialMetaIO::load(const std::string& file_path, SpatialMapMeta& meta) {
    std::ifstream in(file_path);
    if (!in.is_open()) {
        return false;
    }
    
    std::string line;
    std::string current_section;
    int submap_count = 0;
    
    while (std::getline(in, line)) {
        // 跳过空行和注释
        if (line.empty() || line[0] == '#') continue;
        
        // 检测节
        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.size() - 2);
            continue;
        }
        
        // 解析键值对
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        
        // 根据节和键解析
        if (current_section == "version") {
            if (key == "format") meta.version = value;
        }
        else if (current_section == "origin") {
            if (key == "lat") meta.origin_lat = std::stod(value);
            else if (key == "lon") meta.origin_lon = std::stod(value);
            else if (key == "alt") meta.origin_alt = std::stod(value);
        }
        else if (current_section == "statistics") {
            if (key == "total_submaps") meta.total_submaps = std::stoi(value);
            else if (key == "total_keyframes") meta.total_keyframes = std::stoi(value);
            else if (key == "total_cells") meta.total_cells = std::stoi(value);
        }
        else if (current_section == "parameters") {
            if (key == "cell_size") meta.cell_size = std::stod(value);
            else if (key == "submap_size") meta.submap_size = std::stod(value);
            else if (key == "num_directions") meta.num_directions = std::stoi(value);
        }
        else if (current_section == "bounds") {
            if (key == "min_x") meta.map_min_x = std::stod(value);
            else if (key == "max_x") meta.map_max_x = std::stod(value);
            else if (key == "min_y") meta.map_min_y = std::stod(value);
            else if (key == "max_y") meta.map_max_y = std::stod(value);
        }
        else if (current_section == "time") {
            if (key == "create_timestamp") meta.create_timestamp = std::stod(value);
            else if (key == "create_time") meta.create_time_str = value;
        }
        else if (current_section == "submaps") {
            if (key == "count") {
                submap_count = std::stoi(value);
                meta.submap_metas.reserve(submap_count);
            }
            else if (key.substr(0, 7) == "submap_") {
                // 解析子图信息
                SubMapMeta sm;
                std::istringstream ss(value);
                std::string token;
                std::vector<std::string> tokens;
                while (std::getline(ss, token, ',')) {
                    tokens.push_back(token);
                }
                if (tokens.size() >= 9) {
                    sm.id.x = std::stoi(tokens[0]);
                    sm.id.y = std::stoi(tokens[1]);
                    sm.keyframe_count = std::stoi(tokens[2]);
                    sm.cell_count = std::stoi(tokens[3]);
                    sm.min_x = std::stod(tokens[4]);
                    sm.max_x = std::stod(tokens[5]);
                    sm.min_y = std::stod(tokens[6]);
                    sm.max_y = std::stod(tokens[7]);
                    sm.file_name = tokens[8];
                    meta.submap_metas.push_back(sm);
                }
            }
        }
    }
    
    in.close();
    return true;
}

