/*******************************************************
 * SubMapSerializer 类定义
 * 
 * 负责子图的序列化/反序列化
 * 支持二进制格式存储，包含关键帧完整数据（特征点+描述子）
 * 
 * 文件格式版本: V1
 * 
 * 创建日期: 2025-12-15
 *******************************************************/

#pragma once

#include <string>
#include <memory>
#include <vector>
#include <fstream>

#include "submap.h"
#include "keyframe.h"

// 子图文件头（固定大小，便于快速读取）
struct SubMapFileHeader {
    char magic[4] = {'S', 'M', 'A', 'P'};  // 魔数标识
    uint8_t version = 1;                   // 文件格式版本
    uint8_t reserved[3] = {0, 0, 0};       // 保留字段
    int32_t submap_x = 0;                  // 子图 X 坐标
    int32_t submap_y = 0;                  // 子图 Y 坐标
    int32_t cell_count = 0;                // Cell 数量
    int32_t keyframe_count = 0;            // 关键帧总数
    double timestamp = 0.0;                // 序列化时间戳
    uint8_t padding[24] = {0};             // 填充到 64 字节
};

// 关键帧数据块头
struct KeyFrameBlockHeader {
    int32_t kf_index = 0;                  // 关键帧索引
    double timestamp = 0.0;                // 时间戳
    int32_t cell_x = 0;                    // Cell X 坐标
    int32_t cell_y = 0;                    // Cell Y 坐标
    int32_t direction_slot = 0;            // 方向槽位
    int32_t keypoint_count = 0;            // 关键点数量
    int32_t brief_size = 0;                // BRIEF 描述子位数
    int32_t loop_index = -1;               // 回环索引
    
    // 位姿数据（VIO 和优化后）
    double vio_T[3] = {0.0, 0.0, 0.0};     // VIO 位置
    double vio_Q[4] = {1.0, 0.0, 0.0, 0.0}; // VIO 四元数 (w,x,y,z)
    double pg_T[3] = {0.0, 0.0, 0.0};      // 优化后位置
    double pg_Q[4] = {1.0, 0.0, 0.0, 0.0}; // 优化后四元数
    
    // 回环信息
    double loop_info[8] = {0.0};
    
    // RTK 参考信息
    int32_t ref_loc_type = -1;
    double ref_loc_ts = 0.0;
    double ref_loc_xyz[3] = {0.0, 0.0, 0.0};
    double ref_loc_cov[9] = {0.0};
};

class SubMapSerializer {
public:
    SubMapSerializer();
    ~SubMapSerializer();
    
    // ========== 序列化（保存）==========
    
    // 将子图序列化到二进制文件
    // 返回值：true 成功，false 失败
    bool serialize(const std::shared_ptr<SubMap>& submap, const std::string& file_path);
    
    // 序列化多个子图到目录
    bool serializeAll(const std::vector<std::shared_ptr<SubMap>>& submaps, 
                      const std::string& dir_path);
    
    // ========== 反序列化（加载）==========
    
    // 从二进制文件加载子图
    // 返回值：加载成功返回 SubMap，失败返回 nullptr
    std::shared_ptr<SubMap> deserialize(const std::string& file_path);
    
    // 只读取子图头信息（用于快速索引）
    bool readHeader(const std::string& file_path, SubMapFileHeader& header);
    
    // ========== 工具方法 ==========
    
    // 生成子图文件名
    static std::string getSubMapFileName(const SubMapID& id);
    static std::string getSubMapFileName(int submap_x, int submap_y);
    
    // 从文件名解析子图 ID
    static bool parseSubMapFileName(const std::string& filename, SubMapID& id);

private:
    // 序列化单个关键帧
    bool serializeKeyFrame(std::ofstream& out, const std::shared_ptr<KeyFrame>& kf);
    
    // 反序列化单个关键帧
    std::shared_ptr<KeyFrame> deserializeKeyFrame(std::ifstream& in, 
                                                   const KeyFrameBlockHeader& header);
    
    // 验证文件头
    bool validateHeader(const SubMapFileHeader& header);
};


