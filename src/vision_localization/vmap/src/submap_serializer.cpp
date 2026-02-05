/*******************************************************
 * SubMapSerializer 实现
 * 
 * 子图序列化/反序列化实现
 * 
 * 创建日期: 2025-12-15
 *******************************************************/

#include "submap_serializer.h"
#include "parameters.h"

#include <sstream>
#include <ctime>
#include <sys/stat.h>
#include <iomanip>
#include <cstring>
#include <sys/stat.h>

SubMapSerializer::SubMapSerializer() {
}

SubMapSerializer::~SubMapSerializer() {
}

// ========== 序列化（保存）==========

// 获取当前时间戳
static double GetCurrentTimestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

bool SubMapSerializer::serialize(const std::shared_ptr<SubMap>& submap, 
                                  const std::string& file_path) {
    if (!submap) {
        return false;
    }
    
    std::ofstream out(file_path, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }
    
    // 获取所有关键帧
    auto keyframes = submap->getAllKeyFrames();
    
    // 构建文件头
    SubMapFileHeader header;
    header.submap_x = submap->id().x;
    header.submap_y = submap->id().y;
    header.cell_count = submap->getCellCount();
    header.keyframe_count = static_cast<int32_t>(keyframes.size());
    header.timestamp = GetCurrentTimestamp();
    
    // 写入文件头
    out.write(reinterpret_cast<const char*>(&header), sizeof(SubMapFileHeader));
    
    // 写入每个关键帧
    for (const auto& kf : keyframes) {
        if (!serializeKeyFrame(out, kf)) {
            out.close();
            return false;
        }
    }
    
    out.close();
    return true;
}

bool SubMapSerializer::serializeAll(const std::vector<std::shared_ptr<SubMap>>& submaps,
                                     const std::string& dir_path) {
    // 确保目录存在
    struct stat st;
    if (stat(dir_path.c_str(), &st) != 0) {
        mkdir(dir_path.c_str(), 0755);
    }
    
    int success_count = 0;
    for (const auto& submap : submaps) {
        std::string file_path = dir_path + "/" + getSubMapFileName(submap->id());
        if (serialize(submap, file_path)) {
            success_count++;
        }
    }
    
    return success_count == static_cast<int>(submaps.size());
}

bool SubMapSerializer::serializeKeyFrame(std::ofstream& out, 
                                          const std::shared_ptr<KeyFrame>& kf) {
    if (!kf) return false;
    
    // 构建关键帧头
    KeyFrameBlockHeader kf_header;
    kf_header.kf_index = kf->index;
    kf_header.timestamp = kf->time_stamp;
    kf_header.cell_x = kf->cell_x;
    kf_header.cell_y = kf->cell_y;
    kf_header.direction_slot = kf->direction_slot;
    kf_header.keypoint_count = static_cast<int32_t>(kf->keypoints.size());
    // 从描述子获取位数（通常为256）
    kf_header.brief_size = kf->brief_descriptors.empty() ? 256 : 
                           static_cast<int32_t>(kf->brief_descriptors[0].size());
    kf_header.loop_index = kf->loop_index;
    
    // VIO 位姿
    kf_header.vio_T[0] = kf->vio_T_w_i.x();
    kf_header.vio_T[1] = kf->vio_T_w_i.y();
    kf_header.vio_T[2] = kf->vio_T_w_i.z();
    Eigen::Quaterniond vio_q(kf->vio_R_w_i);
    kf_header.vio_Q[0] = vio_q.w();
    kf_header.vio_Q[1] = vio_q.x();
    kf_header.vio_Q[2] = vio_q.y();
    kf_header.vio_Q[3] = vio_q.z();
    
    // 优化后位姿
    kf_header.pg_T[0] = kf->T_w_i.x();
    kf_header.pg_T[1] = kf->T_w_i.y();
    kf_header.pg_T[2] = kf->T_w_i.z();
    Eigen::Quaterniond pg_q(kf->R_w_i);
    kf_header.pg_Q[0] = pg_q.w();
    kf_header.pg_Q[1] = pg_q.x();
    kf_header.pg_Q[2] = pg_q.y();
    kf_header.pg_Q[3] = pg_q.z();
    
    // 回环信息
    for (int i = 0; i < 8; i++) {
        kf_header.loop_info[i] = kf->loop_info(i);
    }
    
    // RTK 参考信息
    kf_header.ref_loc_type = kf->ref_loc_info_.type;
    kf_header.ref_loc_ts = kf->ref_loc_info_.timestamp;
    kf_header.ref_loc_xyz[0] = kf->ref_loc_info_.xyz.x();
    kf_header.ref_loc_xyz[1] = kf->ref_loc_info_.xyz.y();
    kf_header.ref_loc_xyz[2] = kf->ref_loc_info_.xyz.z();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            kf_header.ref_loc_cov[i * 3 + j] = kf->ref_loc_info_.cov(i, j);
        }
    }
    
    // 写入关键帧头
    out.write(reinterpret_cast<const char*>(&kf_header), sizeof(KeyFrameBlockHeader));
    
    // 写入关键点数据
    // 格式: [pt.x, pt.y, norm.x, norm.y] * keypoint_count
    for (size_t i = 0; i < kf->keypoints.size(); i++) {
        float data[4] = {
            kf->keypoints[i].pt.x,
            kf->keypoints[i].pt.y,
            kf->keypoints_norm[i].pt.x,
            kf->keypoints_norm[i].pt.y
        };
        out.write(reinterpret_cast<const char*>(data), sizeof(data));
    }
    
    // 写入 BRIEF 描述子
    // 使用 boost::dynamic_bitset 的序列化
    const size_t desc_bits = static_cast<size_t>(kf_header.brief_size);
    const size_t bytes_per_desc = (desc_bits + 7) / 8;
    std::vector<uint8_t> desc_bytes(bytes_per_desc, 0);
    
    for (const auto& desc : kf->brief_descriptors) {
        std::fill(desc_bytes.begin(), desc_bytes.end(), 0);
        
        for (size_t bit = 0; bit < desc.size(); bit++) {
            if (desc[bit]) {
                desc_bytes[bit / 8] |= (1 << (bit % 8));
            }
        }
        out.write(reinterpret_cast<const char*>(desc_bytes.data()), bytes_per_desc);
    }
    
    return out.good();
}

// ========== 反序列化（加载）==========

std::shared_ptr<SubMap> SubMapSerializer::deserialize(const std::string& file_path) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) {
        return nullptr;
    }
    
    // 读取文件头
    SubMapFileHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(SubMapFileHeader));
    
    if (!validateHeader(header)) {
        in.close();
        return nullptr;
    }
    
    // 创建子图
    SubMapID id;
    id.x = header.submap_x;
    id.y = header.submap_y;
    auto submap = std::make_shared<SubMap>(id);
    
    // 读取每个关键帧
    for (int i = 0; i < header.keyframe_count; i++) {
        // 读取关键帧头
        KeyFrameBlockHeader kf_header;
        in.read(reinterpret_cast<char*>(&kf_header), sizeof(KeyFrameBlockHeader));
        
        if (!in.good()) {
            in.close();
            return nullptr;
        }
        
        auto kf = deserializeKeyFrame(in, kf_header);
        if (!kf) {
            in.close();
            return nullptr;
        }
        
        // 设置空间索引信息
        kf->submap_x = header.submap_x;
        kf->submap_y = header.submap_y;
        
        // 插入到子图
        submap->tryInsertKeyFrame(kf);
    }
    
    in.close();
    return submap;
}

std::shared_ptr<KeyFrame> SubMapSerializer::deserializeKeyFrame(std::ifstream& in,
                                                                  const KeyFrameBlockHeader& header) {
    // 读取关键点数据
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> keypoints_norm;
    keypoints.reserve(header.keypoint_count);
    keypoints_norm.reserve(header.keypoint_count);
    
    for (int i = 0; i < header.keypoint_count; i++) {
        float data[4];
        in.read(reinterpret_cast<char*>(data), sizeof(data));
        
        cv::KeyPoint kp;
        kp.pt.x = data[0];
        kp.pt.y = data[1];
        keypoints.push_back(kp);
        
        cv::KeyPoint kp_norm;
        kp_norm.pt.x = data[2];
        kp_norm.pt.y = data[3];
        keypoints_norm.push_back(kp_norm);
    }
    
    // 读取 BRIEF 描述子
    std::vector<BRIEF::bitset> brief_descriptors;
    brief_descriptors.reserve(header.keypoint_count);
    
    const size_t desc_bits = static_cast<size_t>(header.brief_size);
    const size_t bytes_per_desc = (desc_bits + 7) / 8;
    std::vector<uint8_t> desc_bytes(bytes_per_desc);
    
    for (int i = 0; i < header.keypoint_count; i++) {
        in.read(reinterpret_cast<char*>(desc_bytes.data()), bytes_per_desc);
        
        BRIEF::bitset desc(desc_bits);
        for (size_t bit = 0; bit < desc_bits; bit++) {
            desc[bit] = (desc_bytes[bit / 8] >> (bit % 8)) & 1;
        }
        brief_descriptors.push_back(desc);
    }
    
    if (!in.good()) {
        return nullptr;
    }
    
    // 构建位姿数据
    Eigen::Vector3d vio_T(header.vio_T[0], header.vio_T[1], header.vio_T[2]);
    Eigen::Quaterniond vio_Q(header.vio_Q[0], header.vio_Q[1], header.vio_Q[2], header.vio_Q[3]);
    Eigen::Matrix3d vio_R = vio_Q.toRotationMatrix();
    
    Eigen::Vector3d pg_T(header.pg_T[0], header.pg_T[1], header.pg_T[2]);
    Eigen::Quaterniond pg_Q(header.pg_Q[0], header.pg_Q[1], header.pg_Q[2], header.pg_Q[3]);
    Eigen::Matrix3d pg_R = pg_Q.toRotationMatrix();
    
    // 回环信息
    Eigen::Matrix<double, 8, 1> loop_info;
    for (int i = 0; i < 8; i++) {
        loop_info(i) = header.loop_info[i];
    }
    
    // 创建关键帧（使用加载构造函数）
    cv::Mat empty_image;  // 加载时不需要图像
    auto kf = std::make_shared<KeyFrame>(
        header.timestamp,
        header.kf_index,
        vio_T, vio_R,
        pg_T, pg_R,
        empty_image,
        header.loop_index,
        loop_info,
        keypoints, keypoints_norm, brief_descriptors
    );
    
    // 设置空间索引信息
    kf->cell_x = header.cell_x;
    kf->cell_y = header.cell_y;
    kf->direction_slot = header.direction_slot;
    
    // 设置 RTK 参考信息
    RefLocInfo rli;
    rli.type = header.ref_loc_type;
    rli.timestamp = header.ref_loc_ts;
    rli.xyz = Eigen::Vector3d(header.ref_loc_xyz[0], header.ref_loc_xyz[1], header.ref_loc_xyz[2]);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            rli.cov(i, j) = header.ref_loc_cov[i * 3 + j];
        }
    }
    kf->SetRefLocInfo(rli);
    
    return kf;
}

bool SubMapSerializer::readHeader(const std::string& file_path, SubMapFileHeader& header) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    
    in.read(reinterpret_cast<char*>(&header), sizeof(SubMapFileHeader));
    in.close();
    
    return validateHeader(header);
}

bool SubMapSerializer::validateHeader(const SubMapFileHeader& header) {
    // 检查魔数
    if (header.magic[0] != 'S' || header.magic[1] != 'M' || 
        header.magic[2] != 'A' || header.magic[3] != 'P') {
        return false;
    }
    
    // 检查版本
    if (header.version != 1) {
        return false;
    }
    
    return true;
}

// ========== 工具方法 ==========

std::string SubMapSerializer::getSubMapFileName(const SubMapID& id) {
    return getSubMapFileName(id.x, id.y);
}

std::string SubMapSerializer::getSubMapFileName(int submap_x, int submap_y) {
    std::ostringstream oss;
    // 格式: submap_x_y.smap (x 和 y 可能为负数)
    oss << "submap_" << submap_x << "_" << submap_y << ".smap";
    return oss.str();
}

bool SubMapSerializer::parseSubMapFileName(const std::string& filename, SubMapID& id) {
    // 格式: submap_x_y.smap
    if (filename.substr(0, 7) != "submap_" || filename.substr(filename.size() - 5) != ".smap") {
        return false;
    }
    
    std::string coords = filename.substr(7, filename.size() - 12);  // 去掉 "submap_" 和 ".smap"
    
    // 查找分隔符（第二个下划线或负号后的下划线）
    size_t sep_pos = coords.rfind('_');
    if (sep_pos == std::string::npos || sep_pos == 0) {
        return false;
    }
    
    // 处理可能的负数坐标
    // 格式可能是: "1_2", "-1_2", "1_-2", "-1_-2"
    std::string x_str, y_str;
    
    // 找到真正的分隔位置（第一个坐标可能是负数）
    if (coords[0] == '-') {
        // x 是负数，从第二个字符开始找下划线
        sep_pos = coords.find('_', 1);
    } else {
        sep_pos = coords.find('_');
    }
    
    if (sep_pos == std::string::npos) {
        return false;
    }
    
    x_str = coords.substr(0, sep_pos);
    y_str = coords.substr(sep_pos + 1);
    
    try {
        id.x = std::stoi(x_str);
        id.y = std::stoi(y_str);
    } catch (...) {
        return false;
    }
    
    return true;
}

