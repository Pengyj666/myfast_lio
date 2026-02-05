#ifndef VMAP_SPA_ALIGN_H
#define VMAP_SPA_ALIGN_H

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "common/data_utils.h"
#include "common/data_type.h"

struct TimedPose {
  double timestamp = -1.0;
  Eigen::Vector3d pos;
  Eigen::Quaterniond quat;
};

struct NodeRefPose {
  bool ref_pos_valid = false;
  bool ref_quat_valid = false;

  Eigen::Vector3d ref_pos = Eigen::Vector3d::Zero();
  Eigen::Quaterniond ref_quat = Eigen::Quaterniond::Identity();
  Eigen::Matrix3d ref_pos_cov = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d ref_quat_cov = Eigen::Matrix3d::Zero();
};

struct NodeLoopInfo {
  int ref_id = -1;
  Eigen::Vector3d relative_t = Eigen::Vector3d::Zero();
  Eigen::Quaterniond relative_q = Eigen::Quaterniond::Identity();
  Eigen::Matrix3d relative_t_cov = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d relative_q_cov = Eigen::Matrix3d::Zero();
};

struct TimedAlignNode {
  int id = -1;
  double timestamp = -1.0;

  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();

  std::shared_ptr<NodeRefPose> ref_pose;
  std::shared_ptr<NodeLoopInfo> loop_info;
};

struct VioWithVreloc {
  double timestamp = 0.0;           // sec
  common::ProbPose vio;             // origin_vio_pose
  std::shared_ptr<common::ProbPose> vreloc;
  common::ProbPose align_vio;       // 对齐后的vio_pose
};

// struct PoseGraphNode {
//   int id = -1;
//   double timestamp = -1.0;
//   // local_pose: vio/lio 前端位姿, 保持不变
//   Eigen::Vector3d local_pos = Eigen::Vector3d::Zero();
//   Eigen::Quaterniond local_quat = Eigen::Quaterniond::Identity();

//   // global_pose: map 坐标系下位姿, 通过对齐/回环优化得到
//   Eigen::Vector3d global_pos = Eigen::Vector3d::Zero();
//   Eigen::Quaterniond global_quat = Eigen::Quaterniond::Identity();
// };

// struct PoseGraphEdge {
//   enum Type {LOOP, RELOC, GNSS_REF, FIXED_REF} type;

//   int id = -1;
//   int ref_id = -1;

//   Eigen::Vector3d ref_pos = Eigen::Vector3d::Zero();
//   Eigen::Quaterniond ref_quat = Eigen::Quaterniond::Identity();
//   Eigen::Matrix3d ref_pos_cov = Eigen::Matrix3d::Zero();
//   Eigen::Matrix3d ref_quat_cov = Eigen::Matrix3d::Zero();
// };

// struct PoseGraphData {
//   std::map<int/* node/KF id */, PoseGraphNode> nodes;
//   std::vector<PoseGraphEdge> edges;   // constraints 
// };

std::vector<TimedPose> spa_align(const std::vector<TimedAlignNode> &pg_nodes);

struct AlignConfig {
  double vio_factor = 1.0;
  double vio_align_factor = 0.01;
  double vio_vreloc_factor = 0.1;
};
common::Data_Pose spa_align(std::vector<VioWithVreloc> &vv_vec, const AlignConfig &config);

#endif //VMAP_SPA_ALIGN_H