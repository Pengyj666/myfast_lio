#include "spa_align.h"

#include "common/sysutils.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/slam3d/edge_se3.h"

#include <map>

using namespace utils;

typedef g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType> SlamLinearSolver;
typedef g2o::OptimizationAlgorithmLevenberg OptimizationAlgo;

std::vector<TimedPose> spa_align(const std::vector<TimedAlignNode> &pg_nodes) {
  std::vector<TimedPose> aligned_pg;
  if (pg_nodes.size() < 10) {
    droslog(LogLevel::ERROR, "align_gnss_vio(): pg_nodes.size() = %d < 10", (int)pg_nodes.size());
    return aligned_pg;
  }

  g2o::SparseOptimizer optimizer;

  auto linearSolver = g2o::make_unique<SlamLinearSolver>();
  linearSolver->setBlockOrdering(false);
  auto blockSolver = g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
  OptimizationAlgo *algorithm = new OptimizationAlgo(std::move(blockSolver));
  optimizer.setAlgorithm(algorithm);

  std::vector<Eigen::Isometry3d> T_vio_vec;
  for (int i = 0; i < (int)pg_nodes.size(); ++i) {
    auto &pg = pg_nodes[i];

    Eigen::Isometry3d T_pose = g2o::Isometry3::Identity();
    T_pose.rotate(pg.quat);
    T_pose.pretranslate(pg.pos);
    T_vio_vec.push_back(T_pose);

    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setEstimate(T_pose);
    v->setId(i);
    if (i == 0) {  // 第一个顶点固定不动，用于建立绝对位置约束（如RTK）
      v->setFixed(true);
    } else {
      v->setFixed(false);
    }
    optimizer.addVertex(v);
  }

  const int numVertices = (int)T_vio_vec.size();


  // 增加base顶点，用于构建rtk约束
  const int BASE_VID = std::numeric_limits<int>::max();
  {
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setEstimate(g2o::Isometry3::Identity());
    v->setId(BASE_VID);
    v->setFixed(true);
    optimizer.addVertex(v);
  }

  // 添加VIO相对位置约束边 和 RTK绝对位置约束
  Eigen::Matrix<double, 6, 6> info_vio = Eigen::Matrix<double, 6, 6>::Identity() * 1.0;
  int last_fix_id = -1, last_fix_id2 = -1;
  for (int i = 0; i < numVertices; ++i) {
    // VIO相对位置约束
    if (i + 1 < numVertices) {
      Eigen::Isometry3d T_ji = T_vio_vec[i+1].inverse() * T_vio_vec[i];
      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      e->setInformation(info_vio);
      e->vertices()[0] = optimizer.vertex(i+1);
      e->vertices()[1] = optimizer.vertex(i);
      e->setMeasurement(T_ji);
      optimizer.addEdge(e);
    }

    auto pg = pg_nodes[i];
    // 前N个VIO相对固定 防止前段轨迹漂移过大
    if (i < 100) {
      Eigen::Isometry3d T_ref_pos = g2o::Isometry3::Identity();
      T_ref_pos.rotate(pg.quat);
      T_ref_pos.pretranslate(pg.pos);

      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      Eigen::Matrix<double, 6, 6> info_vio = Eigen::Matrix<double, 6, 6>::Zero();
      double factor = i * i + 1.0;
      info_vio(0,0) = 0.01 / factor;
      info_vio(1,1) = 0.01 / factor;
      info_vio(2,2) = 0.01 / factor;
  
      e->setInformation(info_vio);
      e->vertices()[0] = optimizer.vertex(BASE_VID);
      e->vertices()[1] = optimizer.vertex(i);
      e->setMeasurement(T_ref_pos);
      optimizer.addEdge(e);
    }

    // 回环约束
    if (pg.loop_info.get()) {
      Eigen::Vector3d ref_pos, cur_pos, cur_pos2; 
      Eigen::Quaterniond ref_q, cur_q, cur_q2;

      ref_pos = pg_nodes[pg.loop_info->ref_id].pos;
      ref_q = pg_nodes[pg.loop_info->ref_id].quat;
      
      cur_pos2 = ref_pos + ref_q * pg.loop_info->relative_t;
      cur_q2 = ref_q * pg.loop_info->relative_q;
      
      Eigen::Isometry3d cur_T2 = g2o::Isometry3::Identity();
      cur_T2.rotate(cur_q2);
      cur_T2.pretranslate(cur_pos2);
      
      Eigen::Isometry3d T_cl = cur_T2.inverse() * T_vio_vec[i];

      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      Eigen::Matrix<double, 6, 6> info_loop = Eigen::Matrix<double, 6, 6>::Zero();
      info_loop(0,0) = pg.loop_info->relative_t_cov(0,0);
      info_loop(1,1) = pg.loop_info->relative_t_cov(1,1);
      info_loop(2,2) = pg.loop_info->relative_t_cov(2,2);

      info_loop(3,3) = pg.loop_info->relative_q_cov(0,0);
      info_loop(4,4) = pg.loop_info->relative_q_cov(1,1);
      info_loop(5,5) = pg.loop_info->relative_q_cov(2,2);

      e->setInformation(info_loop);
      e->vertices()[0] = optimizer.vertex(pg.id);
      e->vertices()[1] = optimizer.vertex(pg.loop_info->ref_id);
      e->setMeasurement(T_cl);
      optimizer.addEdge(e);
    }
    
    // RTK绝对位置约束
    if (pg.ref_pose.get()) {
      Eigen::Isometry3d T_ref_pos = g2o::Isometry3::Identity();
      T_ref_pos.rotate(pg.ref_pose->ref_quat);
      T_ref_pos.pretranslate(pg.ref_pose->ref_pos);
      
      if (pg.ref_pose->ref_pos_valid) {
        g2o::EdgeSE3 *e = new g2o::EdgeSE3();
        Eigen::Matrix<double, 6, 6> info_rtk = Eigen::Matrix<double, 6, 6>::Zero();
        info_rtk(0,0) = 0.1;  
        info_rtk(1,1) = 0.1;
        // info_rtk(2,2) = 0.01;
    
        e->setInformation(info_rtk);
        e->vertices()[0] = optimizer.vertex(BASE_VID);
        e->vertices()[1] = optimizer.vertex(i);
        e->setMeasurement(T_ref_pos);
        optimizer.addEdge(e);
      }
      if (pg.ref_pose->ref_quat_valid) {  
        g2o::EdgeSE3 *e = new g2o::EdgeSE3();
        Eigen::Matrix<double, 6, 6> info_rtk = Eigen::Matrix<double, 6, 6>::Zero();
        info_rtk(3,3) = 0.16;
        info_rtk(4,4) = 0.16;
        info_rtk(5,5) = 0.16;
    
        e->setInformation(info_rtk);
        e->vertices()[0] = optimizer.vertex(BASE_VID);
        e->vertices()[1] = optimizer.vertex(i);
        e->setMeasurement(T_ref_pos);
        optimizer.addEdge(e);
      }
    }

    // VIO横滚俯仰角约束
    {
      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      Eigen::Matrix<double, 6, 6> info_rtk = Eigen::Matrix<double, 6, 6>::Zero();
      info_rtk(3,3) = 0.1;
      info_rtk(4,4) = 0.1;
  
      e->setInformation(info_rtk);
      e->vertices()[0] = optimizer.vertex(BASE_VID);
      e->vertices()[1] = optimizer.vertex(i);
      e->setMeasurement(T_vio_vec[i]);
      optimizer.addEdge(e);
    }    
  }

  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  if (optimizer.chi2() <= 0) {
    droslog(LogLevel::ERROR, "optimize_vio_gnss(): chi2 is not positive");
    return aligned_pg;
  }
  int numOptimization = optimizer.optimize(100);
  if (numOptimization <= 0) {
    droslog(LogLevel::ERROR, "optimize_vio_gnss(): optimization failed");
    return aligned_pg;
  }

  for (int i = 0; i < numVertices; i++) {
    g2o::VertexSE3 *v2AfterOpti = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(i));
    auto se3 = v2AfterOpti->estimate();
    const Eigen::Vector3d t_opt = se3.translation();
    const Eigen::Matrix3d R_opt = se3.rotation();

    TimedPose pose_opt;
    pose_opt.pos = t_opt;
    pose_opt.quat = R_opt;
    pose_opt.timestamp = pg_nodes[i].timestamp;

    aligned_pg.push_back(pose_opt);
  }

  return aligned_pg;
}

/* struct VioWithVreloc {
    double timestamp;
    common::Pose vio;          // VIO原始位姿（VIO坐标系）
    common::Pose align_vio;    // VIO应用上次TF后的位姿（地图坐标系）
    std::shared_ptr<common::Pose> vreloc;  // 重定位位姿（地图坐标系，可能为空）
};

// 传入的是这个结构的向量
std::vector<VioWithVreloc> &vv_vec 

typedef g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType> SlamLinearSolver;*/
common::Data_Pose spa_align(std::vector<VioWithVreloc> &vv_vec, const AlignConfig &config) {
  common::Data_Pose tf_pose;
  if (vv_vec.size() < 20) {
    droslog(LogLevel::ERROR, "spa_align(): too few nodes, size=%d", (int)vv_vec.size());
    return tf_pose;
  }

  g2o::SparseOptimizer optimizer;

  auto linearSolver = g2o::make_unique<SlamLinearSolver>(); //  独占所有权的智能指针
  linearSolver->setBlockOrdering(false); // 通常禁用排序，因为其变量顺序已经过优化
  auto blockSolver = g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
  OptimizationAlgo *algorithm = new OptimizationAlgo(std::move(blockSolver));
  optimizer.setAlgorithm(algorithm);

  std::vector<Eigen::Isometry3d> T_vio_vec, T_vio_aligned_vec;
  std::map<int, Eigen::Isometry3d> T_vreloc_map;
  for (int i = 0; i < (int)vv_vec.size(); ++i) {
    auto &pg = vv_vec[i];

    if (pg.vreloc.get()) {  
      Eigen::Isometry3d T_vreloc = g2o::Isometry3::Identity();
      T_vreloc.rotate(pg.vreloc->quat);
      T_vreloc.pretranslate(pg.vreloc->pos);

      T_vreloc_map[i] = T_vreloc;
    }

    Eigen::Isometry3d T_vio = g2o::Isometry3::Identity();
    T_vio.rotate(pg.vio.quat);
    T_vio.pretranslate(pg.vio.pos);
    T_vio_vec.push_back(T_vio);

    Eigen::Isometry3d T_vio_a = g2o::Isometry3::Identity();
    T_vio_a.rotate(pg.align_vio.quat);
    T_vio_a.pretranslate(pg.align_vio.pos);
    T_vio_aligned_vec.push_back(T_vio_a);

    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setEstimate(T_vio_a);
    v->setId(i);
    v->setFixed(false);
    optimizer.addVertex(v);
  }

  const int numVertices = (int)T_vio_vec.size();  // 通常是2147483647，表示最大整数

  // 增加base顶点，用于构建ref约束
  const int BASE_VID = std::numeric_limits<int>::max();
  {
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setEstimate(g2o::Isometry3::Identity());
    v->setId(BASE_VID);
    v->setFixed(true);  // setFixed(true)使其在优化过程中位置不变
    optimizer.addVertex(v);
  }

  /* 原始VIO的相对变换是最准确的！

VIO的优势：
  - 短期内相对精度高
  - 帧间约束准确
  - 不受全局漂移影响

VIO的劣势：
  - 长期累计漂移
  - 绝对位置不准

所以：
  相对约束 → 用原始VIO（权重1.0，高置信度）
  绝对约束 → 用对齐VIO或重定位（权重0.01-0.1，低-中置信度） */
  // 添加VIO相对位置约束边 和 VIO对齐后约束边 
  Eigen::Matrix<double, 6, 6> info_vio = Eigen::Matrix<double, 6, 6>::Identity() * config.vio_factor;
  int last_fix_id = -1, last_fix_id2 = -1;
  for (int i = 0; i < numVertices; ++i) {
    if (i + 1 < numVertices) {
      // VIO相对位置约束
      Eigen::Isometry3d T_ji = T_vio_vec[i+1].inverse() * T_vio_vec[i];
      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      e->setInformation(info_vio);
      e->vertices()[0] = optimizer.vertex(i+1);
      e->vertices()[1] = optimizer.vertex(i);
      e->setMeasurement(T_ji);
      optimizer.addEdge(e);
    }
    
    // VIO对齐后绝对位置约束 权重0.01
    {
      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      Eigen::Matrix<double, 6, 6> info_vio_a = Eigen::Matrix<double, 6, 6>::Identity() * config.vio_align_factor;
  
      e->setInformation(info_vio_a);
      e->vertices()[0] = optimizer.vertex(BASE_VID); // 基坐标系
      e->vertices()[1] = optimizer.vertex(i);
      e->setMeasurement(T_vio_aligned_vec[i]);
      optimizer.addEdge(e);
    }

    // 视觉重定位约束 权重0.1

    if (T_vreloc_map.count(i) > 0) {
      g2o::EdgeSE3 *e = new g2o::EdgeSE3();

      Eigen::Matrix<double, 6, 6> info_vreloc = Eigen::Matrix<double, 6, 6>::Identity() * config.vio_vreloc_factor;
  
      e->setInformation(info_vreloc);
      e->vertices()[0] = optimizer.vertex(BASE_VID);
      e->vertices()[1] = optimizer.vertex(i);
      e->setMeasurement(T_vreloc_map[i]);
      optimizer.addEdge(e);
    }
  }

  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  if (optimizer.chi2() <= 0) {  // 误差加权卡方检验，确保优化过程稳定
    droslog(LogLevel::ERROR, "spa_align(): chi2 is not positive");
    return tf_pose;
  }
  int numOptimization = optimizer.optimize(100);  // 优化100次，确保优化过程稳定
  if (numOptimization <= 0) {
    droslog(LogLevel::ERROR, "spa_align(): optimization failed");
    return tf_pose;
  }

  // 检查对齐结果, 最新3帧
  for (int i = 0; i < 3; i++) {
    // 对齐后位姿
    g2o::VertexSE3 *v2AfterOpti = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(i));
    auto se3 = v2AfterOpti->estimate();  
    const Eigen::Vector3d t_opt = se3.translation();
    Eigen::Vector3d t_vio_a = T_vio_aligned_vec[i].translation();
    double dist = (t_opt - t_vio_a).norm();  // 计算对齐后位姿与原始VIO位姿的距离
    
    droslog(LogLevel::INFO, "spa_align() 对齐校验: dist=%.3f, align_vio->opt_vio:(%.3f,%.3f,%.3f)->(%.3f,%.3f,%.3f)", 
        dist, t_vio_a[0], t_vio_a[1], t_vio_a[2], t_opt[0], t_opt[1], t_opt[2]);
  }

  // 原位姿
  const Eigen::Isometry3d T_pose = T_vio_vec[0];
  const Eigen::Vector3d t_pose = T_pose.translation();
  const Eigen::Matrix3d R_pose = T_pose.rotation();
  // 对齐后位姿
  g2o::VertexSE3 *v2AfterOpt = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(0));
  auto se3 = v2AfterOpt->estimate();
  const Eigen::Vector3d t_opt = se3.translation();
  const Eigen::Matrix3d R_opt = se3.rotation();

  tf_pose.pose.quat = R_opt * R_pose.transpose();   // 计算对齐后位姿与原始VIO位姿的距离
  tf_pose.pose.pos = t_opt - tf_pose.pose.quat * t_pose;  // 计算对齐后位姿与原始VIO位姿的距离
  tf_pose.timestamp = vv_vec[0].timestamp;

  Eigen::Vector3d rpy = GetEulerRPY(tf_pose.pose.quat);  // 计算对齐后位姿与原始VIO位姿的距离
  droslog(LogLevel::INFO, "spa_align() tf_pos=%.3f, %.3f, %.3f, rpy=%.3f, %.3f, %.3f", 
      tf_pose.pose.pos[0], tf_pose.pose.pos[1], tf_pose.pose.pos[2], rpy[0], rpy[1], rpy[2]);

  return tf_pose;
}