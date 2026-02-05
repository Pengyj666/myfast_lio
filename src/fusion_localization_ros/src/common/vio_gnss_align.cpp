#include "common/vio_gnss_align.h"

#include "common/math_utils.h"
#include "common/sysutils.h"
#include "common/log_filters.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/slam3d/edge_se3.h"

typedef g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType> SlamLinearSolver;
typedef g2o::OptimizationAlgorithmLevenberg OptimizationAlgo;

using namespace utils;

common::Data_Pose spa_align(std::vector<SpaNode> &vv_vec, const SpaConfig &config) {
  common::Data_Pose tf_pose;
  if (vv_vec.size() < 20) {
    droslog(LogLevel::ERROR, "spa_align(): too few nodes, size=%d", (int)vv_vec.size());
    return tf_pose;
  }

  if (!config.use_align_pose) {
    droslog(LogLevel::INFO, "spa_align(): 不使用align_pose, 首次对齐");
  }

  g2o::SparseOptimizer optimizer;

  auto linearSolver = g2o::make_unique<SlamLinearSolver>();
  linearSolver->setBlockOrdering(false);
  auto blockSolver = g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
  OptimizationAlgo *algorithm = new OptimizationAlgo(std::move(blockSolver));
  optimizer.setAlgorithm(algorithm);

  std::vector<Eigen::Isometry3d> T_pose_vec, T_align_pose_vec;
  for (int i = 0; i < (int)vv_vec.size(); ++i) {
    auto &node = vv_vec[i];

    Eigen::Isometry3d T_pose = g2o::Isometry3::Identity();
    T_pose.rotate(node.pose.quat);
    T_pose.pretranslate(node.pose.pos);
    T_pose_vec.push_back(T_pose);

    Eigen::Isometry3d T_pose_a = g2o::Isometry3::Identity();
    T_pose_a.rotate(node.align_pose.quat);
    T_pose_a.pretranslate(node.align_pose.pos);
    T_align_pose_vec.push_back(T_pose_a);

    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setEstimate(T_pose_a);
    v->setId(i);
    v->setFixed(false);
    optimizer.addVertex(v);
  }

  const int numVertices = (int)T_pose_vec.size();

  // 增加base顶点，用于构建ref约束
  const int BASE_VID = std::numeric_limits<int>::max();
  {
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setEstimate(g2o::Isometry3::Identity());
    v->setId(BASE_VID);
    v->setFixed(true);
    optimizer.addVertex(v);
  }

  // 添加VIO相对位置约束边 和 VIO对齐后约束边
  Eigen::Matrix<double, 6, 6> info_vio = Eigen::Matrix<double, 6, 6>::Identity() * config.pose_adj_factor;
  for (int i = 0; i < numVertices; ++i) {
    if (i + 1 < numVertices) {
      // VIO相对位置约束
      Eigen::Isometry3d T_ji = T_pose_vec[i+1].inverse() * T_pose_vec[i];
      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      e->setInformation(info_vio);
      e->vertices()[0] = optimizer.vertex(i+1);
      e->vertices()[1] = optimizer.vertex(i);
      e->setMeasurement(T_ji);
      optimizer.addEdge(e);
    }
    
    if (config.use_align_pose) {
      // VIO对齐后绝对位姿约束
      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      Eigen::Matrix<double, 6, 6> info_vio_a = Eigen::Matrix<double, 6, 6>::Identity() * config.pose_align_factor;
      
      e->setInformation(info_vio_a);
      e->vertices()[0] = optimizer.vertex(BASE_VID);
      e->vertices()[1] = optimizer.vertex(i);
      e->setMeasurement(T_align_pose_vec[i]);
      optimizer.addEdge(e);
    }
    {
      // VIO水平姿态约束
      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      Eigen::Matrix<double, 6, 6> info_vio_rp = Eigen::Matrix<double, 6, 6>::Zero();
      info_vio_rp(3,3) = config.pose_rp_factor;
      info_vio_rp(4,4) = config.pose_rp_factor;
      
      e->setInformation(info_vio_rp);
      e->vertices()[0] = optimizer.vertex(BASE_VID);
      e->vertices()[1] = optimizer.vertex(i);
      e->setMeasurement(T_pose_vec[i]);
      optimizer.addEdge(e);
    }

    // gnss约束
    auto &node = vv_vec[i];
    if (node.gnss_ref.get()) {
      Eigen::Isometry3d T_gnss_ref = g2o::Isometry3::Identity();
      T_gnss_ref.pretranslate(node.gnss_ref->pos);

      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      Eigen::Matrix<double, 6, 6> info_rtk = Eigen::Matrix<double, 6, 6>::Zero();
      info_rtk(0,0) = node.gnss_ref->pos_cov(0,0);
      info_rtk(1,1) = node.gnss_ref->pos_cov(1,1);
      info_rtk(2,2) = node.gnss_ref->pos_cov(2,2);
  
      e->setInformation(info_rtk);
      e->vertices()[0] = optimizer.vertex(BASE_VID);
      e->vertices()[1] = optimizer.vertex(i);
      e->setMeasurement(T_gnss_ref);
      optimizer.addEdge(e);
    }

    // 视觉重定位约束
    if (node.reloc_ref.get()) {
      Eigen::Isometry3d T_reloc_ref = g2o::Isometry3::Identity();
      T_reloc_ref.rotate(node.reloc_ref->quat);
      T_reloc_ref.pretranslate(node.reloc_ref->pos);

      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      Eigen::Matrix<double, 6, 6> info_vreloc = Eigen::Matrix<double, 6, 6>::Zero();
      info_vreloc(0,0) = node.reloc_ref->pos_cov(0,0);
      info_vreloc(1,1) = node.reloc_ref->pos_cov(1,1);
      info_vreloc(2,2) = node.reloc_ref->pos_cov(2,2);
      info_vreloc(3,3) = node.reloc_ref->quat_cov(0,0);
      info_vreloc(4,4) = node.reloc_ref->quat_cov(1,1);
      info_vreloc(5,5) = node.reloc_ref->quat_cov(2,2);
  
      e->setInformation(info_vreloc);
      e->vertices()[0] = optimizer.vertex(BASE_VID);
      e->vertices()[1] = optimizer.vertex(i);
      e->setMeasurement(T_reloc_ref);
      optimizer.addEdge(e);
    }
  }

  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  if (optimizer.chi2() <= 0) {
    droslog(LogLevel::ERROR, "spa_align(): chi2 is not positive");
    return tf_pose;
  }
  int numOptimization = optimizer.optimize(100);
  if (numOptimization <= 0) {
    droslog(LogLevel::ERROR, "spa_align(): optimization failed");
    return tf_pose;
  }

  // 检查对齐结果, 最新2帧
  for (int i = 0; i < 2; i++) {
    // 对齐后位姿
    g2o::VertexSE3 *v2AfterOpti = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(i));
    auto se3 = v2AfterOpti->estimate();
    const Eigen::Vector3d t_opt = se3.translation();
    Eigen::Vector3d t_vio_a;
    if (config.use_align_pose) {
      t_vio_a = T_align_pose_vec[i].translation();
    } else {
      t_vio_a = T_pose_vec[i].translation();
    }
    double dist = (t_opt - t_vio_a).norm();
    
    droslog(LogLevel::INFO, "spa_align() 对齐校验: use_align_pose=%d, dist=%.3f, align_vio->opt_vio:(%.3f,%.3f,%.3f)->(%.3f,%.3f,%.3f)", 
        config.use_align_pose, dist, t_vio_a[0], t_vio_a[1], t_vio_a[2], t_opt[0], t_opt[1], t_opt[2]);
  }

  // 原位姿
  const Eigen::Isometry3d T_pose = T_pose_vec[0];
  const Eigen::Vector3d t_pose = T_pose.translation();
  const Eigen::Matrix3d R_pose = T_pose.rotation();
  // 对齐后位姿
  g2o::VertexSE3 *v2AfterOpt = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(0));
  auto se3 = v2AfterOpt->estimate();
  const Eigen::Vector3d t_opt = se3.translation();
  const Eigen::Matrix3d R_opt = se3.rotation();

  tf_pose.pose.quat = R_opt * R_pose.transpose();
  tf_pose.pose.pos = t_opt - tf_pose.pose.quat * t_pose;
  tf_pose.timestamp = vv_vec[0].timestamp;

  Eigen::Vector3d rpy = GetEulerRPY(tf_pose.pose.quat);
  droslog(LogLevel::INFO, "spa_align() tf_pos=%.3f, %.3f, %.3f, rpy=%.3f, %.3f, %.3f", 
      tf_pose.pose.pos[0], tf_pose.pose.pos[1], tf_pose.pose.pos[2], rpy[0], rpy[1], rpy[2]);

  return tf_pose;
}


common::Data_Pose vio_gnss_init(const std::vector<SpaNode> &vv_vec) {
  common::Data_Pose result;

  droslog(LogLevel::INFO, "vio_gnss_init(): vv_vec.size() = %d", (int)vv_vec.size());
  const double result_ts = vv_vec[0].timestamp;

  g2o::SparseOptimizer optimizer;

  auto linearSolver = g2o::make_unique<SlamLinearSolver>();
  linearSolver->setBlockOrdering(false);
  auto blockSolver = g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
  OptimizationAlgo *algorithm = new OptimizationAlgo(std::move(blockSolver));
  optimizer.setAlgorithm(algorithm);

  std::vector<Eigen::Isometry3d> T_pose_vec;
  for (int i = 0; i < (int)vv_vec.size(); ++i) {
    auto &node = vv_vec[i];

    Eigen::Isometry3d T_pose = g2o::Isometry3::Identity();
    T_pose.rotate(node.pose.quat);
    T_pose.pretranslate(node.pose.pos);
    T_pose_vec.push_back(T_pose);

    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setEstimate(T_pose);
    v->setId(i);
    v->setFixed(false);
    optimizer.addVertex(v);
  }

  const int numVertices = (int)T_pose_vec.size();
  int last_gnss_idx = 0;

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
  for (int i = 0; i < numVertices; ++i) {
    if (i + 1 < numVertices) {
      // VIO相对位置约束
      Eigen::Isometry3d T_ji = T_pose_vec[i+1].inverse() * T_pose_vec[i];
      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      e->setInformation(info_vio);
      e->vertices()[0] = optimizer.vertex(i+1);
      e->vertices()[1] = optimizer.vertex(i);
      e->setMeasurement(T_ji);
      optimizer.addEdge(e);
    }

    // VIO姿态约束
    {
      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      Eigen::Matrix<double, 6, 6> info_vio_rp = Eigen::Matrix<double, 6, 6>::Zero();
      info_vio_rp(3,3) = 0.0001;
      info_vio_rp(4,4) = 0.0001;

      e->setInformation(info_vio_rp);
      e->vertices()[0] = optimizer.vertex(BASE_VID);
      e->vertices()[1] = optimizer.vertex(i);
      e->setMeasurement(T_pose_vec[i]);
      optimizer.addEdge(e);
    }
    
    // gnss约束
    auto &node = vv_vec[i];
    if (node.gnss_ref.get()) {
      last_gnss_idx = i;
      Eigen::Isometry3d T_gnss_ref = g2o::Isometry3::Identity();
      T_gnss_ref.pretranslate(node.gnss_ref->pos);

      g2o::EdgeSE3 *e = new g2o::EdgeSE3();
      Eigen::Matrix<double, 6, 6> info_rtk = Eigen::Matrix<double, 6, 6>::Zero();
      info_rtk(0,0) = 0.09;
      info_rtk(1,1) = 0.09;
      info_rtk(2,2) = 0.01;
  
      e->setInformation(info_rtk);
      e->vertices()[0] = optimizer.vertex(BASE_VID);
      e->vertices()[1] = optimizer.vertex(i);
      e->setMeasurement(T_gnss_ref);
      optimizer.addEdge(e);
    }
  }

  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  if (optimizer.chi2() <= 0) {
    droslog(LogLevel::ERROR, "vio_gnss_init(): chi2 is not positive");
    return result;
  }
  int numOptimization = optimizer.optimize(100);
  if (numOptimization <= 0) {
    droslog(LogLevel::ERROR, "vio_gnss_init(): optimization failed");
    return result;
  }
  droslog(LogLevel::INFO, "vio_gnss_init(): chi2: %f, active chi2: %f, num optimization: %d", 
      optimizer.chi2(), optimizer.activeChi2(), numOptimization);
  
  // 原位姿
  const Eigen::Isometry3d T_pose = T_pose_vec[last_gnss_idx];
  const Eigen::Vector3d t_pose = T_pose.translation();
  const Eigen::Matrix3d R_pose = T_pose.rotation();
  // 对齐后位姿
  g2o::VertexSE3 *v2AfterOpt = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(last_gnss_idx));
  auto se3 = v2AfterOpt->estimate();
  const Eigen::Vector3d t_opt = se3.translation();
  const Eigen::Matrix3d R_opt = se3.rotation();

  result.pose.quat = R_opt * R_pose.transpose();
  result.pose.pos = t_opt - result.pose.quat * t_pose;
  result.timestamp = vv_vec[last_gnss_idx].timestamp;

  Eigen::Vector3d rpy = GetEulerRPY(result.pose.quat);
  droslog(LogLevel::INFO, "vio_gnss_init() result: pos=%.3f, %.3f, %.3f, rpy=%.3f, %.3f, %.3f", 
      result.pose.pos[0], result.pose.pos[1], result.pose.pos[2], rpy[0], rpy[1], rpy[2]);

  return result;
}