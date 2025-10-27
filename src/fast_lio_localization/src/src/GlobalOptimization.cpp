#include "GlobalOptimization.h"
#include <iostream>
using namespace std;

// 准备数据点
std::vector<double> timestamps;
std::vector<Eigen::Vector3d> positions;
// 定义全局变量

GlobalOptimization::GlobalOptimization()
{
}

GlobalOptimization::~GlobalOptimization()
{
}

// 相对位姿残差：以i到j的相对平移和旋转作为观测
struct RelativePoseResidual {
    RelativePoseResidual(const Eigen::Vector3d& t_ij, const Eigen::Quaterniond& q_ij, double weight)
        : t_ij_(t_ij), q_ij_(q_ij), weight_(weight) {
            // 确保输入数据有效
            if (!t_ij.allFinite()) {
                std::cerr << "Warning: t_ij contains NaN or Inf values" << std::endl;
            }
            if (!q_ij.vec().allFinite() || std::abs(q_ij.norm() - 1.0) > 1e-3) {
                std::cerr << "Warning: q_ij is not valid or not normalized" << std::endl;
            }
            if (!std::isfinite(weight) || weight <= 0) {
                std::cerr << "Warning: weight is invalid: " << weight << std::endl;
            }
        }

    template <typename T>
    bool operator()(const T* const qi, const T* const ti,
                    const T* const qj, const T* const tj,
                    T* residuals) const {
        // qi, qj: quaternion (w,x,y,z)
        Eigen::Quaternion<T> Qi(qi[0], qi[1], qi[2], qi[3]);
        Eigen::Quaternion<T> Qj(qj[0], qj[1], qj[2], qj[3]);

        Qi.normalize();
        Qj.normalize();
        // 计算预期相对变换 Qij = Qi^{-1} * Qj
        Eigen::Quaternion<T> Qij = Qi.conjugate() * Qj;

        // 旋转残差（四元数差的小角度近似）
        Eigen::Matrix<T,3,1> r_rot = T(2.0) * Qij.vec();

        // 平移残差 tij_obs - (Qi^{-1} * (tj - ti))
        Eigen::Matrix<T,3,1> Ti(ti[0], ti[1], ti[2]);
        Eigen::Matrix<T,3,1> Tj(tj[0], tj[1], tj[2]);
        Eigen::Matrix<T,3,1> pred = Qi.conjugate() * (Tj - Ti);
        Eigen::Matrix<T,3,1> r_trans = pred - Eigen::Matrix<T,3,1>(T(t_ij_.x()), T(t_ij_.y()), T(t_ij_.z()));

        // 组合 residuals: [tx,ty,tz, rx,ry,rz]
        residuals[0] = T(weight_) * r_trans(0);
        residuals[1] = T(weight_) * r_trans(1);
        residuals[2] = T(weight_) * r_trans(2);
        residuals[3] = T(weight_) * r_rot(0);
        residuals[4] = T(weight_) * r_rot(1);
        residuals[5] = T(weight_) * r_rot(2);

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& t_ij, const Eigen::Quaterniond& q_ij, double weight) {
        return new ceres::AutoDiffCostFunction<RelativePoseResidual, 6, 4, 3, 4, 3>(
            new RelativePoseResidual(t_ij, q_ij, weight));
    }

    Eigen::Vector3d t_ij_;
    Eigen::Quaterniond q_ij_;
    double weight_;
};

void GlobalOptimization::optimize() {
    if (localizationPoseMap.size() > 50) {
        localizationPoseMap.erase(localizationPoseMap.begin());
        best_inliers.erase(best_inliers.begin());
    }


    // 将localizationPoseMap按时间戳排序并拷贝到向量中
    std::vector<double> keys;
    for (const auto &kv : localizationPoseMap) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());

    int N = (int)keys.size();

    // 参数块：为每个位姿建立 q(4) 和 t(3)
    std::vector<std::array<double,4>> qs(N);
    std::vector<std::array<double,3>> ts(N);

    for (int i = 0; i < N; ++i) {
        const auto &pose = localizationPoseMap[keys[i]];
        if (pose.size() != 7) {
            std::cerr << "Invalid pose data size: " << pose.size() << std::endl;
            return;
        }
        // pose = {x,y,z, qw, qx, qy, qz}
        ts[i][0] = pose[0]; ts[i][1] = pose[1]; ts[i][2] = pose[2];
        qs[i][0] = pose[3]; qs[i][1] = pose[4]; qs[i][2] = pose[5]; qs[i][3] = pose[6];

            // 规范化四元数
        double norm = sqrt(qs[i][0]*qs[i][0] + qs[i][1]*qs[i][1] + qs[i][2]*qs[i][2] + qs[i][3]*qs[i][3]);
        if (norm > 0) {
            qs[i][0] /= norm;
            qs[i][1] /= norm;
            qs[i][2] /= norm;
            qs[i][3] /= norm;
        }else {
            std::cerr << "Quaternion norm too small: " << norm << std::endl;
            // 设置为单位四元数
            qs[i][0] = 1.0;
            qs[i][1] = 0.0;
            qs[i][2] = 0.0;
            qs[i][3] = 0.0;
        }
    }

    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 20;
    ceres::Solver::Summary summary;

    ceres::LocalParameterization* quat_param = new ceres::QuaternionParameterization();

    // 添加参数块
    for (int i = 0; i < N; ++i) {
        problem.AddParameterBlock(qs[i].data(), 4, quat_param);
        problem.AddParameterBlock(ts[i].data(), 3);
        // 可选择固定第一个位姿作为全局参考
        if (i == 0) {
            problem.SetParameterBlockConstant(qs[i].data());
            problem.SetParameterBlockConstant(ts[i].data());
        }
    }

    // 构建相邻位姿约束（来自localizationPoseMap的VIO观测）
    for (int i = 0; i < N-1; ++i) {
        // 计算相对位姿 i->i+1
        Eigen::Quaterniond qi(qs[i][0], qs[i][1], qs[i][2], qs[i][3]);
        Eigen::Quaterniond qj(qs[i+1][0], qs[i+1][1], qs[i+1][2], qs[i+1][3]);
        Eigen::Vector3d ti(ts[i][0], ts[i][1], ts[i][2]);
        Eigen::Vector3d tj(ts[i+1][0], ts[i+1][1], ts[i+1][2]);
        if (qi.norm() < 1e-6 || qj.norm() < 1e-6) {
            continue; // 跳过无效数据
        }

        Eigen::Quaterniond q_ij = qi.conjugate() * qj;
        Eigen::Vector3d t_ij = qi.conjugate() * (tj - ti);

        if (!q_ij.vec().allFinite() || !t_ij.allFinite()) {
            continue;
        }

        double weight = best_inliers[i];
        // double weight = 1.0 + 9.0 * exp(-best_inlier / 0.05);
        
        ceres::CostFunction* cost = RelativePoseResidual::Create(t_ij, q_ij, weight);
        problem.AddResidualBlock(cost, nullptr, qs[i].data(), ts[i].data(), qs[i+1].data(), ts[i+1].data());
    }

    // 求解
    ceres::Solve(options, &problem, &summary);

    // 将优化后的结果写回 localizationPoseMap
    for (int i = 0; i < N; ++i) {
        double key = keys[i];
        std::vector<double> pose(7);
        pose[0] = ts[i][0]; pose[1] = ts[i][1]; pose[2] = ts[i][2];
        pose[3] = qs[i][0]; pose[4] = qs[i][1]; pose[5] = qs[i][2]; pose[6] = qs[i][3];
        localizationPoseMap[key] = pose;
    }
}

