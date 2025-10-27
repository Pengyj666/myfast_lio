#ifndef GLOBALOPTIMIZATION_H
#define GLOBALOPTIMIZATION_H 

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <thread>
#include <chrono>
#include <mutex>
#include <map>
#include <vector>
#include "ceres/ceres.h"

extern std::vector<double> timefitnessstamps;

// PolynomialResidual 结构体完整定义
struct PolynomialResidual {
    PolynomialResidual(double t, double observed_value, int degree)
        : t_(t), observed_value_(observed_value), degree_(degree) {}
    
    template <typename T>
    bool operator()(const T* const coeffs, T* residual) const {
        // 计算多项式值
        T predicted_value = T(0.0);
        T t_pow = T(1.0);
        
        for (int i = 0; i <= degree_; ++i) {
            predicted_value += coeffs[i] * t_pow;
            t_pow *= T(t_);
        }
        
        // 残差 = 预测值 - 观测值
        residual[0] = predicted_value - T(observed_value_);
        return true;
    }
    
private:
    double t_;
    double observed_value_;
    int degree_;
};

class GlobalOptimization
{
public:    
    std::map<double, std::vector<double>> localizationPoseMap;
    std::vector<double> best_inliers;
public:
    GlobalOptimization();
    ~GlobalOptimization();
    void optimize();

protected:


};

#endif