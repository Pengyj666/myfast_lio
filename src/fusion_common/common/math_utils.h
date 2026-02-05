#ifndef COMMON_MATH_UTILS_H
#define COMMON_MATH_UTILS_H

#include <Eigen/Core>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <cmath>

namespace utils {

constexpr double kDEG2RAD = M_PI / 180.0;  // deg->rad
constexpr double kRAD2DEG = 180.0 / M_PI;  // rad -> deg
constexpr double G_m_s2 = 9.81;            // 重力大小

constexpr size_t kINVALID_ID = std::numeric_limits<size_t>::max();

/// 将角度保持在正负PI以内
inline void KeepAngleInPI(double& angle) {
    while (angle < -M_PI) {
        angle = angle + 2 * M_PI;
    }
    while (angle > M_PI) {
        angle = angle - 2 * M_PI;
    }
}

/// 将角度保持在正负PI以内
inline double KeepAngleInPI(const double& _angle) {
    double angle = _angle;
    while (angle < -M_PI) {
        angle = angle + 2 * M_PI;
    }
    while (angle > M_PI) {
        angle = angle - 2 * M_PI;
    }
    return angle;
}

/// 将角度保持在正负PI/2以内
inline void KeepAngleInPI2(double& angle) {
    while (angle < -M_PI_2) {
        angle = angle + 2 * M_PI_2;
    }
    while (angle > M_PI_2) {
        angle = angle - 2 * M_PI_2;
    }
}

/// 将角度保持在正负PI/2以内
inline double KeepAngleInPI2(const double& _angle) {
    double angle = _angle;
    while (angle < -M_PI_2) {
        angle = angle + 2 * M_PI_2;
    }
    while (angle > M_PI_2) {
        angle = angle - 2 * M_PI_2;
    }
    return angle;
}

template <typename T>
T rad2deg(const T& radians) {
    return radians * 180.0 / M_PI;
}

template <typename T>
T deg2rad(const T& degrees) {
    return degrees * M_PI / 180.0;
}

template <typename T, typename T2>
void limit_in_range(T&& num, T2&& min_limit, T2&& max_limit) {
    if (num < min_limit) {
        num = min_limit;
    }
    if (num >= max_limit) {
        num = max_limit;
    }
}

inline Eigen::Matrix3d GetSkewMatrix(const Eigen::Vector3d & v)
{
  Eigen::Matrix3d w;
  w << 0., -v(2), v(1), v(2), 0., -v(0), -v(1), v(0), 0.;

  return w;
}

}  // namespace utils

#endif  // COMMON_MATH_UTILS_H
