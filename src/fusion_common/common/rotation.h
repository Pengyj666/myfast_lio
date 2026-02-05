#ifndef UTILS_COMMON_ROTATION_H_
#define UTILS_COMMON_ROTATION_H_

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace common {

// 欧拉角->旋转矩阵
// 旋转顺序: 内旋转(每次基于载体自身坐标系)，Yaw[-pi, pi]->Pitch[-pi/2, pi/2]->Roll[-pi, pi]
// return Rm = R(X-roll) * R(Y-pitch) * R(Z-yaw)
// c1 = cos(r), s1 = sin(r)
// c2 = cos(p), s2 = sin(p)
// c3 = cos(y), s3 = sin(y)
// RotMat = | c2*c3 -c1*s3+s1*s2*c3  s1*s3+c1*s2*c3 |
//          | c2*s3  c1*c3+s1*s2*s3 -s1*c3+c1*s2*s3 |
//          | -s2    s1*c2           c1*c2          |
template<typename T>
Eigen::Matrix<T, 3, 3> EularAng2RotMat(const T &roll, const T &pitch, const T &yaw) {
  T c1 = std::cos(roll),  s1 = std::sin(roll);
  T c2 = std::cos(pitch), s2 = std::sin(pitch);
  T c3 = std::cos(yaw),   s3 = std::sin(yaw);

  Eigen::Matrix<T, 3, 3> ret;
  ret <<  c2*c3, -c1*s3+s1*s2*c3,  s1*s3+c1*s2*c3,
          c2*s3,	c1*c3+s1*s2*s3, -s1*c3+c1*s2*s3,
         -s2   ,  s1*c2         ,	c1*c2         ;
  return ret;
}

// rpy[0]-roll, rpy[1]-pitch, rpy[2]-yaw
template<typename T>
Eigen::Matrix<T, 3, 3> EularAng2RotMat(const Eigen::Matrix<T, 3, 1> &rpy) {
  return EularAng2RotMat(rpy[0], rpy[1], rpy[2]);
}

// 旋转矩阵->欧拉角
// 旋转顺序: 内旋转(每次基于载体自身坐标系)，Yaw[-pi, pi]->Pitch[-pi/2, pi/2]->Roll[-pi, pi]
// return RotMat = R(X-roll) * R(Y-pitch) * R(Z-yaw)
// c1 = cos(r), s1 = sin(r)
// c2 = cos(p), s2 = sin(p)
// c3 = cos(y), s3 = sin(y)
// RotMat = | c2*c3 -c1*s3+s1*s2*c3  s1*s3+c1*s2*c3 |
//          | c2*s3  c1*c3+s1*s2*s3 -s1*c3+c1*s2*s3 |
//          | -s2    s1*c2           c1*c2          |
// ret[0]-roll, ret[1]-pitch, ret[2]-yaw
template<typename T>
Eigen::Matrix<T, 3, 1> RotMat2EularAng(const Eigen::Matrix<T, 3, 3> &Rm) {
  // 参考: 严恭敏P246
  Eigen::Matrix<T, 3, 1> ret;
  ret[1] = std::asin(-Rm(2,0));   // asin() return [-pi/2, pi/2], note: abs(Rm(2,0)) must <= 1.0
  if (std::abs(Rm(2,0)) > T(0.999999)) {
    // abs(pitch) ~= pi/2, LOCK, assume yaw=0.0
    // s2 ~= -1: sin(roll-yaw) ~=  Rm(0,1), cos(roll-yaw) ~=  Rm(0,2)
    // s2 ~= 1 : sin(roll+yaw) ~= -Rm(0,1), cos(roll+yaw) ~= -Rm(0,2)
    ret[0] = std::atan2(Rm(0,1), Rm(0,2));
    ret[2] = 0.0;
  } else {
    ret[0] = std::atan2(Rm(2,1), Rm(2,2));
    ret[2] = std::atan2(Rm(1,0), Rm(0,0));
  }
  return ret;
}

} // namespace common

#endif//UTILS_COMMON_ROTATION_H_