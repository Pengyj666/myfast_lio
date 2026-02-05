#include "geo_utils/geo_utils.h"

#include <tf/LinearMath/Matrix3x3.h>
#include <tf/LinearMath/Quaternion.h>

#include "common/common_def.h"

using namespace common;

namespace utils {

// return (x0, y0)->(x1, y1)
double get_yaw(double x1, double y1, double x0, double y0) {
  if(std::abs(y1-y0) < k_epsilon)
    return (x1-x0 > 0)? 0 : k_pi;
  if(std::abs(x1-x0) < k_epsilon)
    return (y1-y0 > 0)? k_pi/2.0 : -k_pi/2.0;
  return (std::atan2((y1-y0), (x1-x0)));
}

Eigen::Vector3d GetEulerRPY(const Eigen::Quaterniond &q) {
  double roll, pitch, yaw;
  tf::Matrix3x3(tf::Quaternion(q.x(), q.y(), q.z(), q.w())).getRPY(roll, pitch, yaw);
  return Eigen::Vector3d(roll, pitch, yaw);
}

Eigen::Vector3d GetEulerRPY(const Eigen::Matrix3d &R) {
  double roll, pitch, yaw;
  Eigen::Quaterniond q(R);
  tf::Matrix3x3(tf::Quaternion(q.x(), q.y(), q.z(), q.w())).getRPY(roll, pitch, yaw);
  return Eigen::Vector3d(roll, pitch, yaw);
}

} // namespace utils