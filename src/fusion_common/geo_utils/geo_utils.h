#include <Eigen/Core>
#include <Eigen/Geometry>

namespace utils {

// return (x0, y0)->(x1, y1)
double get_yaw(double x1, double y1, double x0 = 0.0, double y0 = 0.0);

// [0]-r, [1]-p, [2]-y
Eigen::Vector3d GetEulerRPY(const Eigen::Quaterniond &q);
// [0]-r, [1]-p, [2]-y
Eigen::Vector3d GetEulerRPY(const Eigen::Matrix3d &R);

} // namespace utils