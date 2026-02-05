#include <Eigen/Core>
#include <Eigen/Geometry>

#include "common/data_type.h"

namespace utils {

// JOHN_NOTE 位姿转换相关
// 这里设置各个模块之间的偏移量，供位姿转换使用
// 主要两个转换基准
// 1. 各模块转换到GPS中心: Gps
// 2. 各模块转换到运动中心(机器后轮轴中心): Base
class TFHelper {
 public:
  static TFHelper* Instance() {
    static TFHelper ins;
    return &ins;
  }
  ~TFHelper() {}

  // Gps 在 Base 中的位姿, meter, rad
  void SetParams_Gps2Base(double x, double y, double z, double roll, double pitch, double yaw);
  // Lidar 在 Base 中的位姿, meter, rad
  void SetParams_Lidar2Base(double x, double y, double z, double roll, double pitch, double yaw);
  // Lidar 在 Gps 中的位姿, meter, rad
  void SetParams_Lidar2Gps(double x, double y, double z, double roll, double pitch, double yaw);
  
  // Vio 在 Gps 中的位姿, meter, rad
  void SetParams_Vio2Gps(double x, double y, double z, double roll, double pitch, double yaw);
  // Imu 在 Gps 中的位姿, meter, rad
  void SetParams_Imu2Gps(double x, double y, double z, double roll, double pitch, double yaw);
  // Vio.Imu 在 Gps 中的位姿, meter, rad
  void SetParams_VioImu2Gps(double x, double y, double z, double roll, double pitch, double yaw);

  common::Pose TF_Vio2Gps(const Eigen::Vector3d& t, const Eigen::Quaterniond& q);
  common::Pose TF_Gps2Vio(const Eigen::Vector3d& t, const Eigen::Quaterniond& q);
  common::Pose TF_Imu2Gps(const Eigen::Vector3d& t, const Eigen::Quaterniond& q);
  common::Pose TF_Lidar2Gps(const Eigen::Vector3d& t, const Eigen::Quaterniond& q);
  
  common::Pose TF_Gps2Base(const Eigen::Vector3d& t, const Eigen::Quaterniond& q);
  common::Pose TF_Lidar2Base(const Eigen::Vector3d& t, const Eigen::Quaterniond& q);

  Eigen::Vector3d Vio2Gps_t() const { return Vio2Gps_t_; }
  Eigen::Vector3d Lidar2Gps_t() const { return Lidar2Gps_t_; }
  Eigen::Vector3d Gps2Base_t() const { return Gps2Base_t_; }
  Eigen::Vector3d Lidar2Base_t() const { return Lidar2Base_t_; }

 private:
  TFHelper() {}
  TFHelper(const TFHelper&) = delete;
  TFHelper& operator=(const TFHelper&) = delete;

  // ------ to Base
  
  Eigen::Vector3d Gps2Base_t_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond Gps2Base_q_ = Eigen::Quaterniond::Identity();

  Eigen::Vector3d Vio2Base_t_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond Vio2Base_q_ = Eigen::Quaterniond::Identity();
  
  Eigen::Vector3d Lidar2Base_t_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond Lidar2Base_q_ = Eigen::Quaterniond::Identity();
  
  // ------ to Gps

  Eigen::Vector3d Base2Gps_t_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond Base2Gps_q_ = Eigen::Quaterniond::Identity();

  Eigen::Vector3d Vio2Gps_t_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond Vio2Gps_q_ = Eigen::Quaterniond::Identity();

  Eigen::Vector3d Lidar2Gps_t_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond Lidar2Gps_q_ = Eigen::Quaterniond::Identity();
  
  Eigen::Vector3d Imu2Gps_t_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond Imu2Gps_q_ = Eigen::Quaterniond::Identity();
  
  Eigen::Vector3d VioImu2Gps_t_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond VioImu2Gps_q_ = Eigen::Quaterniond::Identity();
};

} // namespace utils