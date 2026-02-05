#ifndef ESKF_ESKF_H
#define ESKF_ESKF_H

#include <atomic>
#include <mutex>
#include "common/data_type.h"
#include "common/timed_queue.h"

class Eskf {
 public:
  struct Params {
    double acc_noise = 0.01; 
    double gyro_noise = 0.0001; 
    double acc_bias_noise = 0.000001;
    double gyro_bias_noise = 0.00000001;
    double rpy_update_factor = 0.3;
    double est_yaw_noise = 0.0064;
    double est_yaw_update_factor = 0.5;

    Eigen::Vector3d I_p_Gps = Eigen::Vector3d::Zero();  // Gps Frame 在 IMU Frame 下的位置

    double gravity = -9.81007;
  };
  struct State {
    double timestamp = 0.0;
    Eigen::Vector3d W_p_I = Eigen::Vector3d::Zero();      // IMU Frame 在 World Frame 下的位置
    Eigen::Vector3d W_v_I = Eigen::Vector3d::Zero();      // IMU Frame 在 World Frame 下的速度
    Eigen::Matrix3d W_R_I = Eigen::Matrix3d::Identity();  // IMU Frame 在 World Frame 下的姿态

    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 15, 15> cov;

    common::Data_Imu imu_data;
  };
  
  Eskf();
  ~Eskf();
  
  void ResetFusion();
  bool IsValid() { return initialized_.load(); }

  void SetParams(const Params &params);
  bool SetInitState(const Eigen::Matrix3d &W_R_I, const Eigen::Vector3d &W_p_I, double timestamp);

  // IMU预测和观测更新，状态输出
  bool ProcessImuData(const common::Data_Imu &imu, State &state);
  bool ProcessGpsData(const common::Data_Gnss &gps);
  bool ProcessSpeedData(const common::Data_WheelVel &vel);

 private:
  bool ProcInitCache();
  
  Eigen::Matrix3d GetInit_WRI();
  Eigen::Vector3d GetInit_WpI();
  double GetInit_timestamp();
  
  void Predict(const common::Data_Imu &last_imu, const common::Data_Imu &cur_imu, State &state);

  bool UpdateStateByGpsPosition(const common::Data_Gnss &gps, State &state);
  bool UpdateStateBySpeed(const common::Data_WheelVel &vel, State &state);

  void ComputeJacobianAndResidual(
    const Eigen::Vector3d & G_p_Gps, const State &state, Eigen::Matrix<double, 3, 15> & jacobian,
    Eigen::Vector3d & residual);

  // 误差加名义状态
  void AddDeltaToState(const Eigen::Matrix<double, 15, 1> &delta_x, State &state);

 private:
  Params params_;
  
  Eigen::Vector3d gravity_;
  Eigen::Vector3d I_p_Gps_;   // 外参GPS在IMU坐标系中的位置

  std::mutex init_mutex_;
  Eigen::Matrix3d init_W_R_I_;
  Eigen::Vector3d init_W_p_I_;
  double init_timestamp_;

  std::atomic_bool initialized_;

  std::mutex state_mutex_;
  State state_;
  
 private:
  std::mutex imu_data_q_mutex_;
  utils::TimedQueue<common::Data_Imu> imu_data_q_;

  std::mutex gps_data_q_mutex_;
  utils::TimedQueue<common::Data_Gnss> gps_data_q_;
  
  std::mutex vel_data_q_mutex_;
  utils::TimedQueue<common::Data_WheelVel> vel_data_q_;  
};

#endif  // ESKF_ESKF_H
