#include "eskf/eskf.h"

#include <sstream>

#include "common/log_filters.h"
#include "common/sysutils.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"
#include "common/math_utils.h"

using namespace utils;
using namespace common;

constexpr double kDegreeToRadian = M_PI / 180.;
constexpr double kRadianToDegree = 180. / M_PI;

Eskf::Eskf() { initialized_.store(false); }

Eskf::~Eskf() {}

void Eskf::ResetFusion()
{
  droslog(LogLevel::INFO, "Eskf::ResetFusion() called ++++++");
  initialized_.store(false);
  {
    std::lock_guard<std::mutex> lock(init_mutex_);
    init_timestamp_ = -1.0;
  }
  droslog(LogLevel::INFO, "Eskf::ResetFusion() called ------");
}

void Eskf::SetParams(const Params &params) {
  droslog(LogLevel::INFO, "Eskf::SetParams() called ++++++");
  params_ = params;

  // 重力
  gravity_ << 0.0, 0.0, params_.gravity;
 
  // 外参
  I_p_Gps_ = params_.I_p_Gps;

  droslog(LogLevel::INFO, "Eskf::SetParams() acc_noise = %.6f, gyro_noise = %.6f, acc_bias_noise = %.8f, gyro_bias_noise = %.8f", 
      params_.acc_noise, params_.gyro_noise, params_.acc_bias_noise, params_.gyro_bias_noise);
  droslog(LogLevel::INFO, "Eskf::SetParams() I_p_Gps: %.3f, %.3f, %.3f, gravity: %.3f", 
      I_p_Gps_[0], I_p_Gps_[1], I_p_Gps_[2], gravity_[2]);

  imu_data_q_.reset(64);
  gps_data_q_.reset(32);
  vel_data_q_.reset(64);

  ResetFusion();
  droslog(LogLevel::INFO, "Eskf::SetParams() called ------");
}

bool Eskf::SetInitState(const Eigen::Matrix3d &W_R_I, const Eigen::Vector3d & W_p_I, double timestamp)
{
  {
    std::lock_guard<std::mutex> lock(init_mutex_);
    init_W_R_I_ = W_R_I;
    init_W_p_I_ = W_p_I;
    init_timestamp_ = timestamp;
  }

  auto rpy = GetEulerRPY(W_R_I);
  droslog(LogLevel::INFO, "Eskf::SetInitState(): init_timestamp = %.3f, init_W_p_I= %.3f %.3f %.3f, init_W_rpy_I= %.3f %.3f %.3f", 
      timestamp, W_p_I.x(), W_p_I.y(), W_p_I.z(), rpy[0], rpy[1], rpy[2]);

  return ProcInitCache();
}

bool Eskf::ProcInitCache()
{
  droslog(LogLevel::INFO, "Eskf::ProcInitCache() called ++++++");

  // 处理航向角初始化后的缓存数据
  double init_ts = GetInit_timestamp();
  std::vector<Data_Imu> imu_data_cache;
  std::vector<Data_Gnss> gps_data_cache;
  {
    double next_ts = init_ts - 0.03;
    std::lock_guard<std::mutex> lock(imu_data_q_mutex_);
    int ind = imu_data_q_.findAfter(next_ts);
    if (ind > 0) {
      for (; ind >= 0; ind--) {
        imu_data_cache.push_back(imu_data_q_[ind]);
      }
    }
  }
  {
    double next_ts = init_ts - 0.03;
    std::lock_guard<std::mutex> lock(gps_data_q_mutex_);
    int ind = gps_data_q_.findAfter(next_ts);
    if (ind > 0) {
      for (; ind >= 0; ind--) {
        gps_data_cache.push_back(gps_data_q_[ind]);
      }
    }
  }

  if (imu_data_cache.size() == 0 && gps_data_cache.size() == 0) {
    droslog(LogLevel::ERROR, "Eskf::ProcInitCache() failed, no gps data and imu data");
  }

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.timestamp = GetInit_timestamp();
    state_.W_R_I = GetInit_WRI();
    state_.W_p_I = GetInit_WpI();
    state_.W_v_I.setZero();
    state_.ba.setZero();
    state_.bg.setZero();
    state_.cov.setZero();
    state_.cov.block<3, 3>(0, 0) = 1.0 * Eigen::Matrix3d::Identity();  // position std: 1 m
    state_.cov.block<3, 3>(3, 3) = 0.09 * Eigen::Matrix3d::Identity();  // velocity std: 0.3 m/s
    state_.cov.block<2, 2>(6, 6) = 5. * kDegreeToRadian * 5. * kDegreeToRadian * Eigen::Matrix2d::Identity();   // pitch, roll std: 5 degree.
    state_.cov(8, 8) = 5. * kDegreeToRadian * 5. * kDegreeToRadian;  // yaw std: 5 degree.

    state_.cov.block<3, 3>(9, 9) = 0.01* Eigen::Matrix3d::Identity();  // ba std: 0.1 m/s^2
    state_.cov.block<3, 3>(12, 12) = 0.0001 * Eigen::Matrix3d::Identity();  // bg std: 0.01 rad/s
    state_.imu_data.imu.acc.setZero();
    state_.imu_data.imu.gyro.setZero();
  }
  
  {
    auto rpy = GetEulerRPY(state_.W_R_I);
    droslog(LogLevel::INFO, "Eskf::ProcInitCache() ts=%.3f, W_p_I=(%.3f, %.3f, %.3f), rpy=(%.3f, %.3f, %.3f)", 
        state_.timestamp, state_.W_p_I[0], state_.W_p_I[1], state_.W_p_I[2], rpy[0], rpy[1], rpy[2]);
  }
  droslog(LogLevel::INFO, "Eskf::ProcInitCache() 缓存的数据, imu=%d, gps=%d", 
      (int)imu_data_cache.size(), (int)gps_data_cache.size());

  int i = 0, j = 0;
  while (i < (int)imu_data_cache.size() && j < (int)gps_data_cache.size()) {
    const auto &c_imu = imu_data_cache[i];
    const auto &c_gps = gps_data_cache[j];
    if (c_imu.timestamp <= c_gps.timestamp) {
      Predict(state_.imu_data, c_imu, state_);
      i++;
    } else {
      UpdateStateByGpsPosition(c_gps, state_);
      droslog(LogLevel::INFO, "Eskf::ProcInitCache() UpdateStateByGpsPosition() ts=%.3f, W_p_I=(%.3f, %.3f, %.3f), c_gps.ts=%.3f, gps(xyz)=(%.3f, %.3f, %.3f)", 
          state_.timestamp, state_.W_p_I[0], state_.W_p_I[1], state_.W_p_I[2],
          c_gps.timestamp, c_gps.gnss.enu[0], c_gps.gnss.enu[1], c_gps.gnss.enu[2]);
      j++;
    }
    auto rpy = GetEulerRPY(state_.W_R_I);
    droslog(LogLevel::INFO, "Eskf::ProcInitCache() 缓存更新: ts=%.3f, W_p_I=(%.3f, %.3f, %.3f), rpy=(%.3f, %.3f, %.3f)",
        state_.timestamp, state_.W_p_I[0], state_.W_p_I[1], state_.W_p_I[2], rpy[0], rpy[1], rpy[2]);
  }
  initialized_.store(true);
  
  droslog(LogLevel::INFO, "Eskf::ProcInitCache() called ------");
  return true;
}

// 处理IMU数据
bool Eskf::ProcessImuData(const common::Data_Imu &imu, State &state)
{
  {
    std::lock_guard<std::mutex> lock(imu_data_q_mutex_);
    if (imu_data_q_.size() > 0 && imu.timestamp <= imu_data_q_(0)) {
      if (imu.timestamp != imu_data_q_(0)) {
        droslog(LogLevel::WARN, "Eskf::ProcessImuData() IMU data error, cur_ts=%.3f, last_ts=%.3f", 
            imu.timestamp, imu_data_q_(0));
      }
      return false;
    }
    imu_data_q_.emplace_back(imu, imu.timestamp);
  }

  // 1. InitGRI_timestamp_ < 0.0, 说明航向角还未初始化
  if (GetInit_timestamp() < 0.0 || !initialized_.load()) {
    return false;
  }

  // 预测
  Predict(state_.imu_data, imu, state_);
  state = state_;
  
  // GPS 观测
  {
    static double pre_ts = 0.0;
    std::lock_guard<std::mutex> lock(gps_data_q_mutex_);
    if (gps_data_q_.size() > 0 && pre_ts < gps_data_q_(0)) {
      UpdateStateByGpsPosition(gps_data_q_[0], state_);
      pre_ts = gps_data_q_(0);
    }
  }
  // 速度观测
  {
    static double pre_ts = 0.0;
    std::lock_guard<std::mutex> lock(vel_data_q_mutex_);
    if (vel_data_q_.size() > 0 && pre_ts < vel_data_q_(0)) {
      UpdateStateBySpeed(vel_data_q_[0], state_);
      pre_ts = vel_data_q_(0);
    }
  }
  state = state_;
  
  return true;
}

bool Eskf::ProcessGpsData(const common::Data_Gnss &gps)
{
  {
    std::lock_guard<std::mutex> lock(gps_data_q_mutex_);
    if (gps_data_q_.size() > 0 && gps.timestamp <= gps_data_q_(0)) {
      if (gps.timestamp != gps_data_q_(0)) {
        droslog(LogLevel::WARN, "Eskf::ProcessGpsData() GPS data error, cur_ts=%.3f, last_ts=%.3f", gps.timestamp, gps_data_q_(0));
      }
      return false;
    }
    gps_data_q_.emplace_back(gps, gps.timestamp);
  }

  return true;
}

bool Eskf::ProcessSpeedData(const common::Data_WheelVel &vel)
{
  std::lock_guard<std::mutex> lock(vel_data_q_mutex_);
  if (vel_data_q_.size() > 0 && vel.timestamp <= vel_data_q_(0)) {
    droslog(LogLevel::WARN, "Eskf::ProcessSpeedData() Speed data error, cur_ts=%.3f, last_ts=%.3f", 
        vel.timestamp, vel_data_q_(0));
    return false;
  }
  vel_data_q_.emplace_back(vel, vel.timestamp);
  return true;
}

// 初始化imu加速度噪声,角速度噪声，加速度bias噪声，角速度bias噪声
void Eskf::Predict(const common::Data_Imu &last_imu, const common::Data_Imu &cur_imu, State &state)
{
  // Time
  const double delta_t = cur_imu.timestamp - last_imu.timestamp;
  const double delta_t2 = delta_t * delta_t;
  if (delta_t <= 0.0 || delta_t > 0.2) {
    droslog(LogLevel::WARN, "Eskf::Predict() imu时间异常, cur_imu.ts=%.3f, last_imu.ts=%.3f, dts=%.3f", 
        cur_imu.timestamp, last_imu.timestamp, delta_t);
  }

  // Set last state
  State last_state = state;

  // Acc and gyro.
  const Eigen::Vector3d acc_unbias = 0.5 * (last_imu.imu.acc + cur_imu.imu.acc) - last_state.ba;
  const Eigen::Vector3d gyro_unbias = 0.5 * (last_imu.imu.gyro + cur_imu.imu.gyro) - last_state.bg;

  state.W_p_I = last_state.W_p_I + last_state.W_v_I * delta_t +
                0.5 * (last_state.W_R_I * acc_unbias + gravity_) * delta_t2;
  state.W_v_I = last_state.W_v_I + (last_state.W_R_I * acc_unbias + gravity_) * delta_t;

  Eigen::Vector3d delta_angle_axis = gyro_unbias * delta_t;

  if (delta_angle_axis.norm() > 1e-12) {
    state.W_R_I = last_state.W_R_I *
        Eigen::AngleAxisd(delta_angle_axis.norm(), delta_angle_axis.normalized()).toRotationMatrix();
  }

  // Covariance of the error-state.
  Eigen::Matrix<double, 15, 15> Fx = Eigen::Matrix<double, 15, 15>::Identity();
  Fx.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * delta_t;
  Fx.block<3, 3>(3, 6) = -state.W_R_I * GetSkewMatrix(acc_unbias) * delta_t;
  Fx.block<3, 3>(3, 9) = -state.W_R_I * delta_t;
  if (delta_angle_axis.norm() > 1e-12) {
    Fx.block<3, 3>(6, 6) = Eigen::AngleAxisd(delta_angle_axis.norm(), delta_angle_axis.normalized())
                             .toRotationMatrix()
                             .transpose();
  } else {
    Fx.block<3, 3>(6, 6).setIdentity();
  }
  Fx.block<3, 3>(6, 12)  = - Eigen::Matrix3d::Identity() * delta_t;

  Eigen::Matrix<double, 15, 12> Fi = Eigen::Matrix<double, 15, 12>::Zero();
  Fi.block<12, 12>(3, 0) = Eigen::Matrix<double, 12, 12>::Identity();

  Eigen::Matrix<double, 12, 12> Qi = Eigen::Matrix<double, 12, 12>::Zero();
  Qi.block<3, 3>(0, 0) = delta_t2 * params_.acc_noise * Eigen::Matrix3d::Identity();
  Qi.block<3, 3>(3, 3) = delta_t2 * params_.gyro_noise * Eigen::Matrix3d::Identity();
  Qi.block<3, 3>(6, 6) = delta_t * params_.acc_bias_noise * Eigen::Matrix3d::Identity();
  Qi.block<3, 3>(9, 9) = delta_t * params_.gyro_bias_noise * Eigen::Matrix3d::Identity();

  state.cov = Fx * last_state.cov * Fx.transpose() + Fi * Qi * Fi.transpose();

  // Time and imu.
  state.timestamp = cur_imu.timestamp;
  state.imu_data = cur_imu;
  return;
}

bool Eskf::UpdateStateByGpsPosition(const common::Data_Gnss &gps, State &state)
{
  if (gps.timestamp + 0.2 < state.timestamp) {
    droslog(LogLevel::WARN, "Eskf::UpdateStateByGpsPosition() 数据异常, gps时间戳小于当前状态时间戳, gps.ts=%.3f, state.ts=%.3f",
        gps.timestamp, state.timestamp);
    return false;
  }

  {
    static SimpleLogFilter log_filter(5000);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::INFO, "Eskf::UpdateStateByGpsPosition() gps.type=%s, gps.ts=%.3f, gps(xyz)=(%.3f, %.3f, %.3f), state.ts=%.3f, state(enu)=(%.3f, %.3f, %.3f)",
          gps.gnss.rtk_type.c_str(), 
          gps.timestamp, gps.gnss.enu[0], gps.gnss.enu[1], gps.gnss.enu[2],
          state.timestamp, state.W_p_I[0], state.W_p_I[1], state.W_p_I[2]);
    }
  }
  Eigen::Matrix<double, 3, 15> H;
  Eigen::Vector3d residual;

  Eigen::Vector3d G_p_Gps = gps.gnss.enu;
  ComputeJacobianAndResidual(G_p_Gps, state, H, residual);

  const Eigen::Matrix3d & V = gps.gnss.cov;

  // EKF.
  const Eigen::MatrixXd & P = state.cov;
  const Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + V).inverse();
  const Eigen::VectorXd delta_x = K * residual;

  // Add delta_x to state.
  AddDeltaToState(delta_x, state);

  if (state.W_p_I.hasNaN() || state.W_R_I.hasNaN()) {
    static SimpleLogFilter log_filter(5000);
    if (log_filter.Output(GetNow_Steady())) {
      std::stringstream ss;
      ss << "K: \n" << K << "\nP: \n" << P << "\nH: \n" << H << "\nV: \n" << V  << "\nresidual: \n" << residual << "\ndelta_x: \n" << delta_x << std::endl;
      droslog(LogLevel::ERROR, "Eskf::UpdateStateByGpsPosition() ESKF计算异常, 出现NaN值, 各数据状态为: %s", ss.str().c_str());
    }
  }

  // Covarance.
  const Eigen::MatrixXd I_KH = Eigen::Matrix<double, 15, 15>::Identity() - K * H;
  state.cov = I_KH * P * I_KH.transpose() + K * V * K.transpose();

  return true;
}

// update by speed
bool Eskf::UpdateStateBySpeed(const common::Data_WheelVel &vel, State &state)
{
  if (vel.timestamp + 0.2 < state.timestamp) {
    return false;
  }

  Eigen::Vector3d residual;
  Eigen::Vector3d W_v_I = state.W_R_I * vel.vel.vel;
  residual = W_v_I - state.W_v_I ;

  //return true;
  Eigen::Matrix<double, 3, 15> H;
  H.setZero();
  H.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

  const Eigen::Matrix3d & V = vel.cov;
  const Eigen::MatrixXd & P = state.cov;
  const Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + V).inverse();
  const Eigen::VectorXd delta_x = K * residual;

  // Add delta_x to state.
  // AddDeltaToState(delta_x, state);
  Eigen::Vector3d delta_vel = delta_x.block<3, 1>(3, 0);
  Eigen::Vector3d delta_ba = delta_x.block<3, 1>(9, 0);
  
  static SimpleLogFilter log_filter(3000);
  if (log_filter.Output(GetNow_Steady())) {
    droslog(LogLevel::INFO, "Eskf::UpdateStateBySpeed() 速度更新: lv=%.3f, W_v_I=%.3f, %.3f, %.3f, state.W_v_I=%.3f, %.3f, %.3f, update_delta_v=%.3f, %.3f, %.3f, state.ba=%.3f, %.3f, %.3f, update_delta_ba=%.3f, %.3f, %.3f",
        vel.vel.vel[0], W_v_I.x(), W_v_I.y(), W_v_I.z(), state.W_v_I.x(), state.W_v_I.y(), state.W_v_I.z(), delta_vel.x(), delta_vel.y(), delta_vel.z(), 
        state.ba.x(), state.ba.y(), state.ba.z(), delta_ba.x(), delta_ba.y(), delta_ba.z());
  }
    
  state.ba += delta_ba;
  state.W_v_I += delta_vel;

  // Covarance
  const Eigen::MatrixXd I_KH = Eigen::Matrix<double, 15, 15>::Identity() - K * H;
  state.cov = I_KH * P * I_KH.transpose() + K * V * K.transpose();

  return true;
}

void Eskf::AddDeltaToState(const Eigen::Matrix<double, 15, 1> & delta_x, State &state)
{
  state.W_p_I += delta_x.block<3, 1>(0, 0);
  state.W_v_I += delta_x.block<3, 1>(3, 0);

  state.ba += delta_x.block<3, 1>(9, 0) * 0.2;
  state.bg += delta_x.block<3, 1>(12, 0);

  double ang_vel = state.imu_data.imu.gyro.norm();
  if (delta_x.block<3, 1>(6, 0).norm() > 1e-12 && ang_vel > 0.001) {
    state.W_R_I *=
      Eigen::AngleAxisd(delta_x.block<3, 1>(6, 0).norm(), delta_x.block<3, 1>(6, 0).normalized())
        .toRotationMatrix();
  }
}

void Eskf::ComputeJacobianAndResidual(
  const Eigen::Vector3d & G_p_Gps, const State & state, Eigen::Matrix<double, 3, 15> & jacobian,
  Eigen::Vector3d & residual)
{
  const Eigen::Vector3d & W_p_I = state.W_p_I;
  const Eigen::Matrix3d & W_R_I = state.W_R_I;

  // Convert wgs84 to ENU frame.
  // Compute residual.
  residual = G_p_Gps - (W_p_I + W_R_I * I_p_Gps_);

  // Compute jacobian.`
  jacobian.setZero();
  jacobian.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();  //
  //jacobian.block<3, 3>(0, 6) = -W_R_I * GetSkewMatrix(I_p_Gps_);
  return;
}

Eigen::Matrix3d Eskf::GetInit_WRI() {
  std::lock_guard<std::mutex> lock(init_mutex_);
  return init_W_R_I_; 
}

Eigen::Vector3d Eskf::GetInit_WpI() {
  std::lock_guard<std::mutex> lock(init_mutex_);
  return init_W_p_I_; 
}

double Eskf::GetInit_timestamp() {
  std::lock_guard<std::mutex> lock(init_mutex_);
  return init_timestamp_; 
}