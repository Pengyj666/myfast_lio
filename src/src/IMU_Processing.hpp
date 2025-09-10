#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "use-ikfom.hpp"
#include "preprocess.h"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  Eigen::Matrix<double, 12, 12> Q;
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double first_lidar_time;
  int lidar_type;

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_;
  sensor_msgs::ImuConstPtr last_imu_;
  deque<sensor_msgs::ImuConstPtr> v_imu_;
  vector<Pose6D> IMUpose;
  vector<M3D>    v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double start_timestamp_;
  double last_lidar_end_time_;
  int    init_iter_num = 1;
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;
  Q = process_noise_cov();
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}

/**
 * @brief 重置IMU处理模块的状态
 * 
 * 该函数将IMU处理模块的所有状态变量重置为初始值，包括清除存储的IMU数据、
 * 重置初始姿态估计、清空点云数据等操作，为下一次IMU处理做好准备。
 * 
 * @param 无
 * @return 无
 */
void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  
  // 重置IMU的平均加速度和角速度测量值
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  
  // 重置上一时刻的角速度和初始化标志
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  
  // 重置时间戳和初始化迭代次数
  start_timestamp_  = -1;
  init_iter_num     = 1;
  
  // 清空存储的IMU数据和位姿信息
  v_imu_.clear();
  IMUpose.clear();
  
  // 重置IMU消息和点云数据指针
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

/**
 * @brief 初始化IMU相关参数，包括重力方向、陀螺仪偏置、加速度和角速度的协方差等。
 *        同时对加速度测量值进行归一化处理，使其符合单位重力加速度方向。
 *
 * @param meas 包含IMU和激光雷达时间戳等信息的测量组
 * @param kf_state 扩展卡尔曼滤波器状态对象，用于初始化状态和协方差
 * @param N 用于统计IMU数据数量，作为均值与协方差计算中的计数器
 */
void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/

  V3D cur_acc, cur_gyr;

  // 如果是第一帧数据，则进行初始化操作
  if (b_first_frame_)
  {
    Reset(); // 重置IMU处理状态
    N = 1;   // 初始化计数器
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
  }

  // 遍历所有IMU数据，累计计算加速度和角速度的均值与协方差
  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    // 增量更新均值
    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    // 增量更新协方差
    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    N ++;
  }

  // 初始化滤波器状态：设置重力方向、陀螺仪偏置、LiDAR与IMU之间的外参等
  state_ikfom init_state = kf_state.get_x();
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2); // 将加速度归一化为重力方向

  init_state.bg  = mean_gyr; // 设置陀螺仪偏置为角速度均值
  init_state.offset_T_L_I = Lidar_T_wrt_IMU; // 设置LiDAR相对于IMU的平移外参
  init_state.offset_R_L_I = Lidar_R_wrt_IMU; // 设置LiDAR相对于IMU的旋转外参
  kf_state.change_x(init_state);

  // 初始化误差协方差矩阵P，设置各状态量的初始协方差
  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity(); // 初始化为单位矩阵

  // 设置特定状态的协方差值
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;  // 位置噪声
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001; // 姿态噪声
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001; // 速度噪声
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;  // 加速度偏置噪声
  init_P(21,21) = init_P(22,22) = 0.00001; // 角速度偏置噪声

  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back(); // 记录最后一个IMU数据
}

/**
 * @brief 对激光雷达点云进行去畸变处理，利用IMU数据对点云中每个点的时间戳进行运动补偿
 * 
 * 该函数通过将IMU测量值与激光雷达点云时间对齐，使用扩展卡尔曼滤波器（EKF）估计传感器在各时刻的姿态，
 * 并对点云中的每个点进行逆向运动补偿，从而消除由于传感器运动引起的点云畸变。
 *
 * @param meas 包含IMU和激光雷达数据的测量组，其中IMU用于预测姿态，激光雷达用于去畸变
 * @param kf_state 扩展卡尔曼滤波器状态对象，用于前向传播和获取当前状态
 * @param pcl_out 输出的去畸变后的点云数据
 */
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** 将上一帧末尾的IMU数据添加到当前帧头部，以保证时间连续性 ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();

  double pcl_beg_time = meas.lidar_beg_time;
  double pcl_end_time = meas.lidar_end_time;

  /*** 特殊处理MARSIM类型激光雷达的时间设置 ***/
  if (lidar_type == MARSIM) {
      pcl_beg_time = last_lidar_end_time_;
      pcl_end_time = meas.lidar_beg_time;
  }

  /*** 按照时间偏移量对点云进行排序 ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);

  /*** 初始化IMU位姿存储容器并插入初始状态 ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** 遍历IMU数据进行前向传播，计算每个IMU时刻的状态 ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;

  input_ikfom in;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    
    /*** 跳过早于上次激光结束时间的IMU数据 ***/
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    /*** 计算角速度和加速度平均值 ***/
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    /*** 加速度归一化处理 ***/
    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    /*** 根据时间戳确定传播时间间隔 ***/
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    /*** 设置输入并执行EKF预测步骤 ***/
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    /*前3行3列：陀螺仪协方差
    3-6行3-6列：加速度计协方差
    6-9行6-9列：陀螺仪偏置协方差
    9-12行9-12列：加速度计偏置协方差*/
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    kf_state.predict(dt, Q, in);

    /*** 保存当前IMU时刻的状态信息 ***/
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** 计算帧结束时刻的姿态预测值 ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);
  
  imu_state = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  /*** 对点云中每个点进行逆向运动补偿（去畸变） ***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return;

  if(lidar_type != MARSIM){
      auto it_pcl = pcl_out.points.end() - 1;
      for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
      {
          auto head = it_kp - 1;
          auto tail = it_kp;
          R_imu<<MAT_FROM_ARRAY(head->rot);
          vel_imu<<VEC_FROM_ARRAY(head->vel);
          pos_imu<<VEC_FROM_ARRAY(head->pos);
          acc_imu<<VEC_FROM_ARRAY(tail->acc);
          angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

          /*** 遍历点云中时间大于当前IMU时间的点，进行补偿 ***/
          for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
          {
              dt = it_pcl->curvature / double(1000) - head->offset_time;

              /*** 构造点的变换矩阵并进行逆向补偿 ***/
              M3D R_i(R_imu * Exp(angvel_avr, dt));

              V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
              V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
              V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!

              /*** 更新点坐标为补偿后的位置 ***/
              it_pcl->x = P_compensate(0);
              it_pcl->y = P_compensate(1);
              it_pcl->z = P_compensate(2);

              if (it_pcl == pcl_out.points.begin()) break;
          }
      }
  }
}

/**
 * @brief 处理IMU和激光雷达数据，进行初始对准和点云去畸变
 * 
 * 该函数主要完成以下任务：
 * 1. 如果是初始阶段，则进行IMU初始化处理；
 * 2. 如果已完成初始化，则使用IMU数据对激光雷达点云进行运动畸变矫正；
 * 
 * @param[in] meas 包含IMU和激光雷达数据的测量组
 * @param[in,out] kf_state 扩展卡尔曼滤波状态，用于存储和更新系统状态
 * @param[out] cur_pcl_un_ 去畸变后的当前帧点云指针
 */
void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  // 检查是否有IMU数据，如果没有则直接返回
  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  // IMU初始化处理阶段
  if (imu_need_init_)
  {
    /// 执行IMU初始化操作，估计重力方向、陀螺仪偏置等初始状态
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    // 保存最新的IMU数据
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    // 判断初始化是否完成
    if (init_iter_num > MAX_INI_COUNT)
    {
      // 根据重力大小调整加速度计协方差
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      // 恢复协方差设置
      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      //ROS_INFO("IMU Initial Done");
      ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
                imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
      //cout << "Debug: File=" << __FILE__ << ", Line=" << __LINE__ << ", Function=" << __FUNCTION__ << endl;
    }

    return;
  }

  // 使用IMU数据对激光雷达点云进行去畸变处理
  UndistortPcl(meas, kf_state, *cur_pcl_un_);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
