#ifndef _LASERMAPPING_HELP_H
#define _LASERMAPPING_HELP_H


#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <nav_msgs/Path.h>

#include "IMU_Processing.h"
#include "ikd-Tree/ikd_Tree.h"

// using namespace std;

/**
 * @brief 初始化时间常量定义
 * 
 * 该宏定义用于设置系统的初始时间值，单位为秒
 */
#define INIT_TIME           (0.1)

/**
 * @brief 激光点协方差常量定义
 * 
 * 该宏定义用于设置激光雷达点云数据的协方差值，
 * 用于表示激光点的测量不确定性
 */
#define LASER_POINT_COV     (0.001)

/**
 * @brief 最大数量限制常量定义
 * 
 * 该宏定义用于设置系统中可处理的最大数据点数量，
 * 用于内存分配和数组大小限制   
 */
#define MAXN                (720000)

/**
 * @brief 发布帧周期常量定义
 * 
 * 该宏定义用于设置数据发布的时间周期，单位为秒，
 * 控制数据发布的频率
 */
#define PUBFRAME_PERIOD     (20)

/**
 * @brief 激光点数量限制常量定义
 * 
 * 该宏定义用于设置激光点数量限制，用于控制数据处理中的点数量限制 多少个激光点同时处理
 */

#define NUM_SCAN 3

// 外部声明时间日志变量
extern double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
extern bool   runtime_pos_log, pcd_save_en, time_sync_en, extrinsic_est_en, path_en;

// 外部声明残差相关变量
extern float res_last[100000];
extern float DET_RANGE;
extern double time_diff_lidar_to_imu;

// 外部声明线程同步变量
extern condition_variable sig_buffer;

// 外部声明点云相关变量
extern PointCloudXYZI::Ptr accumulated_cloud;
extern mutex accumulated_cloud_mutex;
extern string root_dir;
extern string map_file_path, lid_topic, imu_topic;

// 外部声明参数和状态变量
extern double res_mean_last, total_residual;
extern double last_timestamp_lidar, last_timestamp_imu;
extern double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
extern double filter_size_corner_min, filter_size_surf_min, filter_size_map_min, fov_deg;
extern double cube_len, HALF_FOV_COS, FOV_DEG, total_distance, lidar_end_time, first_lidar_time;
extern int    effct_feat_num, time_log_counter, scan_count, publish_count;
extern int    iterCount, feats_down_size, NUM_MAX_ITERATIONS, laserCloudValidNum, pcd_save_interval, pcd_index;
extern int    txt_save_interval;
extern bool   point_selected_surf[100000];
extern bool   flg_exit;
extern int    scan_num;
extern double lidar_mean_scantime;
extern bool lidar_pushed ,flg_EKF_inited, flg_first_scan;
extern MeasureGroup Measures;
extern bool Localmap_Initialized ;
// 外部声明容器变量
extern vector<vector<int>>  pointSearchInd_surf;
extern vector<BoxPointType> cub_needrm;
extern vector<PointVector>  Nearest_Points;
extern vector<double>       extrinT;
extern vector<double>       extrinR;
extern deque<double>                     time_buffer;
extern deque<PointCloudXYZI::Ptr>        lidar_buffer;
extern deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

// 外部声明地图保存相关变量
extern std::atomic<bool> save_map;

extern KD_TREE<PointType> ikdtree;

// 外部声明点云指针变量
extern PointCloudXYZI::Ptr featsFromMap;
extern PointCloudXYZI::Ptr feats_undistort;
extern PointCloudXYZI::Ptr feats_down_body;
extern PointCloudXYZI::Ptr normvec;
extern PointCloudXYZI::Ptr laserCloudOri;
extern PointCloudXYZI::Ptr corr_normvect;
extern PointCloudXYZI::Ptr _featsArray;
extern PointCloudXYZI::Ptr down_map;
// extern PointCloudXYZI::Ptr pcl_wait_pub;
extern PointCloudXYZI::Ptr pcl_wait_save;

// 外部声明坐标点变量
extern V3D euler_cur;
extern V3D Lidar_T_wrt_IMU;
extern M3D Lidar_R_wrt_IMU;

/*** EKF inputs and output ***/
// 扩展卡尔曼滤波器实例，模板参数为状态类型、噪声维度和输入类型
extern esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
// EKF状态点，存储当前估计的状态信息
extern state_ikfom state_point;
// 激光雷达位置向量
extern vect3 pos_lid;

// 外部声明处理模块智能指针
extern shared_ptr<Preprocess> p_pre;
extern shared_ptr<ImuProcess> p_imu;

void init_imu_extrin();
void SigHandle(int sig);

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);
void lasermap_fov_segment();
bool sync_packages();
void dump_lio_state_to_log(FILE *fp ); 
void map_incremental(); 

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po);
void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po);
void points_cache_collect();
void pointBodyToWorld(PointType const * const pi, PointType * const po );
void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s);

void init_pos_log();
void close_pos_log();
void save_runtime_pos_log();
void debug_runtime_pos_log(double solve_H_time);
bool init_kdtree();
void save_PCL_Storage();
bool  imu_pretreatment();
void resizePointCloud();

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}




#endif