// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include <std_msgs/Bool.h>
#include "preprocess.h"
#include <fstream>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include "ikd-Tree/ikd_Tree.h"
// #include "fast_lio_localization.hpp" 

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

#define localization 0

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = true, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;

/**************************/

float res_last[100000] = {0.0};
// 检测范围定义，用于确定物体检测的最大距离
float DET_RANGE = 50.0f;

// 移动阈值常量，用于判断物体是否发生移动的最小距离标准
const float MOV_THRESHOLD = 0.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

mutex txt_save_mutex;
PointCloudXYZI::Ptr accumulated_cloud(new PointCloudXYZI());

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0 , num_scan = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
int    txt_save_interval = 2; // 保存txt文件的间隔
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = true, dense_pub_en = false, scan_body_pub_en = true;
int lidar_type;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;

bool reduce_framerate = false;   // 是否降低帧率的标志变量
deque<double>                     temp_time_buffer;
deque<PointCloudXYZI::Ptr>        temp_lidar_buffer;


deque<sensor_msgs::Imu::ConstPtr> imu_buffer;


bool   save_map = false; // 是否保存地图的标志变量
ofstream save_map_file; // 用于保存地图数据的文件流对象

bool init_flag = false;
Eigen::Matrix4f pose_tf = Eigen::Matrix4f::Identity();

/**
 * @brief 从地图中提取的特征点云
 * 
 * 该点云存储从地图中提取的原始特征点，包含XYZ坐标和强度信息
 */
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());

/**
 * @brief 去畸变后的特征点云
 * 
 * 该点云存储经过运动畸变校正后的特征点，用于提高配准精度
 */
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());

/**
 * @brief 降采样后的机体坐标系特征点云
 * 
 * 该点云存储在机体坐标系下经过降采样处理的特征点，减少计算量
 */
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());

/**
 * @brief 降采样后的世界坐标系特征点云
 * 
 * 该点云存储转换到世界坐标系并降采样后的特征点，用于全局匹配
 */
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());

/**
 * @brief 法向量点云
 * 
 * 该点云存储特征点的法向量信息，容量预分配为100000个点
 * 用于点云配准中的几何约束计算
 */
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));

/**
 * @brief 原始激光点云
 * 
 * 该点云存储原始的激光雷达数据点，容量预分配为100000个点
 * 用于后续的特征提取和处理
 */
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));

/**
 * @brief 对应法向量点云
 * 
 * 该点云存储与特征点对应的法向量信息，容量预分配为100000个点
 * 用于优化算法中的约束条件计算
 */
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));

/**
 * @brief 特征点云数组
 * 
 * 该指针用于存储特征点云的数组结构，具体分配在后续代码中进行
 */
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

// ikd-Tree树用于增量最近邻搜索
KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
// 扩展卡尔曼滤波器的测量数据组，用于存储传感器测量信息
MeasureGroup Measures;
// 扩展卡尔曼滤波器实例，模板参数为状态类型、噪声维度和输入类型
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
// EKF状态点，存储当前估计的状态信息
state_ikfom state_point;
// 激光雷达位置向量
vect3 pos_lid;

// 轨迹路径消息，用于发布机器人的运动轨迹
nav_msgs::Path path;
// 里程计消息，存储滤波后的位置和姿态信息
nav_msgs::Odometry odomAftMapped;
// 四元数消息，用于表示机器人姿态
geometry_msgs::Quaternion geoQuat;
// 位姿消息，包含机器人在body坐标系下的位姿信息
geometry_msgs::PoseStamped msg_body_pose;

// 点云预处理模块的智能指针
shared_ptr<Preprocess> p_pre(new Preprocess());
// IMU数据处理模块的智能指针
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

/**
 * @brief 将LIO状态信息写入日志文件
 * 
 * 该函数将当前LIO系统的状态信息格式化输出到指定的日志文件中，
 * 包括时间戳、旋转角度、位置、角速度、线速度、加速度、陀螺仪偏置、
 * 加速度计偏置和重力向量等信息。
 * 
 * @param fp 指向日志文件的文件指针，用于写入状态信息
 */
inline void dump_lio_state_to_log(FILE *fp)  
{
    // 将旋转矩阵转换为欧拉角形式表示的旋转角度
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    
    // 写入相对于首次激光雷达时间的时间戳
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    
    // 写入旋转向量（欧拉角）
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));
    
    // 写入位置坐标
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2));
    
    // 写入角速度（此处默认为0）
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);
    
    // 写入线速度
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2));
    
    // 写入加速度（此处默认为0）
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);
    
    // 写入陀螺仪偏置
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));
    
    // 写入加速度计偏置
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));
    
    // 写入重力向量
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]);
    
    // 写入换行符并刷新文件缓冲区
    fprintf(fp, "\r\n");
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    // p_body是激光雷达坐标系 -IMU坐标系 - 世界坐标系 转换到世界坐标系
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

/**
 * @brief 收集点云缓存数据
 * 
 * 该函数用于从ikdtree中获取被移除的点云数据，并将其存储到历史记录中。
 * 可以用于后续的点云数据处理或分析。
 * 
 * @note 该函数不接受任何参数，无返回值
 */
void points_cache_collect()
{
    PointVector points_history;
    // 从ikdtree中获取被移除的点云数据
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
/**
 * @brief 根据激光雷达的视场角（FOV）对局部地图进行分段管理，动态更新局部地图边界。
 * 
 * 此函数用于维护一个随传感器位置变化而动态调整的局部地图区域。当传感器靠近当前局部地图边界时，
 * 会根据设定的阈值移动局部地图区域，并删除旧区域中的点云数据以保持地图更新。
 * 
 * 主要功能包括：
 * - 初始化局部地图区域；
 * - 判断是否需要移动局部地图；
 * - 计算新的局部地图边界；
 * - 删除旧区域中的点云数据。
 * 
 * @note 该函数不接受参数，也不返回任何值。
 */
void lasermap_fov_segment()
{
    // 清空需要删除的立方体列表
    cub_needrm.clear();
    
    // 重置KD树删除计数器和时间统计
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    

    // 将X轴方向的点从机体坐标系转换到世界坐标系
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);

    // 获取当前激光雷达的位置
    V3D pos_LiD = pos_lid;

    // 如果局部地图尚未初始化，则初始化局部地图边界并返回
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    // 计算当前位置到局部地图各边的距离，并判断是否需要移动局部地图
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }

    // 如果不需要移动局部地图，则直接返回
    if (!need_move) return;

    // 定义新的局部地图边界和临时边界变量
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;

    // 计算移动距离
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
 cout << "mov_dist " << mov_dist << endl;
    // 根据距离判断是否需要移动局部地图的各个边界，并记录需要删除的区域
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }

    // 更新局部地图边界
    LocalMap_Points = New_LocalMap_Points;

    // 收集需要缓存的点云数据
    //points_cache_collect();

    // 删除旧区域中的点云数据，并统计删除时间和删除点数
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

/**
 * @brief 标准点云回调函数，处理来自激光雷达的点云数据
 * @param msg 输入的点云数据消息指针
 * 
 * 该函数负责接收激光雷达点云数据，进行预处理并存储到缓冲区中。
 * 主要功能包括：时间戳检查、点云预处理、数据缓冲管理等。
 */
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
   
    // assert(msg->height == 1);
    // 加锁保护共享数据缓冲区
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    
    // 检查时间戳是否出现回退，如果回退则清空缓冲区
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    // cout << "Debug: File=" << __FILE__ << ", Line=" << __LINE__ << ", Function=" << __FUNCTION__ << endl;
    // 创建新的点云对象并进行预处理
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    // 将处理后的点云数据和时间戳存入缓冲区
    if(reduce_framerate){
        if(num_scan == NUM_SCAN){
            // 将临时缓冲区的数据追加到主缓冲区
            lidar_buffer.insert(lidar_buffer.end(), temp_lidar_buffer.begin(), temp_lidar_buffer.end());
            time_buffer.insert(time_buffer.end(), temp_time_buffer.begin(), temp_time_buffer.end());
            
            // 清空临时缓冲区
            temp_lidar_buffer.clear();
            temp_time_buffer.clear();
            num_scan =0;
        } else {
            temp_lidar_buffer.push_back(ptr);
            temp_time_buffer.push_back(msg->header.stamp.toSec());
            num_scan ++;
        }
    }else{
        lidar_buffer.push_back(ptr);
        time_buffer.push_back(msg->header.stamp.toSec());
    }
    
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    // 记录预处理耗时
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    // cout<<"s_plot11[scan_count]: "<<s_plot11[scan_count]<<endl;
    // 解锁并通知等待的线程
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}


double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
/**
 * @brief 处理Livox激光雷达点云数据的回调函数
 * @param msg Livox自定义消息的常量指针，包含激光雷达点云数据
 * 
 * 该函数主要功能包括：
 * 1. 时间戳检查和同步处理
 * 2. IMU与LiDAR数据的时间同步检测
 * 3. 点云数据预处理
 * 4. 将处理后的数据存入缓冲区
 * 
 * 函数内部维护了时间戳、缓冲区等状态信息，确保数据处理的时序正确性
 */

void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    // 加锁保护共享缓冲区
    mtx_buffer.lock();
    
    // 记录预处理开始时间，用于性能统计
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    
    // 检查时间戳是否回退，如果回退则清空激光雷达缓冲区
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    // 检查IMU和LiDAR数据是否同步（在未启用时间同步时）
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    // 自动时间同步：计算LiDAR相对于IMU的时间差
    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    // 创建点云对象并进行预处理
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    
    // 将处理后的点云数据和时间戳存入缓冲区
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    // 记录预处理耗时用于性能分析
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    
    // 解锁并通知等待的线程
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

/**
 * @brief IMU数据回调函数，用于处理IMU传感器数据并将其存储到缓冲区中
 * @param msg_in IMU传感器数据的消息指针
 * 
 * 该函数主要功能包括：
 * 1. 接收IMU原始数据并进行时间戳调整
 * 2. 检查时间戳有效性，处理时间回退情况
 * 3. 将处理后的IMU数据存入缓冲区供其他模块使用
 * 
 * 时间戳处理逻辑：
 * - 根据激光雷达与IMU的时间差进行时间同步调整
 * - 支持基于时间差阈值的时间同步使能控制
 * 
 * 缓冲区管理：
 * - 使用互斥锁保证线程安全
 * - 检测时间回退并清空缓冲区
 * - 通知等待线程缓冲区状态变化
 */
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    // 调整IMU消息的时间戳，补偿激光雷达与IMU之间的时间差
    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    
    // 如果时间差超过阈值且时间同步使能，则进行额外的时间同步调整
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    // 加锁保护缓冲区操作
    mtx_buffer.lock();

    // 检查时间戳是否出现回退，如果回退则清空IMU缓冲区
    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    // 将处理后的IMU消息添加到缓冲区
    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    
    // 通知所有等待缓冲区的线程
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
/**
 * @brief 同步激光雷达和IMU数据包，确保时间戳对齐
 * 
 * 该函数用于同步激光雷达扫描数据与IMU数据。首先检查激光雷达和IMU缓冲区是否为空，
 * 若不为空，则将激光雷达数据加入测量组，并根据点云数据计算扫描结束时间。
 * 然后从IMU缓冲区中提取时间早于激光雷达结束时间的数据，一并加入测量组。
 * 
 * @param meas 测量组引用，用于存储同步后的激光雷达和IMU数据
 * @return bool 返回true表示成功同步一组数据，返回false表示数据不足或未满足时间条件
 */
bool sync_packages(MeasureGroup &meas)
{
    //std::lock_guard<std::mutex> lock(mtx_buffer);
    
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }
 
    /*** 处理激光雷达数据：将前端的激光雷达扫描数据加入测量组 ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();

        // 根据点云数量和最后一个点的曲率信息估算扫描结束时间
        if (meas.lidar->points.size() <= 1) // 点数太少，使用平均扫描时间
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        // 如果是MARSIM类型激光雷达，扫描结束时间等于开始时间
        if(lidar_type == MARSIM)
            lidar_end_time = meas.lidar_beg_time;

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    // 检查最新的IMU时间戳是否已经覆盖到激光雷达扫描结束时间
    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** 提取并填充IMU数据：从IMU缓冲区取出时间在lidar_end_time之前的所有数据 ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }
    

    // 弹出已处理的激光雷达数据
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
/**
 * @brief 增量式构建地图点云，对输入的特征点进行坐标变换、下采样判断，并将需要添加的点插入到KD树中。
 *
 * 该函数的主要流程包括：
 * 1. 将点从体坐标系（body frame）转换到世界坐标系（world frame）；
 * 2. 判断每个点是否需要加入地图（通过近邻点距离和下采样策略）；
 * 3. 将需要添加的点增量式地插入到ikd-tree中。
 *
 * @note 该函数不返回任何值，但会更新全局变量：ikdtree、add_point_size、kdtree_incremental_time。
 */
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);

    // 遍历所有降采样后的特征点
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));

        /* decide if need add to map  决定是否需要添加到地图*/
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 

            // 计算当前点所在的下采样网格中心点
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;

            float dist  = calc_dist(feats_down_world->points[i],mid_point);

            // 增加距离阈值判断，避免添加过多近邻点
            if (dist > filter_size_map_min * 0.8) {
                PointToAdd.push_back(feats_down_world->points[i]);
                continue;
            }

            // 如果最近邻点与当前网格中心距离过大，则不需要下采样，直接保留
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && 
                fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && 
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }

            // 检查是否有更靠近网格中心的点存在，若有则不添加当前点
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }

            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            // 如果没有近邻点或EKF未初始化，则直接添加该点
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    // 将筛选后的点增量式地添加到KD树中并统计时间
    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, false);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    //add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;

    // if(save_map){
    //     std::lock_guard<std::mutex> lock(txt_save_mutex);
    //     for (const auto& point : PointToAdd) {
    //         accumulated_cloud->push_back(point);
    //     }
    //     for (const auto& point : PointNoNeedDownsample) {
    //         accumulated_cloud->push_back(point);
    //     }
    // }
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
/**
 * @brief 发布激光雷达点云数据到ROS话题，并可选择性地保存为PCD文件
 * 
 * 该函数主要完成以下功能：
 * 1. 如果使能(scan_pub_en为true)，将去畸变或降采样后的点云转换到世界坐标系并发布；
 * 2. 如果使能地图保存(pcd_save_en为true)，则将点云累积并定期保存为PCD文件。
 * 
 * @param[in] pubLaserCloudFull ROS发布者，用于发布转换后的点云消息
 */
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    // 如果允许发布点云数据
    if(scan_pub_en)
    {
        cout << "publish frame at: " << static_cast<long long>(lidar_end_time * 1e6) << " microseconds" << endl;

        // ros::Time current_time = ros::Time::now();
        // ROS_INFO("Current ROS time: %f", current_time.toSec());
        // 根据dense_pub_en标志选择使用去畸变点云还是降采样点云
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        // 创建用于存储世界坐标系下点云的新点云对象
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        // 将每个点从机体坐标系转换到世界坐标系
        if (init_flag) {
            // 使用pose_tf进行坐标变换
            for (int i = 0; i < size; i++) {
                PointType point_body = laserCloudFullRes->points[i];
                
                // 构造齐次坐标
                Eigen::Vector4f point_homo(point_body.x, point_body.y, point_body.z, 1.0);
                
                // 应用变换矩阵
                Eigen::Vector4f point_transformed = pose_tf * point_homo;
                
                // 填充到世界坐标系点云
                laserCloudWorld->points[i].x = point_transformed(0);
                laserCloudWorld->points[i].y = point_transformed(1);
                laserCloudWorld->points[i].z = point_transformed(2);
                laserCloudWorld->points[i].intensity = point_body.intensity;
            }
        } else {
            for (int i = 0; i < size; i++) {
                RGBpointBodyToWorld(&laserCloudFullRes->points[i], 
                                   &laserCloudWorld->points[i]);
            }
        }

        // 构造ROS点云消息并发布
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    // 如果使能PCD文件保存功能
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        // 创建用于存储世界坐标系下点云的新点云对象
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        // 将每个点从机体坐标系转换到世界坐标系
        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        // 将转换后的点云累加到等待保存的点云中
        *pcl_wait_save += *laserCloudWorld;

        // 计数器递增，用于控制保存间隔
        static int scan_wait_num = 0;
        scan_wait_num ++;
        // 当累积点云数量足够且达到保存间隔时，保存点云到PCD文件
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

/**
 * @brief 发布经过IMU坐标系变换的激光点云数据
 * 
 * 该函数将去畸变后的激光点云数据从雷达坐标系转换到IMU坐标系，
 * 然后发布到ROS话题中供其他节点使用。
 * 
 * @param pubLaserCloudFull_body ROS发布者对象，用于发布转换后的点云数据
 */
void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    // 获取去畸变后点云数据的大小
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    // 将雷达坐标系下的点云转换到IMU坐标系下
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    // 将点云数据转换为ROS消息格式并发布
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    
    // 更新发布计数器
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

/**
 * @brief 发布激光点云地图数据到ROS话题
 * 
 * 该函数将内部存储的点云特征数据转换为ROS消息格式，
 * 并发布到指定的Publisher中，用于地图可视化或后续处理
 * 
 * @param pubLaserCloudMap ROS发布者对象，用于发布点云地图数据
 */
void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    // 创建ROS点云消息对象
    sensor_msgs::PointCloud2 laserCloudMap;
    
    // 将PCL点云数据转换为ROS消息格式
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    
    // 设置消息时间戳和坐标系
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    
    // 发布点云地图数据
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

/**
 * @brief 发布里程计信息，并广播对应的TF变换
 * 
 * 该函数用于将处理后的里程计数据发布到ROS系统中，并通过TF广播坐标变换。
 * 里程计的位姿由`set_posestamp`函数设置，同时会从卡尔曼滤波器中获取协方差信息填充到消息中。
 * 最后，使用TF库广播从"camera_init"到"body"的坐标变换。
 *
 * @param[in] pubOdomAftMapped 用于发布里程计消息的ROS发布者对象
 */
void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    // 设置里程计消息的帧ID和时间戳
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);

    // 填充位姿信息
    set_posestamp(odomAftMapped.pose);

    // 发布里程计消息
    pubOdomAftMapped.publish(odomAftMapped);

    // 从卡尔曼滤波器获取协方差矩阵，并重新排列以适配ROS标准格式
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    // 广播TF变换：从camera_init到body
    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;

    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));

    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);

    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

/**
 * @brief 共享模型函数，用于计算激光点与地图中最近表面之间的残差，并构建EKF更新所需的观测矩阵H和观测向量。
 * 
 * 该函数的主要任务包括：
 * 1. 将激光点从体坐标系转换到世界坐标系；
 * 2. 搜索每个点在地图中的最近邻点并估计局部平面；
 * 3. 计算点到平面的距离作为观测残差；
 * 4. 构建观测雅可比矩阵（H）和观测向量（h）供EKF使用。
 *
 * @param s 当前状态估计结构体，包含位姿、偏移等信息。
 * @param ekfom_data EKF共享数据结构，用于传递观测相关数据（如H矩阵、观测值等）。
 */
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** 最近表面搜索与残差计算 **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* 坐标变换：将点从体坐标系转换到世界坐标系 */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** 在地图中查找最近的表面点 **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.05f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** 构建观测雅可比矩阵H和观测向量 ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** 获取最近表面的法向量 ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** 计算观测雅可比矩阵H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** 观测值：点到最近表面的距离 ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

/**
 * @brief 保存点云地图的回调函数，用于在用户按下Ctrl+S键时保存点云地图。
 * 
 * @param msg 保存地图的标志位，true表示保存地图，false表示取消保存地图。
 */
void save_map_cbk(const std_msgs::Bool::ConstPtr &msg){
    string map_output_dir = root_dir + "/Lader_Map";
    mkdir(map_output_dir.c_str(), 0777);


  
    save_map = msg->data;

    if(save_map){
        string map_filename = root_dir + "src/StaticMap/static_map.txt";

        save_map_file.open(map_filename, ios::out | ios::app); // app追加的形式   以覆盖模式打开文件ios::trunc)

        
        if (!save_map_file.is_open()) {
            ROS_ERROR("Cannot open file %s for writing point cloud data", map_filename.c_str());
            return;
        }
    }else{
        save_map_file.close();
    }

}


/**
 * @brief 在主循环中调用的地图保存函数示例
 * 
 */
void exportStaticMapExample() {
    if (save_map_file.is_open()) {
        #if 0
        // 使用字符串流预处理所有数据
        std::lock_guard<std::mutex> lock(txt_save_mutex);
        stringstream ss;
        ss << fixed << setprecision(6);
        
        for (size_t i = 0; i < feats_down_world->points.size(); ++i) {
            ss << feats_down_world->points[i].x << " " 
               << feats_down_world->points[i].y << " " 
               << feats_down_world->points[i].z << " " 
               << feats_down_world->points[i].intensity << "\n";
        }
        
        // 一次性写入所有数据
        save_map_file << ss.str();
        save_map_file.flush(); // 确保数据写入磁盘
        #endif


    }     

    #if 1
    std::lock_guard<std::mutex> lock(txt_save_mutex);
    int size = feats_undistort->points.size();
    // 创建用于存储世界坐标系下点云的新点云对象
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(size, 1));

    // 将每个点从机体坐标系转换到世界坐标系
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&feats_undistort->points[i], \
                            &laserCloudWorld->points[i]);
    }

    // 将转换后的点云累加到等待保存的点云中
    *accumulated_cloud += *laserCloudWorld;


    #endif
}
PointCloudXYZI::Ptr down_map(new PointCloudXYZI()); 
/**
 * @brief 即时点，将点云从雷达坐标系转换到世界坐标系，并保存到点云中
 * 
 */
bool loadExistingMap() {
    string map_file_path = string(ROOT_DIR) + "/PCD/accumulated_map.pcd";
    
    // 检查地图文件是否存在
    ifstream file(map_file_path);
    if (!file.good()) {
        cout << "No existing map found at: " << map_file_path << endl;
        return false;
    }
    file.close();
    
    // 加载点云地图
    PointCloudXYZI::Ptr map_cloud(new PointCloudXYZI());
    if (pcl::io::loadPCDFile<PointType>(map_file_path, *map_cloud) != 0) {
        cout << "Failed to load map from: " << map_file_path << endl;
        return false;
    }


    
    // 创建体素网格滤波器并执行下采样
    pcl::VoxelGrid<PointType> sor; 
    sor.setInputCloud(map_cloud); 
    sor.setLeafSize(0.4, 0.4, 0.4);
    sor.filter(*down_map);


    cout << "Successfully loaded map with " << map_cloud->size() << " points" << endl;
    // ikdtree.set_downsample_param(filter_size_map_min);

    // ikdtree.Build(down_map->points);
    
    return true;
}


bool lidarLocalization(){
    #if localization

    if (!init_flag) { 
            /*60秒时候的位姿
            position: 
            x: 8.605346800971496
            y: -7.8000119416576394
            z: 2.9070033479885584
            orientation: 
            x: 0.015169017370231281
            y: 0.35820880499712554
            z: 0.9329845567919434
            w: 0.031562156489285384
        */
        // //如参初始位姿估计 ，  可以由GPS给
        float x=8.605346800971496, y=-7.8000119416576394, z=2.9070033479885584, roll=0.0, pitch=0.0, yaw=0.0;
        // 创建四元数（从欧拉角转换）
        // tf2::Quaternion quat_tf2;
        // quat_tf2.setRPY(roll, pitch, yaw);
        // Eigen::Quaternionf eigen_quat(quat_tf2.w(), quat_tf2.x(), quat_tf2.y(), quat_tf2.z());

        Eigen::Quaternionf eigen_quat(0.031562156489285384, 
                                        0.015169017370231281,  
                                        0.35820880499712554,   
                                        0.9329845567919434);


        // 创建4x4位姿变换矩阵
        Eigen::Matrix4f pose_estimation = Eigen::Matrix4f::Identity();

        // 设置平移部分（来自state_point的位置）
        pose_estimation(0, 3) = x;
        pose_estimation(1, 3) = y;
        pose_estimation(2, 3) = z;

        

        Eigen::Matrix3f rotation = eigen_quat.toRotationMatrix();

        // 设置旋转部分
        pose_estimation.block<3,3>(0,0) = rotation;

        // // IMU预积分处理
        // p_imu->Process(Measures, kf, feats_undistort);
        // state_point = kf.get_x();
        // if (feats_undistort->empty() || (feats_undistort == NULL))
        // {
        //     ROS_WARN("IMU No point, skip this scan!\n");
        //     return false;
        // }

        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }

        init_localization(laserCloudWorld,filter_size_map_min,filter_size_surf_min,fov_deg,down_map);
        auto localization_resl =globalLocalization(state_point, geoQuat,pose_estimation);
        if(localization_resl.second){
            init_flag = true;
            pose_tf = localization_resl.first;
        }

        cout<<"pose_estimation: \n"<<pose_estimation(0,3)<< pose_estimation(1,3)<<endl;
        cout<<"geoQuat: "<<geoQuat.x<<" "<<geoQuat.y<<" "<<geoQuat.z<<" "<<geoQuat.w<<endl;
        cout<<"state_point.pos: "<<state_point.pos.transpose()<<endl;
        cout<<"state_point.rot: \n"<<state_point.rot.matrix()<<endl;
    }else{
        globalLocalization(state_point, geoQuat);
    }

    return true;
    #endif

}

/**
 * @brief 主函数，负责初始化ROS节点、加载参数、订阅传感器数据并执行激光雷达SLAM主循环。
 * 
 * 该函数完成以下主要任务：
 * 1. 初始化ROS节点 "laserMapping"
 * 2. 从参数服务器加载配置参数
 * 3. 初始化IMU处理模块和卡尔曼滤波器
 * 4. 订阅激光雷达和IMU数据
 * 5. 执行主循环进行点云去畸变、特征匹配、状态估计和地图更新
 * 6. 发布里程计、路径和点云数据
 * 7. 保存轨迹日志和点云地图
 * 
 * @param argc 命令行参数个数
 * @param argv 命令行参数数组
 * @return int 程序退出状态码，正常退出返回0
 */
int main(int argc, char** argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    // 从参数服务器加载配置参数
    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    // nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    // nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<string>("common/lid_topic",lid_topic,"/vanjee_points722f");
    nh.param<string>("common/imu_topic", imu_topic,"/vanjee_lidar_imu_packets");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", lidar_type, AVIA); //AVIA
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2); //2
    //nh.param<int>("point_filter_num", p_pre->point_filter_num, 2); 
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, true);
    //nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, true); //0
     nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0); //0
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, true);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<bool>("pcd_save/save_map", save_map, false);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());

    p_pre->lidar_type = lidar_type;

    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** 变量定义 ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    // 设置视场角相关参数
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    // 初始化点选择和残差数组
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    // 设置激光雷达与IMU的外参和协方差
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->lidar_type = lidar_type;
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** 调试日志文件初始化 ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    /*** ROS订阅和发布初始化 ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);

    // //订阅
    ros::Subscriber create_lader_map = nh.subscribe<std_msgs::Bool>("create_map", 100,save_map_cbk);

    string map_filename = root_dir + "src/StaticMap/static_map.txt";

    save_map_file.open(map_filename, ios::out | ios::app);

    
    if (!save_map_file.is_open()) {
        ROS_ERROR("Cannot open file %s for writing point cloud data", map_filename.c_str());
        return 0;
    }
    //bool loadMap = loadExistingMap();
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    
    // 主循环
    while (status)
    {   
        if (flg_exit) break;
        ros::spinOnce();
        // 同步激光雷达和IMU数据包
        if(sync_packages(Measures)) //拿到雷达结束前时间段内的所有imu数据
        {
            if(flg_first_scan){
                p_imu->Reset();
                Measures.imu.clear();
                flg_first_scan = false;
                continue;   
            }

            // IMU预积分处理
            p_imu->Process(Measures, kf, feats_undistort);

            state_point = kf.get_x();

            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("IMU No point, skip this scan!\n");
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            // if(loadMap){
            //     lidarLocalization();
            // }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            /*** 根据激光雷达视场角分割地图 ***/
            lasermap_fov_segment();

            /*** 对扫描中的特征点进行降采样 ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            
            /*** 初始化地图kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            // int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;
            cout<<"[ mapping ]: kdtree_size: "<<kdtree_size_st<<endl;
            /*** ICP和迭代卡尔曼滤波更新 ***/
            ROS_INFO("Downsampled points: %d", feats_down_size);
            if (feats_down_size < 5)
            {
                ROS_WARN("ICP No point, skip this scan {feats_down_size}!\n");
                //ROS_WARN("Original undistorted points: %zu", feats_undistort->points.size());
                continue;
            }
            t1 = omp_get_wtime();



            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            // fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            // <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // 如果需要查看地图点，改为"if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
                //cout<<"[map_update]: map size: "<<featsFromMap->points.size()<<endl;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** 迭代状态估计 ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();
            //cout<< "t_update_end - t_update_start: "<< t_update_end - t_update_start <<endl;
            // cout << "match_time = " << static_cast<long long>(match_time * 1e6) << endl;
            // cout<< "ikd_tree.size()  =  "<< static_cast<long long>(ikdtree.size())<<endl;


            /******* 发布里程计信息 *******/
            publish_odometry(pubOdomAftMapped);

            /*** 将特征点添加到地图kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* 发布点云数据 *******/
            if (path_en)                         publish_path(pubPath);
            if ((scan_pub_en || pcd_save_en) )      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);
            if(save_map)                         exportStaticMapExample(); //存txt静态地图 ，换了存pcd



            /*** 调试变量记录 ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                /*

            - `s_plot[time_log_counter]`: **总处理时间** - 从t0到t5的完整处理时间，即`T5 - T0`
                    t0: 开始时间 - 整个处理循环开始的时间点
                    t1: IMU预处理和降采样完成时间 - 包括IMU预积分处理和点云降采样
                    t2: ICP匹配前时间 - 特征匹配前的准备工作完成时间
                    t3: 状态估计完成时间 - EKF迭代优化完成的时间
                    t4: 未使用 - 代码中没有明显使用t4
                    t5: 结束时间 - 地图增量更新完成的时间点

            - `s_plot2[time_log_counter]`: **扫描点数量** - 当前帧去畸变后的点云数量，即`feats_undistort->points.size()`
            - `s_plot3[time_log_counter]`: **KD树增量时间** - 向KD树添加新点的时间，即`kdtree_incremental_time`
            - `s_plot4[time_log_counter]`: **KD树搜索时间** - 在KD树中搜索近邻点的时间，即`kdtree_search_time`
            - `s_plot5[time_log_counter]`: **KD树删除计数** - 从KD树中删除的点的数量，即`kdtree_delete_counter`
            - `s_plot6[time_log_counter]`: **KD树删除时间** - 从KD树删除点所用的时间，即`kdtree_delete_time`
            - `s_plot7[time_log_counter]`: **KD树初始大小** - 处理前KD树中的点数，即`kdtree_size_st`
            - `s_plot8[time_log_counter]`: **KD树最终大小** - 处理后KD树中的点数，即`kdtree_size_end`
            - `s_plot9[time_log_counter]`: **平均消耗时间** - 截至当前帧的平均总处理时间，即`aver_time_consu`
            - `s_plot10[time_log_counter]`: **新增点数** - 本次添加到地图中的点的数量，即`add_point_size`

                */
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                //printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
          

            // cout << "t1 at: " << static_cast<long long>((t1 -t0) * 1e6) << " microseconds" << endl;
            // cout << "t2 at: " << static_cast<long long>((t2-t1) * 1e6) << " microseconds" << endl;
            // cout << "t3 at: " << static_cast<long long>((t3 -t2)* 1e6) << " microseconds" << endl;
            // //cout << "t4 at: " << static_cast<long long>(t4 * 1e6) << " microseconds" << endl;
            // cout << "t5 at: " << static_cast<long long>((t5 -t3) * 1e6) << " microseconds" << endl;
            double t6 = omp_get_wtime() - t0;
            cout << "t6 at: " << static_cast<long long>(t6 * 1e6) << " microseconds" << endl;
            //cout << "s_plot11 at: " << static_cast<long long>(s_plot11[0]* 1e6) << " microseconds" << endl;
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** 保存地图 ****************/
    /* 1. 确保有足够的内存  这个是按照帧率数量保存
    /* 2. pcd保存会严重影响实时性能 **/ 
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    
    //保存地图    这个是按照开关标志  save_map
    save_map = false;
    save_map_file.close();
    cout<<"accumulated_cloud size: "<<accumulated_cloud->size()<<endl;
    // 当累积点云数量,保存点云到PCD文件
    if (accumulated_cloud->size() > 0 && save_map == false) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        // 格式化时间戳字符串
        std::stringstream timestamp_ss;
        timestamp_ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << "_" << std::setfill('0') << std::setw(3) << ms.count();

        string all_points_dir(string(ROOT_DIR) + "/PCD/map_" + timestamp_ss.str() + ".pcd");
        pcl::PCDWriter pcd_writer;
        cout << "Saving accumulated point cloud to " << all_points_dir << endl;
        pcd_writer.writeBinary(all_points_dir, *accumulated_cloud);
        accumulated_cloud->clear();
    }
                 
        // 保存时间日志
    if (runtime_pos_log)
        {
            vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
            FILE *fp2;
            string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
            fp2 = fopen(log_dir.c_str(),"w+");
            fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
            for (int i = 0;i<time_log_counter; i++){
                //s_plot11   雷达入参预处理耗时
                fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
                t.push_back(T1[i]);
                s_vec.push_back(s_plot9[i]);
                s_vec2.push_back(s_plot3[i] + s_plot6[i]);
                s_vec3.push_back(s_plot4[i]);
                s_vec5.push_back(s_plot[i]);
            }
            fclose(fp2);
        }
        


    fout_out.close();
    fout_pre.close();



    return 0;
}
