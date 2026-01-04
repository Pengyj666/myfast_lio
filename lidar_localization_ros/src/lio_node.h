
#ifndef LIO_NODE_H
#define LIO_NODE_H


#include <omp.h>
#include <math.h>
#include <thread>
#include <csignal>
#include <unistd.h>
#include <so3_math.h>
#include <Eigen/Core>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Float64.h>
#include <fstream>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include "mower_msgs/Trigger.h"
#include "lio_code.h"

#include "lio_helper.h"
#include "IMU_Processing.h"
#include "ikd-Tree/ikd_Tree.h"
#include "lio_code.h"
#include "use-ikfom.hpp"
#include "preprocess.h"

#include "common/sysutils.h"
#include "droslog/log.h"
#include "droslog/logclient.h"
#include "common/offset_timer.h"

namespace utils {
   inline p_log_func dros_log_func_ptr;
}
using namespace utils;
class lioNode{
private:
    // ROS相关成员
    ros::Subscriber sub_pcl;
    ros::Subscriber sub_imu ;
    ros::Publisher pubOffsetTs;
    ros::Publisher pubLaserCloudFull;
    ros::Publisher pubLaserCloudFull_body ;
    ros::Publisher pubCloud_body;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubOdomAftMappedBase ;
    ros::Publisher pubPath ;
    ros::ServiceServer serv_ctrl_mapping;
    ros::ServiceServer serv_save_mapping;

    // 轨迹路径消息，用于发布机器人的运动轨迹
    nav_msgs::Path path;
    // 里程计消息，存储滤波后的位置和姿态信息
    nav_msgs::Odometry odomAftMapped;

    nav_msgs::Odometry odomAftMappedBase;
    // 四元数消息，用于表示机器人姿态
    geometry_msgs::Quaternion geoQuat;
    // 位姿消息，包含机器人在body坐标系下的位姿信息
    geometry_msgs::PoseStamped msg_body_pose;

    mutex accumulated_cloud_mutex;
    PointCloudXYZI::Ptr accumulated_cloud;
    
    std::atomic<bool>  odom_file_initialized = {false};
    std::atomic<bool> path_en = {true};
    std::atomic<bool> save_map{false};
    std::ofstream odom_file;
    
    bool time_sync_en = false;
    bool   scan_pub_en = true, dense_pub_en = false, scan_body_pub_en = true;
    int lidar_type;
    double lidar_d;
    bool init_flag = false;
    double time_diff_lidar_to_imu;
    string lid_topic, imu_topic;
    vector<double> translation_body;

    Eigen::Matrix4f pose_tf = Eigen::Matrix4f::Identity();

    std::shared_ptr<LioCode>  lio_controller;
    std::shared_ptr<LioHelper>  lio_helper;
    std::shared_ptr<Preprocess> p_pre;
public:
    lioNode(ros::NodeHandle &nh);
    ~lioNode();
    void publish_path();
    void publish_odometry();
    void publish_frame_body();
    void publish_body();
    void publish_frame_world( );
    void init_param( ros::NodeHandle & nh);
    void init_subAndpub( ros::NodeHandle & nh);

    void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) ;
    void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) ;
    void publish_this();
    bool save_map_cbk(mower_msgs::TriggerRequest &req,mower_msgs::TriggerResponse &res);
    bool ctrl_mapping_cbk(mower_msgs::TriggerRequest &req,mower_msgs::TriggerResponse &res);
    void reset();
    bool init();
    void start();
    void set_geoQuat(state_ikfom& state_point);

    void save_map_point(vector<PointType, Eigen::aligned_allocator<PointType>>  PointToAdd,
                    vector<PointType, Eigen::aligned_allocator<PointType>>  PointNoNeedDownsample);
    void clear_map_point();

    OffsetTimer* OffsetTimerIns() {
        static OffsetTimer offset_timer("lidar");
        return &offset_timer;
    };

    template<typename T>
    void set_posestamp(T & out);
    template<typename T>
    void set_basepose(T & out);
};

template<typename T>
void lioNode::set_posestamp(T & out)
{
    auto state_point = lio_helper->get_state_point();
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}

template<typename T>
void lioNode::set_basepose(T & out)
{ 
    auto state_point = lio_helper->get_state_point();
    double lidar_Radian = lidar_d*M_PI/180;
    M3D calibrateTilt_X;
    M3D calibrateTilt_Z;
    calibrateTilt_X << 1, 0, 0,
                    0, cos(lidar_Radian), sin(lidar_Radian),
                    0, -sin(lidar_Radian),cos(lidar_Radian);

    calibrateTilt_Z << 0, -1, 0,
                1, 0, 0,
                0, 0, 1;
    M3D total_rot_ = calibrateTilt_Z * calibrateTilt_X;

    V3D pos(state_point.pos(0), state_point.pos(1), state_point.pos(2));
    V3D rotated_pos = total_rot_ * pos; 
    out.pose.position.x = rotated_pos(0);
    out.pose.position.y = rotated_pos(1);
    out.pose.position.z = rotated_pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}


#endif
