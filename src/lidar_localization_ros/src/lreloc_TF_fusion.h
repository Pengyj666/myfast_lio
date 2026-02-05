#ifndef LRELOC_TF_FUSION_H
#define LRELOC_TF_FUSION_H

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <thread>
#include <chrono>
#include <mutex>
#include <nav_msgs/Path.h>


class lrelocTFFusion {
private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_odometry_;
    ros::Subscriber sub_map_to_odom_;
    ros::Publisher pub_map_to_odom_result;
    ros::Publisher pub_localization_;
    ros::Publisher pub_localization_base_;
    ros::Publisher pub_reloc_Path;
    
    nav_msgs::Path reloc_Path;
    // 存储从里程计坐标系到基座坐标系的变换关系
    nav_msgs::Odometry cur_odom_to_baselink_;
    // 存储从地图坐标系到里程计坐标系的变换关系
    nav_msgs::Odometry cur_map_to_odom_;
    
    bool odom_received_;
    bool map_to_odom_received_;
    
    std::mutex odom_mutex_;
    std::mutex map_to_odom_mutex_;
    std::thread fusion_thread;
    
    double FREQ_PUB_LOCALIZATION_;
    double lidar_d;

public:
    lrelocTFFusion(double lidar_d_ = -6.7);
    ~lrelocTFFusion();
    
    Eigen::Matrix4d poseToMatrix(const nav_msgs::Odometry& odom_msg);
    void cbSaveCurOdom(const nav_msgs::OdometryConstPtr& odom_msg);
    void cbSaveMapToOdom(const nav_msgs::OdometryConstPtr& odom_msg);
    void transformFusion();
    void pub_reloc_lio_path(Eigen::Matrix4d &T_map_to_base_link,geometry_msgs::PoseStamped& msg_body_pose);
    void pub_localization_base(Eigen::Matrix4d &T_map_to_base_link,nav_msgs::Odometry& localization_base);
    void pub_localization(Eigen::Matrix4d& T_map_to_base_link,nav_msgs::Odometry &localization);
    void pub_reloc_result(Eigen::Matrix4d &T_map_to_base_link,nav_msgs::Odometry& localization_base);
};

#endif // LRELOC_TF_FUSION_H