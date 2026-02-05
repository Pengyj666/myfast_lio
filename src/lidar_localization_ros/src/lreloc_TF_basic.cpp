#include "lreloc_TF_fusion.h"
#include <tf_conversions/tf_eigen.h>

lrelocTFFusion::lrelocTFFusion(double lidar_d_) : 
    odom_received_(false),
    map_to_odom_received_(false) {
    lidar_d = lidar_d_;
    // Parameters
    nh_.param("freq_pub_localization", FREQ_PUB_LOCALIZATION_, 10.0);
    
    // Subscribers
    sub_odometry_ = nh_.subscribe("/as_lio/org_lio", 3, &lrelocTFFusion::cbSaveCurOdom, this);
    sub_map_to_odom_ = nh_.subscribe("/as_lio/org_reloc_result", 3, &lrelocTFFusion::cbSaveMapToOdom, this);

    // Publisher
    pub_localization_ = nh_.advertise<nav_msgs::Odometry>("/as_lio/org_reloc_lio", 1);
    pub_localization_base_ = nh_.advertise<nav_msgs::Odometry>("/as_lio/reloc_lio", 1);
    pub_map_to_odom_result = nh_.advertise<nav_msgs::Odometry>("/as_lio/reloc_result", 1);
    pub_reloc_Path = nh_.advertise<nav_msgs::Path>("/as_lio/reloc_lio_path", 5);
    
    fusion_thread = std::thread(&lrelocTFFusion::transformFusion, this);
    ROS_INFO("lrelocTFFusion Node Inited...");
    ros::spinOnce();
}

lrelocTFFusion::~lrelocTFFusion() {
    if (fusion_thread.joinable()) {
        fusion_thread.join();
    }
}

Eigen::Matrix4d lrelocTFFusion::poseToMatrix(const nav_msgs::Odometry& odom_msg) {
    const geometry_msgs::Pose& pose = odom_msg.pose.pose;
    
    Eigen::Quaterniond quat(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    Eigen::Matrix3d rotation = quat.toRotationMatrix();
    Eigen::Vector3d translation(pose.position.x, pose.position.y, pose.position.z);
    
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3,3>(0,0) = rotation;
    transform.block<3,1>(0,3) = translation;
    
    return transform;
}

void lrelocTFFusion::cbSaveCurOdom(const nav_msgs::OdometryConstPtr& odom_msg) {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    cur_odom_to_baselink_ = *odom_msg;
    odom_received_ = true;
}

void lrelocTFFusion::cbSaveMapToOdom(const nav_msgs::OdometryConstPtr& odom_msg) {
    std::lock_guard<std::mutex> lock(map_to_odom_mutex_);
    cur_map_to_odom_ = *odom_msg;
    map_to_odom_received_ = true;
}

void lrelocTFFusion::pub_reloc_lio_path(Eigen::Matrix4d &T_map_to_base_link,geometry_msgs::PoseStamped& msg_body_pose){ 
    Eigen::Vector3d xyz = T_map_to_base_link.block<3,1>(0,3);
    Eigen::Matrix3d R = T_map_to_base_link.block<3,3>(0,0);
    Eigen::Quaterniond quat_result(R);

    double lidar_Radian = lidar_d*M_PI/180;
    Eigen::Matrix3d calibrateTilt_X;
    Eigen::Matrix3d calibrateTilt_Z;
    calibrateTilt_X << 1, 0, 0,
                0, cos(lidar_Radian), sin(lidar_Radian),
                0, -sin(lidar_Radian),cos(lidar_Radian);
    calibrateTilt_Z << 0, -1, 0,
            1, 0, 0,
            0, 0, 1;
    Eigen::Matrix3d total_rot_ = calibrateTilt_Z * calibrateTilt_X;

    Eigen::Vector3d pos(xyz.x(), xyz.y(), xyz.z());
    Eigen::Vector3d rotated_pos = total_rot_ * pos; 
    msg_body_pose.pose.position.x = rotated_pos(0);
    msg_body_pose.pose.position.y = rotated_pos(1);
    msg_body_pose.pose.position.z = rotated_pos(2);
    msg_body_pose.pose.orientation.x = quat_result.x();
    msg_body_pose.pose.orientation.y = quat_result.y();
    msg_body_pose.pose.orientation.z = quat_result.z();
    msg_body_pose.pose.orientation.w = quat_result.w();
    reloc_Path.header.frame_id = "map";
    static int count = 0;
    static int jjj = 0;

    count++;
    if(count >= 5000){
        --count;
        reloc_Path.poses.push_back(msg_body_pose);
        reloc_Path.poses.erase(reloc_Path.poses.begin());  
    }else{
        reloc_Path.poses.push_back(msg_body_pose);
    }
    reloc_Path.header.stamp = ros::Time::now();

    if (++jjj % 5 == 0) {
        jjj = 0;
        pub_reloc_Path.publish(reloc_Path);
    }

}

void lrelocTFFusion::pub_localization(Eigen::Matrix4d& T_map_to_base_link,nav_msgs::Odometry &localization){
    // 提取位置和姿态信息
    Eigen::Vector3d xyz = T_map_to_base_link.block<3,1>(0,3);
    Eigen::Matrix3d R = T_map_to_base_link.block<3,3>(0,0);
    Eigen::Quaterniond quat_result(R);

    localization.pose.pose.position.x = xyz.x();
    localization.pose.pose.position.y = xyz.y();
    localization.pose.pose.position.z = xyz.z();
    localization.pose.pose.orientation.x = quat_result.x();
    localization.pose.pose.orientation.y = quat_result.y();
    localization.pose.pose.orientation.z = quat_result.z();
    localization.pose.pose.orientation.w = quat_result.w();

    pub_localization_.publish(localization);

}

void lrelocTFFusion::pub_localization_base(Eigen::Matrix4d &T_map_to_base_link,nav_msgs::Odometry& localization_base){ 
    Eigen::Vector3d xyz = T_map_to_base_link.block<3,1>(0,3);
    Eigen::Matrix3d R = T_map_to_base_link.block<3,3>(0,0);
    Eigen::Quaterniond quat_result(R);

    double lidar_Radian = lidar_d*M_PI/180;
    Eigen::Matrix3d calibrateTilt_X;
    Eigen::Matrix3d calibrateTilt_Z;
    calibrateTilt_X << 1, 0, 0,
                0, cos(lidar_Radian), sin(lidar_Radian),
                0, -sin(lidar_Radian),cos(lidar_Radian);

    calibrateTilt_Z << 0, -1, 0,
            1, 0, 0,
            0, 0, 1;
    Eigen::Matrix3d total_rot_ = calibrateTilt_Z * calibrateTilt_X;

    Eigen::Vector3d pos(xyz.x(), xyz.y(), xyz.z());
    Eigen::Vector3d rotated_pos = total_rot_ * pos; 
    localization_base.pose.pose.position.x = rotated_pos(0);
    localization_base.pose.pose.position.y = rotated_pos(1);
    localization_base.pose.pose.position.z = rotated_pos(2);
    localization_base.pose.pose.orientation.x = quat_result.x();
    localization_base.pose.pose.orientation.y = quat_result.y();
    localization_base.pose.pose.orientation.z = quat_result.z();
    localization_base.pose.pose.orientation.w = quat_result.w();

    pub_localization_base_.publish(localization_base);
}

void lrelocTFFusion::pub_reloc_result(Eigen::Matrix4d &T_map_to_base_link,nav_msgs::Odometry& localization_base) { 
        Eigen::Vector3d xyz = T_map_to_base_link.block<3,1>(0,3);
    Eigen::Matrix3d R = T_map_to_base_link.block<3,3>(0,0);
    Eigen::Quaterniond quat_result(R);

    double lidar_Radian = lidar_d*M_PI/180;
    Eigen::Matrix3d calibrateTilt_X;
    Eigen::Matrix3d calibrateTilt_Z;
    calibrateTilt_X << 1, 0, 0,
                0, cos(lidar_Radian), sin(lidar_Radian),
                0, -sin(lidar_Radian),cos(lidar_Radian);

    calibrateTilt_Z << 0, -1, 0,
            1, 0, 0,
            0, 0, 1;
    Eigen::Matrix3d total_rot_ = calibrateTilt_Z * calibrateTilt_X;

    Eigen::Vector3d pos(xyz.x(), xyz.y(), xyz.z());
    Eigen::Vector3d rotated_pos = total_rot_ * pos; 
    localization_base.pose.pose.position.x = rotated_pos(0);
    localization_base.pose.pose.position.y = rotated_pos(1);
    localization_base.pose.pose.position.z = rotated_pos(2);
    localization_base.pose.pose.orientation.x = quat_result.x();
    localization_base.pose.pose.orientation.y = quat_result.y();
    localization_base.pose.pose.orientation.z = quat_result.z();
    localization_base.pose.pose.orientation.w = quat_result.w();

    pub_map_to_odom_result.publish(localization_base);
}
