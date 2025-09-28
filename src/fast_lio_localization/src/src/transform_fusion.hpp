// transform_fusion_node.cpp
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

class TransformFusion {
private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_odometry_;
    ros::Subscriber sub_map_to_odom_;
    ros::Publisher pub_localization_;
    
    // 存储从里程计坐标系到基座坐标系的变换关系
    nav_msgs::Odometry cur_odom_to_baselink_;
    // 存储从地图坐标系到里程计坐标系的变换关系
    nav_msgs::Odometry cur_map_to_odom_;
    
    bool odom_received_;
    bool map_to_odom_received_;
    
    std::mutex odom_mutex_;
    std::mutex map_to_odom_mutex_;
    
    double FREQ_PUB_LOCALIZATION_;
    
public:
    TransformFusion() : 
        odom_received_(false),
        map_to_odom_received_(false) {
        
        // Parameters
        nh_.param("freq_pub_localization", FREQ_PUB_LOCALIZATION_, 50.0);
        
        // Subscribers
        sub_odometry_ = nh_.subscribe("/Odometry", 1, &TransformFusion::cbSaveCurOdom, this);
        sub_map_to_odom_ = nh_.subscribe("/map_to_odom", 1, &TransformFusion::cbSaveMapToOdom, this);
        
        // Publisher
        pub_localization_ = nh_.advertise<nav_msgs::Odometry>("/localization", 1);
        
        ROS_INFO("Transform Fusion Node Inited...");
    }
    
    Eigen::Matrix4d poseToMatrix(const nav_msgs::Odometry& odom_msg) {
        const geometry_msgs::Pose& pose = odom_msg.pose.pose;
        
        Eigen::Quaterniond quat(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
        Eigen::Matrix3d rotation = quat.toRotationMatrix();
        Eigen::Vector3d translation(pose.position.x, pose.position.y, pose.position.z);
        
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        transform.block<3,3>(0,0) = rotation;
        transform.block<3,1>(0,3) = translation;
        
        return transform;
    }
    
    void cbSaveCurOdom(const nav_msgs::OdometryConstPtr& odom_msg) {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        cur_odom_to_baselink_ = *odom_msg;
        odom_received_ = true;
    }
    
    void cbSaveMapToOdom(const nav_msgs::OdometryConstPtr& odom_msg) {
        std::lock_guard<std::mutex> lock(map_to_odom_mutex_);
        cur_map_to_odom_ = *odom_msg;
        map_to_odom_received_ = true;
    }
    
    void transformFusion() {
        static tf::TransformBroadcaster br;
        ros::Rate rate(FREQ_PUB_LOCALIZATION_);
        
        while (ros::ok()) {
            ros::spinOnce();
            // 获取当前里程计数据（线程安全）
            nav_msgs::Odometry cur_odom;
            bool has_odom = false;
            {
                std::lock_guard<std::mutex> lock(odom_mutex_);
                if (odom_received_) {
                    cur_odom = cur_odom_to_baselink_;
                    has_odom = true;
                }
            }
            
            // 获取当前map到odom变换（线程安全）
            nav_msgs::Odometry map_to_odom;
            bool has_map_to_odom = false;
            {
                std::lock_guard<std::mutex> lock(map_to_odom_mutex_);
                if (map_to_odom_received_) {
                    map_to_odom = cur_map_to_odom_;
                    has_map_to_odom = true;
                }
            }
            
            // 计算T_map_to_odom变换矩阵
            Eigen::Matrix4d T_map_to_odom = Eigen::Matrix4d::Identity();
            if (has_map_to_odom) {
                T_map_to_odom = poseToMatrix(map_to_odom);
            }
            
            // 发布TF变换
            Eigen::Vector3d translation = T_map_to_odom.block<3,1>(0,3);
            Eigen::Matrix3d rotation = T_map_to_odom.block<3,3>(0,0);
            Eigen::Quaterniond quat(rotation);
            
            tf::Transform transform;
            transform.setOrigin(tf::Vector3(translation.x(), translation.y(), translation.z()));
            transform.setRotation(tf::Quaternion(quat.x(), quat.y(), quat.z(), quat.w()));
            br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "camera_init"));
            
            // 如果有里程计数据，发布全局定位的odometry
            if (has_odom) {
                // 计算T_odom_to_base_link变换矩阵
                Eigen::Matrix4d T_odom_to_base_link = poseToMatrix(cur_odom);
                
                // 计算T_map_to_base_link = T_map_to_odom * T_odom_to_base_link
                Eigen::Matrix4d T_map_to_base_link = T_map_to_odom * T_odom_to_base_link;
                
                // 提取位置和姿态信息
                Eigen::Vector3d xyz = T_map_to_base_link.block<3,1>(0,3);
                Eigen::Matrix3d R = T_map_to_base_link.block<3,3>(0,0);
                Eigen::Quaterniond quat_result(R);
                
                // 发布全局定位的odometry
                nav_msgs::Odometry localization;
                localization.header.stamp = cur_odom.header.stamp;
                localization.header.frame_id = "map";
                localization.child_frame_id = "body";
                
                localization.pose.pose.position.x = xyz.x();
                localization.pose.pose.position.y = xyz.y();
                localization.pose.pose.position.z = xyz.z();
                localization.pose.pose.orientation.x = quat_result.x();
                localization.pose.pose.orientation.y = quat_result.y();
                localization.pose.pose.orientation.z = quat_result.z();
                localization.pose.pose.orientation.w = quat_result.w();
                
                localization.twist = cur_odom.twist;
                
                pub_localization_.publish(localization);
            }
            
            rate.sleep();
        }
    }
    
    void run() {
        // 在单独的线程中运行变换融合
        std::thread fusion_thread(&TransformFusion::transformFusion, this);
        
        ros::spin();
        
        if (fusion_thread.joinable()) {
            fusion_thread.join();
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "transform_fusion");
    
    TransformFusion transform_fusion;
    transform_fusion.run();
    
    return 0;
}