#include "lreloc_TF_fusion.h"
#include <tf_conversions/tf_eigen.h>

void lrelocTFFusion::transformFusion() {
    static tf::TransformBroadcaster br;
    ros::Rate rate(FREQ_PUB_LOCALIZATION_);
    
    while (ros::ok()) {
        ros::spinOnce();
        // 获取当前里程计数据（线程安全）
        nav_msgs::Odometry cur_odom;
        static std::atomic<bool> has_odom = {false};
        {
            std::lock_guard<std::mutex> lock(odom_mutex_);
            if (odom_received_) {
                odom_received_= false;
                cur_odom = cur_odom_to_baselink_;
                has_odom.store(true);
            }
        }
        
        // 获取当前map到odom变换（线程安全）
        static nav_msgs::Odometry map_to_odom;
        bool has_map_to_odom = false;
        {
            std::lock_guard<std::mutex> lock(map_to_odom_mutex_);
            if (map_to_odom_received_) {
                map_to_odom = cur_map_to_odom_;
                has_map_to_odom = true;
                map_to_odom_received_ = false;
            }
        }
        
        // 计算T_map_to_odom变换矩阵
        Eigen::Matrix4d T_map_to_odom = Eigen::Matrix4d::Identity();
        // if (has_map_to_odom) {
            T_map_to_odom = poseToMatrix(map_to_odom);
        // }
        
        // 发布TF变换
        Eigen::Vector3d translation = T_map_to_odom.block<3,1>(0,3);
        Eigen::Matrix3d rotation = T_map_to_odom.block<3,3>(0,0);
        Eigen::Quaterniond quat(rotation);
        
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(translation.x(), translation.y(), translation.z()));
        transform.setRotation(tf::Quaternion(quat.x(), quat.y(), quat.z(), quat.w()));
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "camera_init"));
        
        // 如果有里程计数据，发布全局定位的odometry
        if (has_odom.load()) {
            has_odom.store(false);
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

            pub_localization(T_map_to_base_link,localization);

            nav_msgs::Odometry localization_base;
            localization_base.header.stamp = cur_odom.header.stamp;
            localization_base.header.frame_id = "map";
            localization_base.child_frame_id = "body_base";
            pub_localization_base(T_map_to_base_link,localization_base);

            geometry_msgs::PoseStamped msg_body_pose;
            msg_body_pose.header.stamp = cur_odom.header.stamp;
            msg_body_pose.header.frame_id = "map";
            pub_reloc_lio_path(T_map_to_base_link, msg_body_pose);

            if(has_map_to_odom){
                nav_msgs::Odometry map_to_odom_base;
                map_to_odom_base.header.stamp = cur_odom.header.stamp;
                map_to_odom_base.header.frame_id = "map";
                map_to_odom_base.child_frame_id = "map_to_odom_base";
                pub_reloc_result(T_map_to_base_link,map_to_odom_base);
            }
        }
        rate.sleep();
    }
}