#include "globalLocalization.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "fast_lio_localization");

    ros::NodeHandle nh;
    // 初始化参数
    nh.param("map_voxel_size", MAP_VOXEL_SIZE, 0.4);
    nh.param("scan_voxel_size", SCAN_VOXEL_SIZE, 0.1);
    nh.param("freq_localization", FREQ_LOCALIZATION, 0.5);
    nh.param("localization_th", LOCALIZATION_TH, 0.3);
    nh.param("fov", FOV, 6.28);
    nh.param("fov_far", FOV_FAR, 30.0);
    nh.param("map_file_path", map_file_path, std::string("accumulated_map.pcd"));

    GlobalLocalization global_localization(nh);
    // Publisher
    pub_pc_in_map = nh.advertise<sensor_msgs::PointCloud2>("/cur_scan_in_map", 100000);
    pub_submap = nh.advertise<sensor_msgs::PointCloud2>("/submap", 100000);
    pub_map_to_odom = nh.advertise<nav_msgs::Odometry>("/map_to_odom", 100000);
    
    // Subscriber
    sub_cloud_registered = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_registered", 100000, &GlobalLocalization::cbSaveCurScan, &global_localization);
    sub_odometry = nh.subscribe<nav_msgs::Odometry>("/Odometry", 100000, &GlobalLocalization::cbSaveCurOdom, &global_localization);
    ros::Subscriber initial_pose_sub = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/initialpose", 100000, &GlobalLocalization::initialPoseCallback, &global_localization);

    path_pub = nh.advertise<nav_msgs::Path>("/odom_path_test", 10);
    pcl_pub = nh.advertise<sensor_msgs::PointCloud2> ("pcl_output", 10000);

    ROS_INFO("Localization Node Inited...");
    
    cout<<"map_file_path: "<<map_file_path<<endl;
 
    global_localization.run(map_file_path);
    ros::spin(); 
    return 0;
}



