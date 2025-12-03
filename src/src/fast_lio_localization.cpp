#include "globalLocalization.h"
#include "transform_fusion.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "fast_lio_localization");

    ros::NodeHandle nh;
    // 初始化参数
    nh.param("map_voxel_size", MAP_VOXEL_SIZE, 0.2);
    nh.param("scan_voxel_size", SCAN_VOXEL_SIZE, 0.1);
    nh.param("freq_localization", FREQ_LOCALIZATION, 1.0);
    nh.param("localization_th", LOCALIZATION_TH, 0.3);
    nh.param("map_file_path", map_file_path, std::string("accumulated_map.pcd"));
    nh.param("odom_file_path", odom_file_path, std::string(""));

    GlobalLocalization global_localization(nh);
    TransformFusion transform_fusion;    // 启动重定位变换后的位姿发布线程 /as_lio/map_to_odom
    global_localization.run(map_file_path,odom_file_path); 
    ROS_INFO("Localization Node Inited... map_file_path: %s", map_file_path.c_str());
    ros::spin(); 
    return 0;
}
