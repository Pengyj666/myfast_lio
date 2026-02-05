#include "lreloc_node.h"

lreloc_node::lreloc_node(ros::NodeHandle& nh) : 
                        cur_scan(new pcl::PointCloud<PointT>),
                        lreloc(std::make_shared<lreloc_function>()) {
    // Publisher
    pub_pc_in_map = nh.advertise<sensor_msgs::PointCloud2>("/cur_scan_in_map", 3);
    pub_submap = nh.advertise<sensor_msgs::PointCloud2>("/submap", 3);
    pub_map_to_odom = nh.advertise<nav_msgs::Odometry>("/as_lio/org_reloc_result", 3);
    
    // Subscriber
    sub_cloud_registered = nh.subscribe<sensor_msgs::PointCloud2>("/as_lio/cloud_registered", 3, &lreloc_node::cbSaveCurScan, this);
    sub_odometry = nh.subscribe<nav_msgs::Odometry>("/as_lio/org_lio", 3, &lreloc_node::cbSaveCurOdom, this);
    path_pub = nh.advertise<nav_msgs::Path>("/odom_path_test", 3);
    initial_pose_sub = nh.subscribe<nav_msgs::Odometry>("/odom_fused", 3, &lreloc_node::initialPoseCallback, this);
    subOffsetTs = nh.subscribe<std_msgs::Float64>("/as_lio/offset_ts", 3, &lreloc_node::callback_lio_offset_ts, this);

    // Service
    serv_load_mapping_ = nh.advertiseService("/as_lio/loadmap", &lreloc_node::loadMapCallback,this);
    serv_onOroff_relocation_ = nh.advertiseService("/as_lio/onoroff_relocation", &lreloc_node::onoroff_relocation,this);
    cur_odom_queue.reset(2048);
}


bool lreloc_node::init(ros::NodeHandle& nh){

    double MAP_VOXEL_SIZE;
    double SCAN_VOXEL_SIZE;
    double LOCALIZATION_TH;
    std::string map_file_path;
    std::string odom_file_path;
    std::string g_map_root_dir = "/userdata/RobotData/map/";
    // 初始化参数
    nh.param("map_voxel_size", MAP_VOXEL_SIZE, 0.2);
    nh.param("scan_voxel_size", SCAN_VOXEL_SIZE, 0.1);
    nh.param("freq_localization", FREQ_LOCALIZATION, 1.0);
    nh.param("localization_th", LOCALIZATION_TH, 0.3);
    nh.param("map_file_path", map_file_path, std::string("accumulated_map.pcd"));
    nh.param("odom_file_path", odom_file_path, std::string(""));
    nh.param("lidar_d", lidar_d, 0.0);
    if(odom_file_path != ""){
        std::string odom_file = std::string(ROOT_DIR) + odom_file_path;
        cout << "=====================Loading odom from: " << odom_file << endl;
        odom_path_thread_ptr = std::make_unique<std::thread>([this,odom_file]() {
            this->threadOdomPath(odom_file);
        });
    }
        lreloc->init( LOCALIZATION_TH, map_file_path, g_map_root_dir, MAP_VOXEL_SIZE,lidar_d);
    lreloc->regPubMapToOdomCallback([this](std::shared_ptr<Eigen::Matrix4f> transform) {
        this->pub_mapToOdom(*transform);
    });
    cout << "=====================lreloc init done -----------------" << endl;
    return true;
}

lreloc_node::~lreloc_node() {
    if (lreloc) {
        lreloc.reset();
    }
    
    if (odom_path_thread_ptr && odom_path_thread_ptr->joinable()) {
        odom_path_thread_ptr->join();
    }

    if (cur_scan) {
        cur_scan.reset();
    }
}
void lreloc_node::run(){
    cout<< "----------lreloc_function run----------"<<endl;
    int number = 0;

    ros::Rate rate(FREQ_LOCALIZATION);  
    while (ros::ok()) {
        rate.sleep();
        ros::spinOnce();
        lreloc->run();
    }
    ros::spin();
}

void lreloc_node::reset(){

}
