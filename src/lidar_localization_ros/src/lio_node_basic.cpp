#include "lio_node.h"

lioNode::lioNode(ros::NodeHandle &nh):accumulated_cloud(new PointCloudXYZI()){
    translation_body.resize(3,0.0);
    lio_helper = std::make_shared<LioHelper>();
    p_pre = std::make_shared<Preprocess>();
    init_param(nh);
    init_subAndpub(nh);
    lio_controller = std::make_shared<LioCode>(lio_helper);
    cout<<"lio_helper init"<<endl;
}
lioNode::~lioNode(){
    ROS_INFO("lioNode::~lioNode()");
    if(odom_file.is_open()){
        odom_file.close();
    }
    if(lio_controller)     {
        lio_controller->stopAlgorithm();
        lio_controller.reset();
    } 
    if(p_pre)               p_pre.reset();
    if(lio_helper)          lio_helper.reset();
}
bool lioNode::init(){
    bool res = false;
    if(!lio_controller)     lio_controller = std::make_shared<LioCode>(lio_helper);

    if(lio_helper){
        lio_helper->regPubOdomCallback(std::bind(&lioNode::publish_odometry,this));
        lio_helper->regPubPointCloudCallback(std::bind(&lioNode::publish_this,this));
        lio_helper->regSaveMapPointCallback(std::bind(&lioNode::save_map_point, this, std::placeholders::_1, std::placeholders::_2),
                                    std::bind(&lioNode::clear_map_point,this));
        lio_helper->regSetGeoQuatCallback(std::bind(&lioNode::set_geoQuat,this,std::placeholders::_1));
        res=true;
    }
    return res;
}
void lioNode::start(){
  lio_controller->start();
}

void lioNode::reset(){
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";
}

void lioNode::init_subAndpub(ros::NodeHandle &nh){
    sub_pcl = nh.subscribe(lid_topic, 5, &lioNode::standard_pcl_cbk,this);
    sub_imu = nh.subscribe(imu_topic, 5, &lioNode::imu_cbk,this);
    pubOffsetTs = nh.advertise<std_msgs::Float64>("/as_lio/offset_ts", 1);
    pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
        ("/as_lio/cloud_registered", 3);
    pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
        ("/as_lio/cloud_registered_body", 3);
    pubCloud_body = nh.advertise<sensor_msgs::PointCloud2>
        ("/as_lio/cloud_registered_body_obstacle", 3);
    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
        ("/as_lio/org_lio", 3);
    pubOdomAftMappedBase = nh.advertise<nav_msgs::Odometry> 
        ("/as_lio/lio", 3);
    pubPath          = nh.advertise<nav_msgs::Path> 
        ("/as_lio/lio_path", 3);
    serv_ctrl_mapping = nh.advertiseService("/as_lio/ctrl", &lioNode::ctrl_mapping_cbk,this);
    serv_save_mapping = nh.advertiseService("/as_lio/savemap", &lioNode::save_map_cbk,this);

    OffsetTimerIns()->Hello();
}
void lioNode::init_param(ros::NodeHandle &nh)
{ 
    // 从参数服务器加载配置参数
    bool path_en_= true;
    nh.param<bool>("publish/path_en",path_en_, true);
    path_en = path_en_;
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    int num_max_iterations = 0;
    nh.param<int>("max_iteration",num_max_iterations,4);
    lio_helper->num_max_iterations = num_max_iterations;
    nh.param<string>("common/lid_topic",lid_topic,"/vanjee_points722f");
    nh.param<string>("common/imu_topic", imu_topic,"/vanjee_lidar_imu_packets");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_surf",lio_helper->filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",lio_helper->filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",lio_helper->cube_len,200);
    nh.param<float>("mapping/det_range",lio_helper->DET_RANGE,300.f);
    nh.param<double>("mapping/gyr_cov", lio_helper->gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov", lio_helper->acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov", lio_helper->b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov", lio_helper->b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    int lidar_type = 5;
    nh.param<int>("preprocess/lidar_type", lidar_type, AVIA); //AVIA
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2); //2
    p_pre->lidar_type = lidar_type;
    nh.param<bool>("mapping/extrinsic_est_en", lio_helper->extrinsic_est_en, true);
    bool save_map_this = false;
    nh.param<bool>("pcd_save/save_map", save_map_this, false);
    nh.param<vector<double>>("mapping/extrinsic_T", lio_helper->extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", lio_helper->extrinR, vector<double>());
    nh.param<vector<double>>("mapping/translation_body", translation_body, vector<double>());
    nh.param<double>("lidar_d", lidar_d, 0.0);

    save_map.store(save_map_this);

    lio_helper->p_imu->lidar_type = lidar_type;
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="map";
    
    lio_helper->init_imu_extrin();

}

