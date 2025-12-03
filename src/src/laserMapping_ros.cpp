#include "laserMapping_ros.h"

#include <std_msgs/Float64.h>

#include "common/offset_timer.h"

ros::Subscriber sub_pcl;
ros::Subscriber sub_imu ;
ros::Publisher pubOffsetTs;
ros::Publisher pubLaserCloudFull;
ros::Publisher pubLaserCloudFull_body ;
ros::Publisher pubLaserCloudEffect;
ros::Publisher pubLaserCloudMap;
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

deque<double>                     temp_time_buffer;
deque<PointCloudXYZI::Ptr>        temp_lidar_buffer;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

mutex mtx_buffer;

std::ofstream odom_file;
std::atomic<bool>  odom_file_initialized = {false};

bool   scan_pub_en = true, dense_pub_en = false, scan_body_pub_en = true;
int lidar_type;
double lidar_d;

OffsetTimer* OffsetTimerIns() {
  static OffsetTimer offset_timer("lidar");
  return &offset_timer;
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

template<typename T>
void set_basepose(T & out)
{ 
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

    // Eigen::Quaterniond current_quat(geoQuat.w, geoQuat.x, geoQuat.y, geoQuat.z);

    // M3D current_rot = current_quat.toRotationMatrix();
    // M3D total_rot = total_rot_ * current_rot;

    // Eigen::Quaterniond quat(total_rot);
    V3D pos(state_point.pos(0), state_point.pos(1), state_point.pos(2));
    V3D rotated_pos = total_rot_ * pos; 
    out.pose.position.x = rotated_pos(0);
    out.pose.position.y = rotated_pos(1);
    out.pose.position.z = rotated_pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    double roll, pitch, yaw;
    Eigen::Quaterniond q(geoQuat.w, geoQuat.x, geoQuat.y, geoQuat.z);
    tf::Matrix3x3(tf::Quaternion(q.x(), q.y(), q.z(), q.w())).getRPY(roll, pitch, yaw);
    // cout << "/as_lio/lio ===>" <<"roll: " << roll << " pitch: " << pitch << " yaw: " << yaw << endl;
}

void set_geoQuat()
{
    geoQuat.x = state_point.rot.coeffs()[0];
    geoQuat.y = state_point.rot.coeffs()[1];
    geoQuat.z = state_point.rot.coeffs()[2];
    geoQuat.w = state_point.rot.coeffs()[3];
}

void downSizeFilter(){
    downSizeFilterSurf.setInputCloud(feats_undistort);
    downSizeFilterSurf.filter(*feats_down_body);
}

#pragma region 发布点云

// PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

bool init_flag = false;
Eigen::Matrix4f pose_tf = Eigen::Matrix4f::Identity();

/**
 * @brief 发布激光雷达点云数据到ROS话题，并可选择性地保存为PCD文件
 * 
 * 该函数主要完成以下功能：
 * 1. 如果使能(scan_pub_en为true)，将去畸变或降采样后的点云转换到世界坐标系并发布；
 * 2. 如果使能地图保存(pcd_save_en为true)，则将点云累积并定期保存为PCD文件。
 * 
 * @param[in] pubLaserCloudFull ROS发布者，用于发布转换后的点云消息
 */
void publish_frame_world()
{
    // 如果允许发布点云数据
    if(scan_pub_en)
    {
        // cout << "publish frame at: " << static_cast<long long>(lidar_end_time * 1e6) << " microseconds" << endl;

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
void publish_frame_body( )
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

void publish_effect_world( )
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
void publish_map()
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
void publish_path( )
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
 * @brief 发布里程计信息，并广播对应的TF变换
 * 
 * 该函数用于将处理后的里程计数据发布到ROS系统中，并通过TF广播坐标变换。
 * 里程计的位姿由`set_posestamp`函数设置，同时会从卡尔曼滤波器中获取协方差信息填充到消息中。
 * 最后，使用TF库广播从"camera_init"到"body"的坐标变换。
 *
 * @param[in] pubOdomAftMapped 用于发布里程计消息的ROS发布者对象
 */
void publish_odometry()
{
    // 设置里程计消息的帧ID和时间戳
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    odomAftMappedBase.header = odomAftMapped.header;
    odomAftMappedBase.child_frame_id = "body_base";
    odomAftMappedBase.header.stamp = ros::Time().fromSec(lidar_end_time);

    // 填充位姿信息
    set_posestamp(odomAftMapped.pose);
     // 添加保存到txt文件的代码
    if (odom_file.is_open() && odom_file_initialized.load()) {
        odom_file << std::fixed << std::setprecision(9) 
                  << lidar_end_time << ","
                  << odomAftMapped.pose.pose.position.x << ","
                  << odomAftMapped.pose.pose.position.y << ","
                  << odomAftMapped.pose.pose.position.z << ","
                  << odomAftMapped.pose.pose.orientation.x << ","
                  << odomAftMapped.pose.pose.orientation.y << ","
                  << odomAftMapped.pose.pose.orientation.z << ","
                  << odomAftMapped.pose.pose.orientation.w << ","
                  << odomAftMapped.twist.twist.linear.x << ","
                  << odomAftMapped.twist.twist.linear.y << ","
                  << odomAftMapped.twist.twist.linear.z << ","
                  << odomAftMapped.twist.twist.angular.x << ","
                  << odomAftMapped.twist.twist.angular.y << ","
                  << odomAftMapped.twist.twist.angular.z
                  << std::endl;
    }

    // 发布里程计消息
    pubOdomAftMapped.publish(odomAftMapped);
    set_basepose(odomAftMappedBase.pose);

    pubOdomAftMappedBase.publish(odomAftMappedBase);
    auto P = kf.get_P();
    // 从卡尔曼滤波器获取协方差矩阵，并重新排列以适配ROS标准格式
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


/*
* @brief 发布所有需要发布的信息
*/
void publish_this(){
    if (path_en)                         publish_path();
    if ((scan_pub_en || pcd_save_en) )      publish_frame_world();
    if (scan_pub_en && scan_body_pub_en) publish_frame_body();
    // publish_effect_world(pubLaserCloudEffect);
    // publish_map(pubLaserCloudMap);
}


#pragma endregion
void init_subAndpub(ros::NodeHandle &nh){
    sub_pcl = nh.subscribe(lid_topic, 5, standard_pcl_cbk);
    sub_imu = nh.subscribe(imu_topic, 5, imu_cbk);
    pubOffsetTs = nh.advertise<std_msgs::Float64>("/as_lio/offset_ts", 1);
    pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
        ("/as_lio/cloud_registered", 3);
    pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
        ("/as_lio/cloud_registered_body", 3);
    pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
        ("/as_lio/cloud_effected", 3);
    pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
        ("/as_lio/Laser_map", 3);
    pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
        ("/as_lio/org_lio", 3);
    pubOdomAftMappedBase = nh.advertise<nav_msgs::Odometry> 
        ("/as_lio/lio", 3);
    pubPath          = nh.advertise<nav_msgs::Path> 
        ("/as_lio/path", 3);
    serv_ctrl_mapping = nh.advertiseService("/as_lio/ctrl", ctrl_mapping_cbk);
    serv_save_mapping = nh.advertiseService("/as_lio/savemap", save_map_cbk);

    OffsetTimerIns()->Hello();
}
void init_param(ros::NodeHandle &nh)
{ 
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
    bool save_map_this = false;
    nh.param<bool>("pcd_save/save_map", save_map_this, false);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<double>("lidar_d", lidar_d, 0.0);

    save_map.store(save_map_this);
    // 初始化点选择和残差数组
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    p_pre->lidar_type = lidar_type;
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

}
double timediff_lidar_wrt_imu = 0.0;

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
    double msg_ts = msg_in->header.stamp.toSec();
    double sys_ts = ros::Time::now().toSec();
    OffsetTimerIns()->FeedEmb_ts(sys_ts, msg_ts);

    {
        static double pre_ts = 0;
        if (pre_ts + 1.0 < sys_ts) {
            pre_ts = sys_ts;

            double dts = OffsetTimerIns()->GetEmb_dt();
            if (dts > 0.0) {
                std_msgs::Float64 msg;
                msg.data = OffsetTimerIns()->GetEmb_dt();
                pubOffsetTs.publish(msg);   
            }
        }
    }
    
    if(is_running_.load() == 1){
        static double pre_ts = 0.0;
        double cur_ts = msg_in->header.stamp.toSec();
        if (cur_ts < pre_ts) {
            ROS_WARN("Preprocess::WLR722F_handler(): lio_imu time Jump back , dts=%.3f, cur_ts=%.3f, pre_ts=%.3f", cur_ts - pre_ts, cur_ts, pre_ts);
            return;
        }
        if (pre_ts > 0.0 && (cur_ts > pre_ts + 0.2)) {
            ROS_WARN("Preprocess::WLR722F_handler(): lio_imu time jump is large, dts=%.3f, cur_ts=%.3f, pre_ts=%.3f", cur_ts - pre_ts, cur_ts, pre_ts);
        }
        pre_ts = cur_ts;
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
}


bool   timediff_set_flg = false;

/**
 * @brief 标准点云回调函数，处理来自激光雷达的点云数据
 * @param msg 输入的点云数据消息指针
 * 
 * 该函数负责接收激光雷达点云数据，进行预处理并存储到缓冲区中。
 * 主要功能包括：时间戳检查、点云预处理、数据缓冲管理等。
 */
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
   if(is_running_.load() == 1){
            static double pre_ts = 0.0;
        double cur_ts = msg->header.stamp.toSec();
        if (cur_ts < pre_ts) {
            ROS_WARN("Preprocess::WLR722F_handler(): lio_imu time Jump back , dts=%.3f, cur_ts=%.3f, pre_ts=%.3f", cur_ts - pre_ts, cur_ts, pre_ts);
            return;
        }
        if (pre_ts > 0.0 && (cur_ts > pre_ts + 0.5)) {
            ROS_WARN("Preprocess::WLR722F_handler(): lio_imu time jump is large, dts=%.3f, cur_ts=%.3f, pre_ts=%.3f", cur_ts - pre_ts, cur_ts, pre_ts);
        }
        static std::atomic<bool> odd_number = {true};
        bool flag = true;
        if (odd_number.compare_exchange_strong(flag, false))
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
            lidar_buffer.push_back(ptr);
            time_buffer.push_back(msg->header.stamp.toSec());
            
            last_timestamp_lidar = msg->header.stamp.toSec();
            
            // 记录预处理耗时
            s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
            // cout<<"s_plot11[scan_count]: "<<s_plot11[scan_count]<<endl;
            // 解锁并通知等待的线程
            mtx_buffer.unlock();
            sig_buffer.notify_all();
        }else{
            odd_number.store(true);
        }
   }
}

