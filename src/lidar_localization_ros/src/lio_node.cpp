#include "lio_node.h"

void lioNode::set_geoQuat(state_ikfom& state_point)
{
    geoQuat.x = state_point.rot.coeffs()[0];
    geoQuat.y = state_point.rot.coeffs()[1];
    geoQuat.z = state_point.rot.coeffs()[2];
    geoQuat.w = state_point.rot.coeffs()[3];
}
/**
 * @brief 发布激光雷达点云数据到ROS话题  重定位使用
 * 
 * 该函数主要完成以下功能：
 * 1. 如果使能(scan_pub_en为true)，将去畸变或降采样后的点云转换到世界坐标系并发布；
 * 
 * @param[in] pubLaserCloudFull ROS发布者，用于发布转换后的点云消息
 */
void lioNode::publish_frame_world(){
    // 根据dense_pub_en标志选择使用去畸变点云还是降采样点云
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? lio_helper->feats_undistort : lio_helper->feats_down_body);
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
            lio_helper->RGBpointBodyToWorld(&laserCloudFullRes->points[i], 
                                &laserCloudWorld->points[i]);
        }
    }

    // 构造ROS点云消息并发布
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lio_helper->lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFull.publish(laserCloudmsg);
}


/**
 * @brief 发布经过IMU坐标系变换的激光点云数据   
 * 
 * 该函数将去畸变后的激光点云数据从雷达坐标系转换到IMU坐标系，
 * 然后发布到ROS话题中供其他节点使用。
 * 
 * @param pubLaserCloudFull_body ROS发布者对象，用于发布转换后的点云数据
 */
void lioNode::publish_frame_body( ){
    // 获取去畸变后点云数据的大小
    int size = lio_helper->feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    // 将雷达坐标系下的点云转换到IMU坐标系下
    for (int i = 0; i < size; i++)
    {
        lio_helper->RGBpointBodyLidarToIMU(&lio_helper->feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    // 将点云数据转换为ROS消息格式并发布
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg); 
    laserCloudmsg.header.stamp = ros::Time().fromSec(lio_helper->lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
}

/**
 * @brief 发布经过IMU坐标系变换的激光点云数据   避障使用
 * 
 * 
 * @param pubLaserCloudFull_body ROS发布者对象，用于发布转换后的点云数据
 */

void lioNode::publish_body(){
    // 获取去畸变后点云数据的大小
    int size = lio_helper->feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        lio_helper->RGBpointBodyLidarToIMU(&lio_helper->feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i],lidar_d,translation_body);
    }
    if (laserCloudIMUBody->points.empty() || laserCloudIMUBody->points.size() == 0) {
        ROS_WARN("No point to publish in body frame!");
        return;
    }
    // 将点云数据转换为ROS消息格式并发布
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg); 
    laserCloudmsg.header.stamp = ros::Time().fromSec(lio_helper->lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubCloud_body.publish(laserCloudmsg);
}

void lioNode::publish_path( )
{
    set_basepose(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.frame_id = "map";

    /*** if path is too large, the rvis will crash ***/
    static int count = 0;
    static int jjj = 0;
    ++count;
    ++jjj;
    if(count >= 5000){
        --count;
        path.poses.erase(path.poses.begin());  
        path.poses.push_back(msg_body_pose);
    }else{
        path.poses.push_back(msg_body_pose);
    }

    if(jjj % 5 == 0){
        pubPath.publish(path);
        jjj = 0;
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
void lioNode::publish_odometry()
{
    // 设置里程计消息的帧ID和时间戳
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lio_helper->lidar_end_time);
    odomAftMappedBase.header = odomAftMapped.header;
    odomAftMappedBase.child_frame_id = "body_base";
    odomAftMappedBase.header.stamp = ros::Time().fromSec(lio_helper->lidar_end_time);
    odomAftMappedBase_offset.header = odomAftMapped.header;
    odomAftMappedBase_offset.child_frame_id = "body_base_offset";
    odomAftMappedBase_offset.header.stamp = ros::Time().fromSec(lio_helper->lidar_end_time + OffsetTimer_double.load());
    // 发布里程计消息
        set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    set_basepose(odomAftMappedBase.pose);

    pubOdomAftMappedBase.publish(odomAftMappedBase);

    set_basepose(odomAftMappedBase_offset.pose);
    pubOdomAftMappedBase_offset.publish(odomAftMappedBase_offset);


        // printf("Publishing base pose: x=%.6f, y=%.6f, z=%.6f\n",
        //    odomAftMappedBase.pose.pose.position.x,
        //    odomAftMappedBase.pose.pose.position.y,
        //    odomAftMappedBase.pose.pose.position.z);
     // 添加保存到txt文件的代码
    if (odom_file.is_open() && odom_file_initialized.load()) {
        odom_file << std::fixed << std::setprecision(9) 
                  << lio_helper->lidar_end_time << ","
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

      auto P =lio_helper->kf.get_P();

    double pos_cov = (P(0,0) + P(1,1) + P(2,2)) / 3.0;
    double rot_cov = (P(3,3) + P(4,4) + P(5,5)) / 3.0;

    if (pos_cov > 1.0 || rot_cov > 1.0) {
        ROS_WARN("High covariance detected - system may be diverging!");
        // check_and_apply_constraints();
    }
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
void lioNode::publish_this(){
    if (path_en)                         publish_path();
    if (scan_pub_en )      publish_frame_world();
    if (scan_pub_en && scan_body_pub_en) publish_frame_body();
    publish_body();
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
void lioNode::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
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
                OffsetTimer_double.store(OffsetTimerIns()->GetEmb_dt());
                // sensor_msgs::Imu::Ptr imu_msg(new sensor_msgs::Imu(*msg_in));
                // imu_msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() + msg.data);
                // pub_imu_offset_ts.publish(*imu_msg);
	      	    pubOffsetTs.publish(msg);   
            }
        }
    }
    
    if(lio_controller->get_is_running() == 1){
        static double pre_ts = 0.0;
        double cur_ts = msg_in->header.stamp.toSec();
        if (cur_ts < pre_ts) {
            ROS_WARN("Preprocess::WLR722F_handler(): lio_imu time Jump back , dts=%.3f, cur_ts=%.3f, pre_ts=%.3f", cur_ts - pre_ts, cur_ts, pre_ts);
            lio_helper->imu_buffer.clear();
            return;
        }
        if (pre_ts > 0.0 && (cur_ts > pre_ts + 0.2)) {
            ROS_WARN("Preprocess::WLR722F_handler(): lio_imu time jump is large, dts=%.3f, cur_ts=%.3f, pre_ts=%.3f", cur_ts - pre_ts, cur_ts, pre_ts);
        }
        pre_ts = cur_ts;
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
        lio_helper->mtx_buffer.lock();

        lio_helper->last_timestamp_imu = timestamp;

        // 将处理后的IMU消息添加到缓冲区
        lio_helper->imu_buffer.push_back(msg);
        lio_helper->mtx_buffer.unlock();
        
        // 通知所有等待缓冲区的线程
        lio_controller->sig_notify();
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
void lioNode::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
   if(lio_controller->get_is_running() == 1){
        static double pre_ts = 0.0;
        double cur_ts = msg->header.stamp.toSec();
        if (cur_ts < pre_ts) {
            ROS_WARN("Preprocess::WLR722F_handler(): lio_imu time Jump back , dts=%.3f, cur_ts=%.3f, pre_ts=%.3f", cur_ts - pre_ts, cur_ts, pre_ts);
            lio_helper->clear();
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
            lio_helper->mtx_buffer.lock();
            double preprocess_start_time = omp_get_wtime();
            
            // cout << "Debug: File=" << __FILE__ << ", Line=" << __LINE__ << ", Function=" << __FUNCTION__ << endl;
            // 创建新的点云对象并进行预处理
            PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
            p_pre->process(msg, ptr);
            // 将处理后的点云数据和时间戳存入缓冲区
            lio_helper->inset_lidar_buffer(ptr);
            lio_helper->time_buffer.push_back(msg->header.stamp.toSec());
            
            // 解锁并通知等待的线程
            lio_helper->mtx_buffer.unlock();
            lio_controller->sig_notify();
        }else{
           odd_number.store(true);
        }
   }
}


/**
 * @brief 保存点云地图的回调函数，用于在用户按下Ctrl+S键时保存点云地图。
 * 
 * @param msg 保存地图的标志位，true表示保存地图，false表示取消保存地图。
 */
const std::string k_meta_map_fn = "meta_map.txt";
bool lioNode::save_map_cbk(mower_msgs::TriggerRequest &req,mower_msgs::TriggerResponse &res){
    ROS_INFO("===================saveMap_cbk===================== ");
    if(!req.arg.empty()){
        ROS_INFO("save_map_cbk(!req.arg.empty()) ++++++");
        odom_file_initialized.store(false);
        if(odom_file.is_open()){
            odom_file.close();
        }
        const std::string vmap_version = "V1";
        string map_path = string(ROOT_DIR) + req.arg ;
        if (map_path.back() != '/')
            map_path += "/";

        std::string lmap_path = map_path + "lmap/";
         struct stat info;
        if (!IsDirExisting(lmap_path.c_str())) {
            CreateDir(lmap_path.c_str());
        }
        if (accumulated_cloud->size() > 0 ) {
            bool expected = true;
            save_map.compare_exchange_strong(expected,false);
            
            const std::string new_sm_name = GetCurTimeStamp_Sec();

            string save_map_path = lmap_path + new_sm_name+ ".pcd";
            printf("saveMap(): %s",save_map_path.c_str());
                        
            string all_points_dir(save_map_path);
            pcl::PCDWriter pcd_writer;
            pcd_writer.writeBinary(all_points_dir, *accumulated_cloud);

            std::map<std::string, std::string> submaps;
            // 先备份已存meta_map.txt
            std::string meta_map_fn = lmap_path + k_meta_map_fn;
           
            if (!IsDirExisting(meta_map_fn.c_str())) {
                std::ifstream fin(meta_map_fn);
                std::string tmp_version, tmp_name;
                while (fin >> tmp_name >> tmp_version) {
                    if (tmp_version.size() == 2 && tmp_name.size() == 15) {
                        submaps[tmp_name] = tmp_version;
                    } else {
                        break;
                    }
                }
                fin.close();
            }
            submaps[new_sm_name] = vmap_version;
            {
                std::ofstream fout(meta_map_fn);
                for (const auto &it : submaps) {
                fout << it.first << " " << it.second << std::endl;
                ROS_INFO("saveMap(): name=%s, version=%s", it.first.c_str(), it.second.c_str());
                }
                fout.close();
                ROS_INFO("saveMap() done");
            }


            res.result = 1;
            res.message = "Map saved to %s ",save_map_path.c_str() ;
            ROS_INFO( "Map saved to %s", save_map_path.c_str());
            accumulated_cloud->clear();
        }else{
            res.result = 0;
            res.message = "No points to save or map not initialized";
        }
    }else{
        res.result = 0;
        res.message = "The path is empty. Invalid map save request";
    }
    
    return true;
}

bool lioNode::ctrl_mapping_cbk(mower_msgs::TriggerRequest &req,mower_msgs::TriggerResponse &res){
    if(req.arg =="reset_lio" ){
        ROS_INFO("ctrl_mapping_cbk(reset_lio) Resetting LIO ++++++");
        odom_file_initialized.store(false);
        lio_controller->set_is_running(2);   //重置算法
        if(odom_file.is_open()){
            odom_file.close();
        }
        res.result = 1;
        res.message = "LIO reset.";
        ROS_INFO_STREAM(res.message);

    }else if(req.arg =="start_mapping" ){
        ROS_INFO("ctrl_mapping_cbk(start_mapping) start_mapping   ++++++");
        save_map.store(true);
        res.result = 1;
        res.message = "Started mapping";

        // 初始化里程计文件
        if (!odom_file_initialized.load()) {
            odom_file.open(string(string(ROOT_DIR) + "/odometry_data.txt").c_str());
            if (odom_file.is_open()) {
                // 写入表头
                odom_file << "timestamp,x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz" << std::endl;
                odom_file_initialized.store(true);
            }
        }
    }else if(req.arg =="stop_mapping"){
        save_map.store(false);
        res.result = 1;
        res.message = "Stopped mapping";
    }else if(req.arg == "open_lio"){
        if(lio_controller->get_is_running() == 0){         //
            lio_controller->set_is_running(1);
        }

        res.result = 1;
        res.message = "LIO started successfully.";
        ROS_INFO_STREAM(res.message);
    }else if(req.arg == "close_lio"){
        lio_controller->set_is_running(2);
        res.result = 1;
        res.message = "LIO stopped and resources released.";
        ROS_INFO_STREAM(res.message);
    }
    return true;
}

void lioNode::save_map_point(vector<PointType, Eigen::aligned_allocator<PointType>>  PointToAdd,
                    vector<PointType, Eigen::aligned_allocator<PointType>>  PointNoNeedDownsample){ 
    if(save_map.load()){
        {
            std::lock_guard<std::mutex> lock(accumulated_cloud_mutex);
            for (const auto& point : PointToAdd) {
                accumulated_cloud->push_back(point);
            }
            for (const auto& point : PointNoNeedDownsample) {
                accumulated_cloud->push_back(point);
            }
        }
    }
}

void lioNode::clear_map_point(){
    if (accumulated_cloud) {
        accumulated_cloud->clear();
    }
    save_map.store(false);
}


