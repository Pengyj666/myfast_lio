#include "lreloc_node.h"

Eigen::Matrix4f poseToMatrix(const nav_msgs::Odometry& odom_msg) {
    Eigen::Affine3d affine;
    tf::poseMsgToEigen(odom_msg.pose.pose, affine);
    return affine.matrix().cast<float>();
}


void lreloc_node::cbSaveCurOdom(const nav_msgs::OdometryConstPtr& odom_msg) {
    std::lock_guard<std::mutex> lock(cur_odom_mutex);
    Eigen::Matrix4f T_odom_to_base_link = poseToMatrix(*odom_msg);
    cur_odom_queue.emplace_back(T_odom_to_base_link,odom_msg->header.stamp.toSec());
    lreloc->setOdomReceived(true);
}

void lreloc_node::callback_lio_offset_ts(const std_msgs::Float64::ConstPtr &msg) {
    lreloc->setOffsetTs(msg->data);
}

void lreloc_node::cbSaveCurScan(const sensor_msgs::PointCloud2ConstPtr& pc_msg) {
    if(!lreloc->getCalculating() && lreloc->get_isLoadMap() == 2){
        sensor_msgs::PointCloud2 modified_msg = *pc_msg;
        modified_msg.header.frame_id = "camera_init";
        modified_msg.header.stamp = ros::Time::now();
        pub_pc_in_map.publish(modified_msg);

        pcl::fromROSMsg(*pc_msg, *cur_scan);

        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cur_scan, *cur_scan, indices);
        lreloc->set_cur_scan(cur_scan);
        lreloc->setCurScanTime(pc_msg->header.stamp.toSec());
        lreloc->setScanReceived(true);
    }
}

void lreloc_node::initialPoseCallback(const nav_msgs::OdometryConstPtr& msg) {
    std::lock_guard<std::mutex> lock(initial_pose_mutex);
    if(lreloc->get_isLoadMap() == 2 ){
        Eigen::Matrix4f initial_pose = Eigen::Matrix4f::Identity();
    
        nav_msgs::Odometry rotated_odom = *msg;
        
        double lidar_Radian = lidar_d*M_PI/180;
        Eigen::Matrix3d calibrateTilt_X;
        Eigen::Matrix3d calibrateTilt_Z;
        calibrateTilt_X << 1, 0, 0,
                        0, cos(lidar_Radian), sin(lidar_Radian),
                        0, -sin(lidar_Radian),cos(lidar_Radian);

        calibrateTilt_Z << 0, -1, 0,
                    1, 0, 0,
                    0, 0, 1;
        Eigen::Matrix3d total_rot = calibrateTilt_Z * calibrateTilt_X;

        Eigen::Vector3d original_pos(msg->pose.pose.position.x, 
                                    msg->pose.pose.position.y, 
                                    msg->pose.pose.position.z);

        Eigen::Vector3d rotated_pos = total_rot.transpose() * original_pos;

        rotated_odom.pose.pose.position.x = rotated_pos.x();
        rotated_odom.pose.pose.position.y = rotated_pos.y();
        rotated_odom.pose.pose.position.z = rotated_pos.z();
    
        initial_pose = poseToMatrix(rotated_odom);
        initial_pose(0, 3) += 0.428615;
        initial_pose(1, 3) -= 0.012560;
        initial_pose(2, 3) += 0.198539;
        
        static double temp = 28896694.16;

        {
            std::lock_guard<std::mutex> lock(cur_odom_mutex);
            if(cur_odom_queue.size() == 0){
                ROS_WARN("No odom data received");
                return;
            }
            Eigen::Matrix4f quat;

            int idx = cur_odom_queue.findAfter(lreloc->getCurScanTime());

            if (idx >= 0 ) {
                quat = cur_odom_queue[idx];
            }else{
                quat = cur_odom_queue[0];
            }

            Eigen::Matrix4f pose_map_to_odom = (initial_pose.inverse() * quat).eval();
            lreloc->insert_initial_odom_queue(pose_map_to_odom,msg->header.stamp.toSec() - temp); //offsetTs.load()
            lreloc->setPoseReceived(true);
        }
    }
}

const std::string k_meta_map_fn = "meta_map.txt";
bool lreloc_node::loadMapCallback(mower_msgs::TriggerRequest &req,mower_msgs::TriggerResponse &res){
     cout << "============bool lreloc_node::loadMapCallback==========" << endl;
	if(lreloc->get_isLoadMap() == 0 || lreloc->get_isLoadMap() == 2){     //没有加载地图，或者加载完成再调就重新加载新地图
        string map_path = string(ROOT_DIR) + req.arg  ;
        if (map_path.back() != '/')
            map_path += "/";

        std::string lmap_path = map_path + "lmap/";
        struct stat info;
        if (!!stat(lmap_path.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
            utils::CreateDir(lmap_path.c_str());
        }
        std::map<std::string, std::string> submaps;
        std::string meta_map_fn = lmap_path + k_meta_map_fn;
        if (utils::IsFileExisting(meta_map_fn.c_str())) {
            std::ifstream fin(meta_map_fn);
            std::string tmp_name, tmp_version;
            while (fin >> tmp_name >> tmp_version) {
                if (tmp_name.size() == 15 && tmp_version.size() == 2) {
                    std::string map_file_path = lmap_path + tmp_name + ".pcd";
                    lreloc->setMapFilePath(map_file_path);
                    lreloc->set_isLoadMap(1);
                    submaps[tmp_name] = tmp_version;
                }
            }
            fin.close();
            if (submaps.size() == 0) {
                res.result = false;
                res.message = "lreloc_node::loadMapCallback(): No existing map found at: %s", lmap_path.c_str();
                return true;
            }
        } else {
            res.result = false;
            res.message = "No existing map found at: " + lmap_path;
            ROS_INFO("No existing map found at: %s", lmap_path.c_str());
            return true;
        }
        res.result = true;
        res.message = "Map loaded successfully.";
        return true;
    }else{
        res.result = false;
        res.message = "The map is already loaded.";
        ROS_INFO("The map is already loaded.");
        return true;
    }
}


void lreloc_node::threadOdomPath(const std::string& file_path) {
        cout << "threadOdomPath++++++++++++++ " << std::endl;
    std::ifstream file(file_path);
        cout << "file(file_path) " << file_path << std::endl;
    if (!file.is_open()) {
        ROS_ERROR("Could not open file: %s", file_path.c_str());
        return;
    }
    nav_msgs::Path path_msg;
    path_msg.header.frame_id = "map";

    std::string line;
    // 跳过标题行
    std::getline(file, line);
    cout << "Reading file: " << file_path << std::endl;
    // 读取第一行数据
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        double timestamp, x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz;
        char comma; // 用于读取逗号分隔符

        // 按照逗号分隔符读取数据
        ss >> timestamp >> comma 
        >> x >> comma 
        >> y >> comma 
        >> z >> comma 
        >> qx >> comma 
        >> qy >> comma 
        >> qz >> comma 
        >> qw >> comma 
        >> vx >> comma 
        >> vy >> comma 
        >> vz >> comma 
        >> wx >> comma 
        >> wy >> comma 
        >> wz;

        // 检查数据有效性
        if (std::isnan(x) || std::isnan(y) || std::isnan(z) ||
            std::isnan(qx) || std::isnan(qy) || std::isnan(qz) || std::isnan(qw) ||
            std::isinf(x) || std::isinf(y) || std::isinf(z) ||
            std::isinf(qx) || std::isinf(qy) || std::isinf(qz) || std::isinf(qw)) {
            ROS_WARN("Invalid odometry data detected");
            return;
        }
        double lidar_Radian = lidar_d*M_PI/180;
        Eigen::Matrix3d calibrateTilt_X;
        Eigen::Matrix3d calibrateTilt_Z;
        calibrateTilt_X << 1, 0, 0,
                        0, cos(lidar_Radian), sin(lidar_Radian),
                        0, -sin(lidar_Radian), cos(lidar_Radian);
        calibrateTilt_Z << 0, -1, 0,
                        1, 0, 0,
                        0, 0, 1;
        Eigen::Matrix3d total_rot = calibrateTilt_Z * calibrateTilt_X;

        Eigen::Vector3d original_pos(x, y, z);
        Eigen::Vector3d rotated_pos = total_rot * original_pos;
        x = rotated_pos.x();
        y = rotated_pos.y();
        z = rotated_pos.z();
        
        // // 创建并发布里程计消息
        nav_msgs::Odometry odom_msg;
        odom_msg.pose.pose.position.x = x;
        odom_msg.pose.pose.position.y = y;
        odom_msg.pose.pose.position.z = z;
        odom_msg.pose.pose.orientation.x = qx;
        odom_msg.pose.pose.orientation.y = qy;
        odom_msg.pose.pose.orientation.z = qz;
        odom_msg.pose.pose.orientation.w = qw;
        
        odom_msg.twist.twist.linear.x = vx;
        odom_msg.twist.twist.linear.y = vy;
        odom_msg.twist.twist.linear.z = vz;
        odom_msg.twist.twist.angular.x = wx;
        odom_msg.twist.twist.angular.y = wy;
        odom_msg.twist.twist.angular.z = wz;
    

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time::now();
        pose_stamped.header.frame_id = "map";
        pose_stamped.pose = odom_msg.pose.pose;
        path_msg.poses.push_back(pose_stamped);
        
        // 更新路径消息时间戳并发布
        path_msg.header.stamp = ros::Time::now();
        path_pub.publish(path_msg);


        ros::Duration(0.1).sleep(); 
    }
    
    file.close();
}


void lreloc_node::pub_mapToOdom(Eigen::Matrix4f T_map_to_odom){
    // 发布map_to_odom
    nav_msgs::Odometry map_to_odom;
    Eigen::Affine3f affine(T_map_to_odom);
    tf::poseEigenToMsg(Eigen::Affine3d(affine.cast<double>()), map_to_odom.pose.pose);
    map_to_odom.header.stamp = ros::Time().fromSec(lreloc->getCurScanTime());
    map_to_odom.header.frame_id = "map";
    pub_map_to_odom.publish(map_to_odom);
}
