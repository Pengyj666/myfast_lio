#include "globalLocalization.h"


KD_TREE<PointT> kdtree;
// 参数
double MAP_VOXEL_SIZE = 0.2;
double SCAN_VOXEL_SIZE = 0.1;
double FREQ_LOCALIZATION = 3;
double LOCALIZATION_TH = 0.3;
std::string map_file_path = "";
std::string odom_file_path = "";
std::string g_map_root_dir = "/userdata/RobotData/map/";
double lidar_d = -6.7;
using namespace std;

GlobalLocalization::GlobalLocalization(ros::NodeHandle& nh) : 
    global_map(new pcl::PointCloud<PointT>),
    cur_scan(new pcl::PointCloud<PointT>),
    map_cloud(new pcl::PointCloud<PointT>),
    T_map_to_odom(Eigen::Matrix4f::Identity()) {
        // Publisher
        pub_pc_in_map = nh.advertise<sensor_msgs::PointCloud2>("/cur_scan_in_map", 3);
        pub_submap = nh.advertise<sensor_msgs::PointCloud2>("/submap", 3);
        pub_map_to_odom = nh.advertise<nav_msgs::Odometry>("/as_lio/org_reloc_result", 3);
        
        // Subscriber
        sub_cloud_registered = nh.subscribe<sensor_msgs::PointCloud2>("/as_lio/cloud_registered", 3, &GlobalLocalization::cbSaveCurScan, this);
        sub_odometry = nh.subscribe<nav_msgs::Odometry>("/as_lio/org_lio", 3, &GlobalLocalization::cbSaveCurOdom, this);
        path_pub = nh.advertise<nav_msgs::Path>("/odom_path_test", 3);
        initial_pose_sub = nh.subscribe<nav_msgs::Odometry>("/odom_fused", 3, &GlobalLocalization::initialPoseCallback, this);
        subOffsetTs = nh.subscribe<std_msgs::Float64>("/as_lio/offset_ts", 3, &GlobalLocalization::callback_lio_offset_ts, this);
        // Service
        serv_load_mapping_ = nh.advertiseService("/as_lio/loadmap", &GlobalLocalization::loadMapCallback,this);
         
        initial_odom_queue.reset(2048);
        cur_odom_queue.reset(2048);
    }
GlobalLocalization::~GlobalLocalization() {
    if (odom_path_thread.joinable()) {
        odom_path_thread.join(); // 主线程退出前等待线程完成
    }
}
Eigen::Matrix4f GlobalLocalization::poseToMatrix(const geometry_msgs::PoseWithCovarianceStamped& pose_msg) {
    Eigen::Affine3d affine;
    tf::poseMsgToEigen(pose_msg.pose.pose, affine);
    return affine.matrix().cast<float>();
}

Eigen::Matrix4f GlobalLocalization::poseToMatrix(const nav_msgs::Odometry& odom_msg) {
    Eigen::Affine3d affine;
    tf::poseMsgToEigen(odom_msg.pose.pose, affine);
    return affine.matrix().cast<float>();
}


void GlobalLocalization::threadOdomPath(const std::string& file_path) {
    std::ifstream file(file_path);
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

        // // 输出解析结果
        // std::cout << "Timestamp: " << timestamp << std::endl;
        // std::cout << "Position: (" << x << ", " << y << ", " << z << ")" << std::endl;
        // std::cout << "Orientation: (" << qx << ", " << qy << ", " << qz << ", " << qw << ")" << std::endl;
        // std::cout << "Linear velocity: (" << vx << ", " << vy << ", " << vz << ")" << std::endl;
        // std::cout << "Angular velocity: (" << wx << ", " << wy << ", " << wz << ")" << std::endl;
        
        // 检查数据有效性
        if (std::isnan(x) || std::isnan(y) || std::isnan(z) ||
            std::isnan(qx) || std::isnan(qy) || std::isnan(qz) || std::isnan(qw) ||
            std::isinf(x) || std::isinf(y) || std::isinf(z) ||
            std::isinf(qx) || std::isinf(qy) || std::isinf(qz) || std::isinf(qw)) {
            ROS_WARN("Invalid odometry data detected");
            return;
        }
        
        // // 创建并发布里程计消息
        nav_msgs::Odometry odom_msg;
        // odom_msg.header.stamp = ros::Time::now();//ros::Time(timestamp);
        // odom_msg.header.frame_id = "map";
        // odom_msg.child_frame_id = "body";
        
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

#pragma region 回调函数
void GlobalLocalization::cbSaveCurOdom(const nav_msgs::OdometryConstPtr& odom_msg) {
    std::lock_guard<std::mutex> lock(cur_odom_mutex);
    Eigen::Matrix4f T_odom_to_base_link = poseToMatrix(*odom_msg);
    cur_odom_queue.emplace_back(T_odom_to_base_link,odom_msg->header.stamp.toSec());
    odom_received.store(true);
}

void GlobalLocalization::callback_lio_offset_ts(const std_msgs::Float64::ConstPtr &msg) {
    offsetTs.store(msg->data);
}
void GlobalLocalization::cbSaveCurScan(const sensor_msgs::PointCloud2ConstPtr& pc_msg) {
    if(!should_initialize.load() && is_load_map.load() == 2){
        sensor_msgs::PointCloud2 modified_msg = *pc_msg;
        modified_msg.header.frame_id = "camera_init";
        modified_msg.header.stamp = ros::Time::now();
        pub_pc_in_map.publish(modified_msg);

        pcl::fromROSMsg(*pc_msg, *cur_scan);

        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cur_scan, *cur_scan, indices);
        cur_scan_time = pc_msg->header.stamp.toSec();
        scan_received.store(true);
    }
}

void GlobalLocalization::initialPoseCallback(const nav_msgs::OdometryConstPtr& msg) {
    std::lock_guard<std::mutex> lock(initial_pose_mutex);
    if(is_load_map.load() == 2){
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
        Matrix3d total_rot = calibrateTilt_Z * calibrateTilt_X;

        Eigen::Vector3d original_pos(msg->pose.pose.position.x, 
                                    msg->pose.pose.position.y, 
                                    msg->pose.pose.position.z);

        Eigen::Vector3d rotated_pos = total_rot.transpose() * original_pos;

        rotated_odom.pose.pose.position.x = rotated_pos.x();
        rotated_odom.pose.pose.position.y = rotated_pos.y();
        rotated_odom.pose.pose.position.z = rotated_pos.z();
    
        initial_pose = poseToMatrix(rotated_odom);
        // initial_pose(0, 3) += 0.428615;
        // initial_pose(1, 3) -= 0.012560;
        // initial_pose(2, 3) += 0.198539;
        
        static double temp = 28896694.16;
        // if(T_map_to_odom == Eigen::Matrix4f::Identity()){
        //     initial_odom_queue.emplace_back(initial_pose,msg->header.stamp.toSec() - temp);
        //     pose_received.store(true);
        // }else{
            {
                std::lock_guard<std::mutex> lock(cur_odom_mutex);
                if(cur_odom_queue.size() == 0){
                    ROS_WARN("No odom data received");
                    return;
                }
                Eigen::Matrix4f quat;

                int idx = cur_odom_queue.findAfter(cur_scan_time);

                if (idx >= 0 ) {
                    quat = cur_odom_queue[idx];
                }else{
                    quat = cur_odom_queue[0];
                }

                Eigen::Matrix4f pose_map_to_odom = (initial_pose.inverse() * quat).eval();
                initial_odom_queue.emplace_back(pose_map_to_odom,msg->header.stamp.toSec() - temp);
                pose_received.store(true);
            }
        // }
    }
}
const std::string k_meta_map_fn = "meta_map.txt";
bool GlobalLocalization::loadMapCallback(mower_msgs::TriggerRequest &req,mower_msgs::TriggerResponse &res){
     cout << "============bool GlobalLocalization::loadMapCallback==========" << endl;
	if(is_load_map.load() == 0 || is_load_map.load() == 2){     //没有加载地图，或者加载完成再调就重新加载新地图
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
        if (IsFileExisting(meta_map_fn.c_str())) {
            std::ifstream fin(meta_map_fn);
            std::string tmp_name, tmp_version;
            while (fin >> tmp_name >> tmp_version) {
                if (tmp_name.size() == 15 && tmp_version.size() == 2) {
                    map_file_path = lmap_path + tmp_name + ".pcd";
                    submaps[tmp_name] = tmp_version;
                }
            }
            fin.close();
            if (submaps.size() == 0) {
                res.result = false;
                res.message = "SimplePoseGraph::loadMap(): No existing map found at: %s", lmap_path.c_str();
                return true;
            }
        } else {
            res.result = false;
            res.message = "No existing map found at: " + lmap_path;
            ROS_INFO("No existing map found at: %s", lmap_path.c_str());
            return true;
        }
        is_load_map.store(1);
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


#pragma endregion

#pragma region ICP相关函数
// 在 pointToPlaneICP 前预处理点云
pcl::PointCloud<PointT>::Ptr GlobalLocalization::preprocessCloud(
    const pcl::PointCloud<PointT>::Ptr& cloud,
    float voxel_size) {
    // 1. 离群点过滤
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50); // 邻域点数
    sor.setStddevMulThresh(1.5); // 标准差阈值（大于该值视为离群点）
    sor.filter(*cloud_filtered);

    // 2. 降采样
    pcl::PointCloud<PointT>::Ptr cloud_down(new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(cloud_filtered);
    vg.setLeafSize(voxel_size, voxel_size, voxel_size);
    vg.filter(*cloud_down);

    return cloud_down;
}

// 使用中心差分计算雅可比矩阵的示例函数
Eigen::Matrix<double, 1, 6> GlobalLocalization::computeJacobianCentralDifference(
    const PointT& point,
    const Eigen::Matrix<float, 4, 1>& plane_params,
    const Eigen::Matrix4f& transformation,
    double step_size) {
    
    Eigen::Matrix<double, 1, 6> jacobian;
    
    // 点到平面距离函数
    auto pointToPlaneDistance = [](const PointT& p, const Eigen::Matrix<float, 4, 1>& params) {
        return params(0) * p.x + params(1) * p.y + params(2) * p.z + params(3);
    };
    
    // 对每个变换参数计算偏导数
    for (int i = 0; i < 6; i++) {
        // 创建正向扰动变换
        Eigen::Matrix4d delta_transform_plus = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d delta_transform_minus = Eigen::Matrix4d::Identity();
        
        if (i < 3) { // 旋转参数
            Eigen::Vector3d axis(0, 0, 0);
            axis(i) = 1;
            Eigen::Matrix3d R_plus = Eigen::AngleAxisd(step_size, axis).toRotationMatrix();
            Eigen::Matrix3d R_minus = Eigen::AngleAxisd(-step_size, axis).toRotationMatrix();
            
            delta_transform_plus.block<3,3>(0,0) = R_plus;
            delta_transform_minus.block<3,3>(0,0) = R_minus;
        } else { // 平移参数
            Eigen::Vector3d t(0, 0, 0);
            t(i-3) = step_size;
            delta_transform_plus.block<3,1>(0,3) = t;
            t(i-3) = -step_size;
            delta_transform_minus.block<3,1>(0,3) = t;
        }
        
        // 应用扰动
        Eigen::Matrix4d transform_plus = delta_transform_plus * transformation.cast<double>();
        Eigen::Matrix4d transform_minus = delta_transform_minus * transformation.cast<double>();
        
        // 变换点
        Eigen::Vector4d point_homo(point.x, point.y, point.z, 1.0);
        Eigen::Vector4d point_plus = transform_plus * point_homo;
        Eigen::Vector4d point_minus = transform_minus * point_homo;
        
        PointT perturbed_point_plus;
        perturbed_point_plus.x = point_plus(0);
        perturbed_point_plus.y = point_plus(1);
        perturbed_point_plus.z = point_plus(2);
        
        PointT perturbed_point_minus;
        perturbed_point_minus.x = point_minus(0);
        perturbed_point_minus.y = point_minus(1);
        perturbed_point_minus.z = point_minus(2);
        
        // 计算中心差分
        double dist_plus = pointToPlaneDistance(perturbed_point_plus, plane_params);
        double dist_minus = pointToPlaneDistance(perturbed_point_minus, plane_params);
        
        jacobian(i) = (dist_plus - dist_minus) / (2 * step_size);
    }
    
    return jacobian;
}

/**
 * @brief 实现点到平面ICP算法
 * 
 * 该函数实现了基于点到平面距离的ICP算法，参考了激光雷达建图中的匹配策略。
 * 算法流程包括：
 * 1. 对源点云进行预处理
 * 2. 迭代优化位姿变换矩阵
 * 3. 在每次迭代中寻找对应点并估计局部平面
 * 4. 构建观测方程并求解最优变换
 * 
 * @param source 源点云
 * @param target 目标点云(kdtree已构建)
 * @param initial 初始变换矩阵
 * @param max_iterations 最大迭代次数
 * @param transformation_epsilon 收敛阈值
 * @return std::pair<Eigen::Matrix4f, double> 最终的变换矩阵和匹配度评分
 */
#if 1
std::pair<Eigen::Matrix4f, double> GlobalLocalization::pointToPlaneICP(
    const pcl::PointCloud<PointT>::Ptr& source,
    const pcl::PointCloud<PointT>::Ptr& target,
    const Eigen::Matrix4f& initial,
    int max_iterations ,
    double source_size,
    double transformation_epsilon 
) {
    // 预处理源点云
    auto source_down = preprocessCloud(source, source_size);
    
    if (source_down->empty()) {
        ROS_ERROR("Empty point cloud after preprocessing");
        return {initial, std::numeric_limits<double>::max()};
    }
    // 检查目标点云和KD树
    if (!target || target->empty()) {
        ROS_ERROR("Empty target point cloud");
        return {initial, std::numeric_limits<double>::max()};
    }
    
    // 初始化变换矩阵
    Eigen::Matrix4f transformation = initial;
    
    // 创建临时点云用于存储变换后的源点云
    pcl::PointCloud<PointT>::Ptr transformed_source(new pcl::PointCloud<PointT>);
    
    double final_fitness = 1.0;
    
    // 迭代优化过程
    for (int iter = 0; iter < max_iterations; iter++) {
        // 应用当前变换到源点云
        pcl::transformPointCloud(*source_down, *transformed_source, transformation);
        
        // 存储有效对应点的数量
        int effective_points = 0;
        
        // 观测向量和雅可比矩阵
        std::vector<double> residuals;
        std::vector<Eigen::Matrix<double, 1, 6>> jacobians;
        
        // 遍历变换后的源点云中的每个点
        for (size_t i = 0; i < transformed_source->size(); i++) {
            const PointT& point_world = transformed_source->points[i];
            // 检查点的有效性
            if (!pcl::isFinite(point_world)) continue;
            // 查找最近邻点
            PointVector points_near;
            std::vector<float> pointNKNSquaredDistance(NUM_MATCH_POINTS);
            
            // 使用kd-tree搜索最近邻点
            if (kdtree.size() > 0) {  // 添加检查
                kdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointNKNSquaredDistance);
            } else {
                ROS_WARN("KD tree is empty during search");
                continue;
            }
            
            // 检查是否有足够的近邻点
            bool point_selected = (points_near.size() >= NUM_MATCH_POINTS) && 
                                (pointNKNSquaredDistance[NUM_MATCH_POINTS - 1] <= 7.0f);
            
            if (!point_selected) continue;
            
            // 如果有足够的近邻点，估计局部平面
            if (points_near.size() >= NUM_MATCH_POINTS) {
                Eigen::Matrix<float, 4, 1> pabcd;
                if (esti_plane(pabcd, points_near, 0.6f)) {
                    // 计算点到平面的距离作为残差
                    float pd2 = pabcd(0) * point_world.x + 
                                pabcd(1) * point_world.y + 
                                pabcd(2) * point_world.z + 
                                pabcd(3);
                    
                    // 检查是否为有效对应点
                    float point_norm = sqrt(point_world.x * point_world.x + 
                                        point_world.y * point_world.y + 
                                        point_world.z * point_world.z);
                    
                    // 避免除零错误
                    float s = 1.0f - 0.9f * fabs(pd2) / (point_norm + 1e-6f);
                    
                    if (s > 0.9f) {
                        // 存储残差
                        residuals.push_back(-pd2);
                        
                        // // 计算雅可比矩阵
                        // Eigen::Matrix<double, 1, 6> jacobian;
                        // Eigen::Vector3d norm_vec(pabcd(0), pabcd(1), pabcd(2));
                        // Eigen::Vector3d point(point_world.x, point_world.y, point_world.z);
                        
                        // // 对于点到平面的雅可比矩阵:
                        // // [nx, ny, nz, nz*py - ny*pz, nx*pz - nz*px, ny*px - nx*py]
                        // jacobian(0) = norm_vec(0);
                        // jacobian(1) = norm_vec(1);
                        // jacobian(2) = norm_vec(2);
                        // jacobian(3) = norm_vec(2) * point(1) - norm_vec(1) * point(2);
                        // jacobian(4) = norm_vec(0) * point(2) - norm_vec(2) * point(0);
                        // jacobian(5) = norm_vec(1) * point(0) - norm_vec(0) * point(1);
                        
                        // jacobians.push_back(jacobian);

                        // 使用中心差分计算雅可比矩阵
                        Eigen::Matrix<double, 1, 6> jacobian = computeJacobianCentralDifference(
                            point_world, pabcd, transformation);
                        jacobians.push_back(jacobian);
                        effective_points++;
                    }
                }
            }
        }
        
        // 检查是否有足够的有效对应点
        if (effective_points < 15) { 
            ROS_WARN("Not enough effective points (%d) in ICP iteration %d", effective_points, iter);
            break;
        }
        
        // 构建线性系统 Ax = b
        Eigen::MatrixXd A(effective_points, 6);
        Eigen::VectorXd b(effective_points);
        
        for (int i = 0; i < effective_points; i++) {
            A.row(i) = jacobians[i];
            b(i) = residuals[i];
        }
        
        // 使用SVD求解线性系统
        Eigen::VectorXd delta = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
        double sum_sq_residual = 0.0;
        double this_fitness = std::numeric_limits<double>::max();
        for (const auto& res : residuals) {
            sum_sq_residual += res * res;
        }
        this_fitness = sqrt(sum_sq_residual / effective_points);
        // 检查收敛性
        if (delta.norm() < transformation_epsilon) {
            // // 计算最终的匹配度评分
            // double sum_sq_residual = 0.0;
            // for (const auto& res : residuals) {
            //     sum_sq_residual += res * res;
            // }
            // final_fitness = sqrt(sum_sq_residual / effective_points);
            final_fitness = this_fitness;
            break;
        }else if(final_fitness >= this_fitness){
            final_fitness = this_fitness;
        }
        
        // 构建增量变换矩阵
        Eigen::Matrix4d delta_transform = Eigen::Matrix4d::Identity();
        
        // 构建旋转矩阵（从李代数转换）
        Eigen::Vector3d w(delta(0), delta(1), delta(2));
        Eigen::Vector3d t(delta(3), delta(4), delta(5));
        
        double ang = w.norm();
        Eigen::Matrix3d R_delta = Eigen::Matrix3d::Identity();
        if (ang > 1e-12) {
            Eigen::Vector3d axis = w / ang;
            R_delta = Eigen::AngleAxisd(ang, axis).toRotationMatrix();
        }
        
        delta_transform.block<3, 3>(0, 0) = R_delta;
        delta_transform.block<3, 1>(0, 3) = t;

        
        // 更新总变换矩阵 (需要转换为float)
        transformation = (delta_transform.cast<float>() * transformation).eval();
    }
    
    return std::make_pair(transformation, final_fitness);
}
#endif

void GlobalLocalization::kdtree_bulid(int scale,pcl::PointCloud<pcl::PointXYZINormal>::Ptr& map_cloud){
    if (kdtree.Root_Node != nullptr) {
        cout << "============clear kdtree==========" << endl;
        kdtree.clear();
    }
    if (!map_cloud || map_cloud->empty()) {
        ROS_ERROR("Empty or null map cloud provided to kdtree_bulid");
        return;
    }
    pcl::PointCloud<PointT>::Ptr filtered_map(new pcl::PointCloud<PointT>());
    float voxel_size = MAP_VOXEL_SIZE * scale;
    // 清理地图点云
    for (const auto& point : map_cloud->points) {
        if (pcl::isFinite(point) && 
            std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
            std::abs(point.x) < 1000.0 && std::abs(point.y) < 1000.0 && std::abs(point.z) < 1000.0) {
            filtered_map->points.push_back(point);
        }
    }
    
    // 对地图点云进行降采样
    pcl::PointCloud<PointT>::Ptr downsampled_map(new pcl::PointCloud<PointT>());
    {
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(filtered_map);
        // 对于Z轴方向使用更小的体素尺寸以保持精度
        if (scale <= 1.0) {
            voxel.setLeafSize(voxel_size, voxel_size, voxel_size * 0.5);
        } else {
            voxel.setLeafSize(voxel_size, voxel_size, voxel_size* 0.5);
        }
        voxel.filter(*downsampled_map);
    }

    // 为 downsampled_map 计算法向量（用于点面 ICP）
    if (!downsampled_map->empty()) {
        pcl::PointCloud<pcl::Normal>::Ptr down_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::NormalEstimation<PointT, pcl::Normal> down_norm_est;
        pcl::search::KdTree<PointT>::Ptr down_tree(new pcl::search::KdTree<PointT>);
        down_norm_est.setInputCloud(downsampled_map);
        down_norm_est.setSearchMethod(down_tree);
        down_norm_est.setKSearch(10);
        down_norm_est.compute(*down_normals);

        // 将法向量赋回 downsampled_map
        for (size_t i = 0; i < downsampled_map->size() && i < down_normals->size(); ++i) {
            downsampled_map->points[i].normal_x = down_normals->points[i].normal_x;
            downsampled_map->points[i].normal_y = down_normals->points[i].normal_y;
            downsampled_map->points[i].normal_z = down_normals->points[i].normal_z;
        }
    }
    
    // 构建KD树
    kdtree.Build(downsampled_map->points);

}

#pragma endregion

#pragma region 功能代码
bool GlobalLocalization::loadMap(){
    cout<< "----------loadmap----------"<<endl;    


    if(map_cloud && map_cloud->empty()==false){
            cout <<"ma_cloud.size() = " << map_cloud->size() << endl;
        map_cloud->clear();
        map_cloud.reset(new pcl::PointCloud<PointT>);
    }
    if(global_map && global_map->empty()==false){
            cout <<"global_map.size() = " << global_map->size() << endl;
        global_map->clear();
        global_map.reset(new pcl::PointCloud<PointT>);
    }
    cout<< "Loading map from: " << map_file_path << endl;
    // 加载点云地图
    pcl::io::loadPCDFile(map_file_path, *map_cloud);


    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*map_cloud, *map_cloud, indices);



    // auto source_down = preprocessCloud(map_cloud, MAP_VOXEL_SIZE);
    // cout<<"Map size before downsample: "<<map_cloud->points.size()<<", after downsample: "<<source_down->points.size()<<endl;
    kdtree_bulid(1, map_cloud);
    global_map = map_cloud;

    // map_cloud->clear();
    // map_cloud.reset(new pcl::PointCloud<PointT>);
    is_load_map.store(2);
    cout<<"Global map size: "<<global_map->points.size()<<"---------------------------------"<<endl;
    return true;
}

/**
 * @brief: 匹配当前帧点云与全局地图
 * 
 * @param pose_estimation: 当前帧点云的位姿估计结果
 */
bool GlobalLocalization::globalLocalization(const Eigen::Matrix4f& pose_estimation) {
    if (!cur_scan || cur_scan->empty()) {
        ROS_WARN("Empty current scan");
        return false;
    }
    
    if (!global_map || global_map->empty()) {
        ROS_WARN("Empty global map");
        return false;
    }

    auto tic = std::chrono::high_resolution_clock::now();
    Eigen::Matrix4f best_transformation = Eigen::Matrix4f::Identity();
    double best_fitness = std::numeric_limits<double>::max();

    // auto fine_result = pointToPlaneICP(cur_scan, global_map,pose_estimation, 20, 1.5, 0.001);
    
    // // 精匹配阶段
    // Eigen::Matrix4f T_fine = fine_result.first;
    // // 使用较小体素尺寸进行精细匹配
    auto result_fine = pointToPlaneICP(cur_scan, global_map, pose_estimation, 60,0.1, 0.0001);

    best_fitness = result_fine.second;
    best_transformation = result_fine.first;
    
    auto toc = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
    // ROS_INFO("Time: %ld ms", duration.count());

    if (best_fitness < LOCALIZATION_TH) { 
        T_map_to_odom = best_transformation;
        // std::cout << "Debug: File=" << __FILE__ << ", Line=" << __LINE__ << ", Function=" << __FUNCTION__ << std::endl;
        // 发布map_to_odom
        nav_msgs::Odometry map_to_odom;
        Eigen::Affine3f affine(T_map_to_odom);
        tf::poseEigenToMsg(Eigen::Affine3d(affine.cast<double>()), map_to_odom.pose.pose);
        map_to_odom.header.stamp = ros::Time().fromSec(cur_scan_time);
        map_to_odom.header.frame_id = "map";
        pub_map_to_odom.publish(map_to_odom);
        cout<<"T_map_to_odom: \n"<<T_map_to_odom<<endl;

        ROS_WARN("!!! Global localization success !!!");
        return true;
    } else {
        // ROS_WARN("Not match!!!!");
        ROS_INFO("fitness score: %f", best_fitness);

        return false;
    }
}
#pragma endregion

/**
 * @brief 主循环即始化全局定位
 * 
 * @param cur_scan 当前帧点云
 * @param global_map 全局地图
 * @param pose_estimation 当前帧点云的位姿估计
 * @return std::pair<Eigen::Matrix4f, double> 匹配结果，包含位姿估计和匹配得分
 */
void GlobalLocalization::run(std::string map_path,std::string odom_file_path) {
    int number = 0;
    map_file_path = map_path;
    if(odom_file_path != ""){
        std::string odom_file = odom_file_path;
        odom_path_thread = std::thread(&GlobalLocalization::threadOdomPath, this, odom_file);
    }

    ros::Rate rate(FREQ_LOCALIZATION);  
    while (ros::ok()) {
    ros::spinOnce();
        {
            std::lock_guard<std::mutex> lock(map_mutex); 
            cout<< "---------is_load_map---------"<<is_load_map.load()<<endl;    
            if(is_load_map.load() == 1){        //1- 开启地图加载
                is_load_map.store(3);           //3- 加载中
                loadMap();
                continue;
            }else if(is_load_map.load() != 2){      // 等待地图加载完成
                rate.sleep();
                continue;
            }         
        }
        if (pose_received.load() && scan_received.load() && (global_map && global_map->empty()==false)) {
            should_initialize.store(true) ;
            pose_received.store(false);
            scan_received.store(false); 
        }
        if (should_initialize.load()) { 
            auto t1 = std::chrono::high_resolution_clock::now();
            Eigen::Matrix4f quat;
            {
                std::lock_guard<std::mutex> lock(initial_pose_mutex);
                int idx = initial_odom_queue.findAfter(cur_scan_time);
                cout<< "idx: " << idx << endl;
                cout<< "initial_odom_queue.size(): " << initial_odom_queue.size() << endl;

                if (idx >= 0 ) {
                    cout << "initial_odom_queue(idx): " << static_cast<long long>(initial_odom_queue(idx)* 1e6 ) << endl;
                    cout << "initial_odom_queue[idx]: " << initial_odom_queue[idx] << endl;
                    quat = initial_odom_queue[idx];
                }else{
                    cout << "initial_odom_queue(0): " << static_cast<long long>(initial_odom_queue(0)* 1e6 ) << endl;
                    cout << "initial_odom_queue[0]: " << initial_odom_queue[0] << endl;
                    quat = initial_odom_queue[0];
                }
            }
            globalLocalization(quat); 
            cur_scan->clear();
            should_initialize.store(false) ;

            auto t2 = std::chrono::high_resolution_clock::now();
            auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
            ROS_INFO("initialization Time: %ld ms", t3.count());
        }
 
        rate.sleep();
    }
    ros::spin();
}

