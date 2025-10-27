#include "globalLocalization.h"

// ROS相关
ros::Publisher pub_pc_in_map;
ros::Publisher pub_submap;
ros::Publisher pub_map_to_odom;
ros::Subscriber sub_cloud_registered;
ros::Subscriber sub_odometry;
ros::Publisher pcl_pub;
ros::Publisher path_pub;
KD_TREE<PointT> kdtree;
// 参数
double MAP_VOXEL_SIZE = 0.2;
double SCAN_VOXEL_SIZE = 0.1;
double FREQ_LOCALIZATION = 3;
double LOCALIZATION_TH = 0.3;
double FOV =360;
double FOV_FAR = 40.0;
std::string map_file_path = "";
using namespace std;

 GlobalLocalization::GlobalLocalization(ros::NodeHandle& nh) : 
    global_map(new pcl::PointCloud<PointT>),
    cur_scan(new pcl::PointCloud<PointT>),
    map_cloud(new pcl::PointCloud<PointT>),
    T_map_to_odom(Eigen::Matrix4f::Identity()) {
        

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
    
    Eigen::Matrix4f inverseSE3(const Eigen::Matrix4f& trans) {
        Eigen::Matrix4f trans_inverse = Eigen::Matrix4f::Identity();
        // R
        trans_inverse.block<3,3>(0,0) = trans.block<3,3>(0,0).transpose();
        // t
        trans_inverse.block<3,1>(0,3) = -trans_inverse.block<3,3>(0,0) * trans.block<3,1>(0,3);
        return trans_inverse;
    }
    
    pcl::PointCloud<PointT>::Ptr GlobalLocalization::voxelDownSample(pcl::PointCloud<PointT>::Ptr cloud, float voxel_size) {

        pcl::PointCloud<PointT>::Ptr cloud_downsampled(new pcl::PointCloud<PointT>);
        pcl::VoxelGrid<PointT> voxel_grid;
        voxel_grid.setInputCloud(cloud);
        voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
        voxel_grid.filter(*cloud_downsampled);
        
        // 确保输出点云中的所有点都是有效的
        pcl::PointCloud<PointT>::Ptr cleaned_cloud(new pcl::PointCloud<PointT>);
        for (const auto& point : cloud_downsampled->points) {
            if (pcl::isFinite(point) && 
                std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
                std::isfinite(point.normal_x) && std::isfinite(point.normal_y) && std::isfinite(point.normal_z)) {
                cleaned_cloud->points.push_back(point);
            }
        }
        cleaned_cloud->width = cleaned_cloud->points.size();
        cleaned_cloud->height = 1;
        cleaned_cloud->is_dense = true;
        
        return cleaned_cloud;
    }
    
    /**
     * @brief 从全局地图中裁剪出当前LiDAR视角范围内的子地图
     * 
     * 该函数将全局地图点云变换到LiDAR坐标系下，并根据LiDAR的视场角（FOV）和最大探测距离，
     * 提取出位于当前LiDAR视角范围内的地图点，用于后续处理。
     * 同时会发布裁剪后的子地图点云。
     * 
     * @param global_map 全局地图点云指针
     * @param pose_estimation 估计的位姿变换矩阵（地图到odom坐标系）
     * @param cur_odom 当前里程计信息，包含传感器的位姿
     * @return 返回视角范围内的地图点云指针
     */
    pcl::PointCloud<PointT>::Ptr GlobalLocalization::cropGlobalMapInFOV(
        pcl::PointCloud<PointT>::Ptr global_map,
        const Eigen::Matrix4f& pose_estimation,
        const nav_msgs::Odometry& cur_odom) {
        
        // 计算从地图坐标系到LiDAR坐标系的变换矩阵
        Eigen::Matrix4f T_odom_to_base_link = poseToMatrix(cur_odom).cast<float>();
        Eigen::Matrix4f T_map_to_base_link = pose_estimation * T_odom_to_base_link;
        Eigen::Matrix4f T_base_link_to_map = inverseSE3(T_map_to_base_link);
        
        // 将全局地图点云变换到LiDAR坐标系下
        pcl::PointCloud<PointT>::Ptr global_map_in_base_link(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*global_map, *global_map_in_base_link, T_base_link_to_map);
        
        // 根据LiDAR的视场角筛选在视角范围内的点
        pcl::PointCloud<PointT>::Ptr global_map_in_FOV(new pcl::PointCloud<PointT>);
        
        for (const auto& point : global_map_in_base_link->points) {
            bool in_fov = false;

            if (point.x < FOV_FAR && 
                std::atan2(point.y, point.x)*180/M_PI > (90.0+(360.0 - FOV)/2) || std::atan2(point.y, point.x)*180/M_PI < (90.0-(360.0 - FOV)/2)) {
                in_fov = true;
            }

            
            if (in_fov) {
                global_map_in_FOV->points.push_back(point);
            }
        }
        
        global_map_in_FOV->width = global_map_in_FOV->points.size();
        global_map_in_FOV->height = 1;
        
        // 发布裁剪后的子地图点云消息
        sensor_msgs::PointCloud2 submap_msg;
        pcl::toROSMsg(*global_map_in_FOV, submap_msg);
        submap_msg.header.stamp = cur_odom.header.stamp;
        submap_msg.header.frame_id = "map";
        pub_submap.publish(submap_msg);
        
        return global_map_in_FOV;
    }
    

    bool GlobalLocalization::globalLocalization(const Eigen::Matrix4f& pose_estimation) {

        // pcl::PointCloud<PointT>::Ptr global_map_in_FOV =  cropGlobalMapInFOV(global_map, pose_estimation, cur_odom.front());
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

        // for (const auto& candidate : candidates) {
            // auto coarse_result = optimizedManualRegistrationAtScale(cur_scan, true, pose_estimation, 5.0);
            // auto fine_result = optimizedManualRegistrationAtScale(cur_scan, false, coarse_result.first, 1.0);

            // auto coarse_result = pointToPlaneICP(cur_scan, global_map,pose_estimation);
            auto fine_result = pointToPlaneICP(cur_scan, global_map,pose_estimation);
            
            // if (fine_result.second < best_fitness) {
                best_fitness = fine_result.second;
                best_transformation = fine_result.first;
        //     }
        // }
        
        auto toc = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
        // ROS_INFO("Time: %ld ms", duration.count());

        // 当全局定位成功时才更新map2odom
        if (best_fitness < LOCALIZATION_TH) { 
            T_map_to_odom = best_transformation;
            // std::cout << "Debug: File=" << __FILE__ << ", Line=" << __LINE__ << ", Function=" << __FUNCTION__ << std::endl;
            // 发布map_to_odom
            nav_msgs::Odometry map_to_odom;
            Eigen::Affine3f affine(T_map_to_odom);
            tf::poseEigenToMsg(Eigen::Affine3d(affine.cast<double>()), map_to_odom.pose.pose);
            map_to_odom.header.stamp = cur_odom.header.stamp;
            map_to_odom.header.frame_id = "map";
            pub_map_to_odom.publish(map_to_odom);
            cout<<"T_map_to_odom: \n"<<T_map_to_odom<<endl;

            Eigen::Matrix4f T_odom_to_base_link = poseToMatrix(cur_odom).cast<float>();
            
            Eigen::Vector3f xyz0 = T_odom_to_base_link.block<3,1>(0,3);
            Eigen::Matrix3f R0 = T_odom_to_base_link.block<3,3>(0,0);
            Eigen::Quaternionf quat_result0(R0);

            cout<<"T_odom_to_base_link.x0 : \n"<<xyz0.x() << ",y0=" << xyz0.y() <<",z0 = " << xyz0.z()<<endl;
            cout<<"T_odom_to_base_link.quat0 : \n"<<quat_result0.x() << ",y0=" << quat_result0.y() <<",z0 = " << quat_result0.z()<<",w0 = " << quat_result0.w()<<endl;
            // 计算T_map_to_base_link = T_map_to_odom * T_odom_to_base_link
            Eigen::Matrix4f T_map_to_base_link = T_map_to_odom * T_odom_to_base_link;
            
            // 提取位置和姿态信息
            Eigen::Vector3f xyz = T_map_to_base_link.block<3,1>(0,3);
            Eigen::Matrix3f R = T_map_to_base_link.block<3,3>(0,0);
            Eigen::Quaternionf quat_result(R);
            
            cout<<"T_map_to_base_link.x : \n"<<xyz.x() << ",y=" << xyz.y() <<",z = " << xyz.z()<<endl;
            cout<<"T_map_to_base_link.quat : \n"<<quat_result.x() << ",y=" << quat_result.y() <<",z = " << quat_result.z()<<",w = " << quat_result.w()<<endl;

            ROS_WARN("!!! Global localization success !!!");
            return true;
        } else {
            // ROS_WARN("Not match!!!!");
            ROS_INFO("fitness score: %f", best_fitness);

            return false;
        }
    }

       
    
    void GlobalLocalization::initializeGlobalMap(const sensor_msgs::PointCloud2ConstPtr& pc_msg) {
        pcl::fromROSMsg(*pc_msg, *global_map);
        //global_map = voxelDownSample(global_map, MAP_VOXEL_SIZE);
        ROS_INFO("Global map received.");
    }
    
    void GlobalLocalization::cbSaveCurOdom(const nav_msgs::OdometryConstPtr& odom_msg) {
        cur_odom=*odom_msg;
        odom_received = true;
    }
    
    void GlobalLocalization::cbSaveCurScan(const sensor_msgs::PointCloud2ConstPtr& pc_msg) {
        if(!should_initialize && scan_received==false){
            sensor_msgs::PointCloud2 modified_msg = *pc_msg;
            modified_msg.header.frame_id = "camera_init";
            modified_msg.header.stamp = ros::Time::now();
            pub_pc_in_map.publish(modified_msg);

            pcl::fromROSMsg(*pc_msg, *cur_scan);

            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*cur_scan, *cur_scan, indices);


            scan_received = true;
        }

    }
    
    void GlobalLocalization::threadLocalization() {
        ros::Rate rate(FREQ_LOCALIZATION);
        while (ros::ok()) {
            rate.sleep();
            if (initialized && scan_received) {
                auto t1 = std::chrono::high_resolution_clock::now();
                globalLocalization(initial_pose);

                pose_received = false;
                scan_received = false;
                auto t2 = std::chrono::high_resolution_clock::now();
                auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
                ROS_INFO("initialization Time: %ld ms", t3.count());
            }
        }
    }
    // 回调函数
    void GlobalLocalization::initialPoseCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg) {
        Eigen::Matrix4f initial_pose_temp = poseToMatrix(*msg);
        
        // // 为平移部分添加噪声（xyz三个方向）
        // static std::default_random_engine generator;
        // static std::normal_distribution<double> distribution(0.0, 5.0); 
        
        // // 添加噪声到平移部分
        // initial_pose(0,3) += static_cast<float>(distribution(generator)); 
        // initial_pose(1,3) += static_cast<float>(distribution(generator)); 
        // initial_pose(2,3) += static_cast<float>(distribution(generator)); 

        if(!should_initialize && pose_received == false){
            initial_pose = initial_pose_temp;
            pose_received = true;
        }


    }



// 在 pointToPlaneICP 前预处理点云
pcl::PointCloud<PointT>::Ptr GlobalLocalization::preprocessCloud(
    const pcl::PointCloud<PointT>::Ptr& cloud,
    float voxel_size) {
    // 1. 离群点过滤
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50); // 邻域点数
    sor.setStddevMulThresh(1.0); // 标准差阈值（大于该值视为离群点）
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
    double transformation_epsilon 
) {
    // 预处理源点云
    auto source_down = preprocessCloud(source, SCAN_VOXEL_SIZE);
    
    
    if (source_down->empty()) {
        ROS_ERROR("Empty point cloud after preprocessing");
        return {initial, std::numeric_limits<double>::max()};
    }
    
    // 初始化变换矩阵
    Eigen::Matrix4f transformation = initial;
    
    // 创建临时点云用于存储变换后的源点云
    pcl::PointCloud<PointT>::Ptr transformed_source(new pcl::PointCloud<PointT>);
    
    double final_fitness = std::numeric_limits<double>::max();
    
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
            
            // 查找最近邻点
            PointVector points_near;
            std::vector<float> pointNKNSquaredDistance(NUM_MATCH_POINTS);
            
            // 使用kd-tree搜索最近邻点
            kdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointNKNSquaredDistance);
            
            // 检查是否有足够的近邻点
            bool point_selected = (points_near.size() >= NUM_MATCH_POINTS) && 
                                 (pointNKNSquaredDistance[NUM_MATCH_POINTS - 1] <= 5.0f);
            
            if (!point_selected) continue;
            
            // 如果有足够的近邻点，估计局部平面
            if (points_near.size() >= NUM_MATCH_POINTS) {
                Eigen::Matrix<float, 4, 1> pabcd;
                if (esti_plane(pabcd, points_near, 0.05f)) {
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
        }else if(final_fitness > this_fitness){
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
        kdtree.Build(map_cloud->points);
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

    void GlobalLocalization::run(std::string map_file_path ) {
        ros::NodeHandle nh;
        
            // map_file_path = map_file_path;
            
            // 检查地图文件是否存在
            std::ifstream file(map_file_path);
            if (!file.good()) {
                std::cout << "No existing map found at: " << map_file_path << std::endl;
                return;
            }
            file.close();
            
            // 加载点云地图
      
            pcl::io::loadPCDFile(map_file_path, *map_cloud);


            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*map_cloud, *map_cloud, indices);

            // std::thread odom_path_thread(&GlobalLocalization::threadOdomPath,this, std::string("/home/edy/code/lidarcode/src/PCD/odometry_data.txt"));
            // odom_path_thread.detach();
            std::string odom_file = "/home/edy/code/lidarcode/src/PCD/odometry_data.txt";
            odom_path_thread = std::thread(&GlobalLocalization::threadOdomPath, this, odom_file);

            auto source_down = preprocessCloud(map_cloud, MAP_VOXEL_SIZE);

            kdtree_bulid(1, source_down);
            global_map = source_down; // voxelDownSample(final_cleaned_map, MAP_VOXEL_SIZE);

            map_cloud.reset();
            // final_cleaned_map.reset();

            cout<<"Global map size: "<<global_map->points.size()<<endl;

            // 为全局地图计算法向量
            // pcl::PointCloud<pcl::Normal>::Ptr map_normals(new pcl::PointCloud<pcl::Normal>);
            // pcl::NormalEstimation<PointT, pcl::Normal> map_norm_est;
            // map_norm_est.setInputCloud(global_map);
            // pcl::search::KdTree<PointT>::Ptr map_tree(new pcl::search::KdTree<PointT>);
            // map_norm_est.setSearchMethod(map_tree);
            // map_norm_est.setKSearch(10);
            // map_norm_est.compute(*map_normals);

            // // 将法向量赋值给全局地图点
            // for (size_t i = 0; i < global_map->size() && i < map_normals->size(); ++i) {
            //     global_map->points[i].normal_x = map_normals->points[i].normal_x;
            //     global_map->points[i].normal_y = map_normals->points[i].normal_y;
            //     global_map->points[i].normal_z = map_normals->points[i].normal_z;
            // }

        
            int number = 0;
            // initializeGlobalMap(map_msg);
            ros::Rate rate(10);  
            // 初始化
            while (!initialized && ros::ok()) {
            ros::spinOnce();
    

            Eigen::Matrix4f temp_pose;
            
            {
                std::lock_guard<std::mutex> lock(data_mutex);
                if (pose_received && scan_received && (global_map && global_map->empty()==false)) {
                    should_initialize = true;
                    temp_pose = initial_pose;
                    pose_received = false;
                    scan_received = false;
                }
                if (should_initialize) {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    initialized = globalLocalization(temp_pose);
                    should_initialize = false;
                    auto t2 = std::chrono::high_resolution_clock::now();
                    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
                    ROS_INFO("initialization Time: %ld ms", t3.count());
                }
            }
            

            
            rate.sleep();
        }
        
        if (initialized) {
            ROS_INFO("");
            ROS_INFO("Initialize successfully!!!!!!");
            ROS_INFO("");
            pose_received = false; 
            // 开始定期全局定位
            // odom_path_thread = std::thread(&GlobalLocalization::threadOdomPath, this, odom_file);
            std::thread localization_thread(&GlobalLocalization::threadLocalization, this);
            localization_thread.detach();
        }
        
        ros::spin();
    }