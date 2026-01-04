#include "icp_3d.h"

Icp3d::~Icp3d(){
    kdtree.clear();
}
// 在 pointToPlaneICP 前预处理点云
pcl::PointCloud<PointT>::Ptr Icp3d::preprocessCloud(
    const pcl::PointCloud<PointT>::Ptr& cloud,
    float voxel_size) {
    // 1. 离群点过滤
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50); // 邻域点数
    sor.setStddevMulThresh(1.5); // 标准差阈值（大于该值视为离群点）
    sor.filter(*cloud_filtered);

    // // 2. 降采样
    // pcl::PointCloud<PointT>::Ptr cloud_down(new pcl::PointCloud<PointT>);
    // pcl::VoxelGrid<PointT> vg;
    // vg.setInputCloud(cloud_filtered);
    // vg.setLeafSize(voxel_size, voxel_size, voxel_size);
    // vg.filter(*cloud_down);

    return cloud_filtered;
}

// 使用中心差分计算雅可比矩阵的示例函数
Eigen::Matrix<double, 1, 6> Icp3d::computeJacobianCentralDifference(
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
std::pair<Eigen::Matrix4f, double> Icp3d::pointToPlaneICP(
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
        printf("Empty point cloud after preprocessing");
        return {initial, std::numeric_limits<double>::max()};
    }
    // 检查目标点云和KD树
    if (!target || target->empty()) {
        printf("Empty target point cloud");
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
            if (!std::isfinite(point_world.x) || !std::isfinite(point_world.y) || !std::isfinite(point_world.z)) continue;
            
            // 查找最近邻点
            PointVector points_near;
            std::vector<float> pointNKNSquaredDistance(NUM_MATCH_POINTS);
            
            // 使用kd-tree搜索最近邻点
            // 确保 KD 树和目标点云包含足够的点以避免底层搜索越界
            if (kdtree.size() >= NUM_MATCH_POINTS && target->size() >= static_cast<size_t>(NUM_MATCH_POINTS)) {
                kdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointNKNSquaredDistance);
            } else {
                 // printf("Skip nearest search: kdtree.size()=%d, target->size()=%zu\n", kdtree.size(), target->size());
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
            printf("Not enough effective points (%d) in ICP iteration %d", effective_points, iter);
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

void Icp3d::kdtree_bulid(int scale, pcl::PointCloud<pcl::PointXYZINormal>::Ptr& map_cloud){
    if (kdtree.Root_Node != nullptr) {
        cout << "============kdtree_bulid clear kdtree==========" << endl;
        kdtree.Reset_Tree();
    }
    if (!map_cloud || map_cloud->empty()) {
        printf("Empty or null map cloud provided to kdtree_bulid");
        return;
    }
    
    // 首先确保输入点云中的所有点都是有限的
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr clean_map_cloud(new pcl::PointCloud<pcl::PointXYZINormal>());
    for (const auto& point : map_cloud->points) {
        if (pcl::isFinite(point) && 
            std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
            std::abs(point.x) < 1000.0 && std::abs(point.y) < 1000.0 && std::abs(point.z) < 1000.0) {
            clean_map_cloud->points.push_back(point);
        }
    }
    
    if (clean_map_cloud->empty()) {
        printf("Cleaned map cloud is empty");
        return;
    }
    
    float voxel_size = MAP_VOXEL_SIZE * scale;
    // 对地图点云进行降采样
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr downsampled_map(new pcl::PointCloud<pcl::PointXYZINormal>());
    {
        pcl::VoxelGrid<pcl::PointXYZINormal> voxel;  // 使用正确的类型
        voxel.setInputCloud(clean_map_cloud);
        // 对于Z轴方向使用更小的体素尺寸以保持精度
        if (scale <= 1.0) {
            voxel.setLeafSize(voxel_size, voxel_size, voxel_size * 0.5);
        } else {
            voxel.setLeafSize(voxel_size, voxel_size, voxel_size* 0.5);
        }
        voxel.filter(*downsampled_map);
    }

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr filtered_map(new pcl::PointCloud<pcl::PointXYZINormal>());

    // 再次清理地图点云
    for (const auto& point : downsampled_map->points) {
        if (pcl::isFinite(point) && 
            std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
            std::abs(point.x) < 1000.0 && std::abs(point.y) < 1000.0 && std::abs(point.z) < 1000.0) {
            filtered_map->points.push_back(point);
        }
    }
    
    if (filtered_map->empty()) {
        printf("Filtered map cloud is empty after preprocessing");
        return;
    }

    // 为 filtered_map 计算法向量（用于点面 ICP）
    if (!filtered_map->empty()) {
        pcl::PointCloud<pcl::Normal>::Ptr down_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::NormalEstimation<pcl::PointXYZINormal, pcl::Normal> down_norm_est;  // 使用正确的类型
        pcl::search::KdTree<pcl::PointXYZINormal>::Ptr down_tree(new pcl::search::KdTree<pcl::PointXYZINormal>);  // 使用正确的类型
        down_norm_est.setInputCloud(filtered_map);
        down_norm_est.setSearchMethod(down_tree);
        down_norm_est.setKSearch(10);
        down_norm_est.compute(*down_normals);

        // 将法向量赋回 filtered_map
        for (size_t i = 0; i < filtered_map->size() && i < down_normals->size(); ++i) {
            filtered_map->points[i].normal_x = down_normals->points[i].normal_x;
            filtered_map->points[i].normal_y = down_normals->points[i].normal_y;
            filtered_map->points[i].normal_z = down_normals->points[i].normal_z;
        }
    }
    
    // 构建KD树前检查点云大小
    printf("Building KD tree with %zu points\n", filtered_map->size());
    
    // 构建KD树
    kdtree.Build(filtered_map->points);
    
    printf("KD tree built with %d nodes\n", kdtree.size());
}