// global_localization_node.cpp
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/filters/filter.h>   
#include <pcl/common/common.h>   
#include <thread>
#include <chrono>
#include <memory>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <deque>
#include <mutex>
#include <iosfwd>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <omp.h>
#include <pcl/registration/ndt.h>
#include <random>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>


using namespace std;
using PointT = pcl::PointXYZINormal;//PointXYZINormal
// ROS相关
ros::Publisher pub_pc_in_map;
ros::Publisher pub_submap;
ros::Publisher pub_map_to_odom;
ros::Subscriber sub_cloud_registered;
ros::Subscriber sub_odometry;
ros::Publisher pcl_pub;
// 参数
double MAP_VOXEL_SIZE = 0.2;
double SCAN_VOXEL_SIZE = 0.1;
double FREQ_LOCALIZATION = 3;
double LOCALIZATION_TH = 0.3;
double FOV =360;
double FOV_FAR = 50.0;
std::string map_file_path = "";

class GlobalLocalization {
private:
    // 全局地图数据
    pcl::PointCloud<PointT>::Ptr global_map;
    
    // 初始化状态标志
    bool initialized = false;
    
    // 从map坐标系到odom坐标系的变换矩阵
    Eigen::Matrix4f T_map_to_odom;

    // 当前里程计数据和激光扫描数据
    // deque< nav_msgs::Odometry> cur_odom = deque< nav_msgs::Odometry>();
    nav_msgs::Odometry cur_odom;
    pcl::PointCloud<PointT>::Ptr cur_scan;
    // deque<pcl::PointCloud<PointT>::Ptr> lidar_buffer = deque<pcl::PointCloud<PointT>::Ptr>();

    // boost::shared_ptr<geometry_msgs::PoseWithCovarianceStamped const> initial_pose_msg;
    // deque<Eigen::Matrix4f> initial_pose_buffer = deque<Eigen::Matrix4f>();
    Eigen::Matrix4f initial_pose;
    
    std::mutex pose_mutex;
    std::mutex data_mutex;
    bool pose_received = false;
    
    bool odom_received = false;
    bool scan_received = false;
    bool should_initialize = false;

public:
    GlobalLocalization(ros::NodeHandle& nh) : 
    global_map(new pcl::PointCloud<PointT>),
    cur_scan(new pcl::PointCloud<PointT>),
    T_map_to_odom(Eigen::Matrix4f::Identity()) {
        

    }
    
    Eigen::Matrix4f poseToMatrix(const geometry_msgs::PoseWithCovarianceStamped& pose_msg) {
        Eigen::Affine3d affine;
        tf::poseMsgToEigen(pose_msg.pose.pose, affine);
        return affine.matrix().cast<float>();
    }
    
    Eigen::Matrix4f poseToMatrix(const nav_msgs::Odometry& odom_msg) {
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
    
    pcl::PointCloud<PointT>::Ptr voxelDownSample(pcl::PointCloud<PointT>::Ptr cloud, float voxel_size) {
        // pcl::PointCloud<PointT>::Ptr cloud_downsampled(new pcl::PointCloud<PointT>);
        // pcl::VoxelGrid<PointT> voxel_grid;
        // voxel_grid.setInputCloud(cloud);
        // voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
        // voxel_grid.filter(*cloud_downsampled);
        // return cloud_downsampled;

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
    pcl::PointCloud<PointT>::Ptr cropGlobalMapInFOV(
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
    
/**
 * @brief 在指定尺度下执行点云配准
 * 
 * 该函数使用ICP算法将扫描点云配准到地图点云上。首先根据尺度参数对扫描点云进行降采样，
 * 然后配置并执行ICP算法，最后返回配准变换矩阵和匹配度评分。
 * 
 * @param scan 待配准的扫描点云指针
 * @param map 作为参考的地图点云指针
 * @param initial 初始变换矩阵
 * @param scale 尺度参数，用于调整降采样体素大小和最大对应距离
 * @return std::pair<Eigen::Matrix4f, double> 配准结果，包含最终变换矩阵和匹配度评分
 */
std::pair<Eigen::Matrix4f, double> registrationAtScale_ptop(const pcl::PointCloud<PointT>::Ptr& scan,
                                                       const pcl::PointCloud<PointT>::Ptr& map,
                                                       const Eigen::Matrix4f& initial,
                                                       double scale) {
            // 检查输入点云
            if (!global_map || global_map->points.empty()) {
                ROS_WARN("Invalid global map");
                return {Eigen::Matrix4f::Identity(), std::numeric_limits<double>::max()};
            }
                // 检查输入点云
            if (!scan || !map) {
                ROS_WARN("Null point cloud pointer in registration");
                return {Eigen::Matrix4f::Identity(), std::numeric_limits<double>::max()};
            }
    
            std::vector<int> indices_scan, indices_map;
            pcl::removeNaNFromPointCloud(*scan, *scan, indices_scan);
            pcl::removeNaNFromPointCloud(*map, *map, indices_map);

            pcl::PointCloud<PointT>::Ptr filtered_scan(new pcl::PointCloud<PointT>());
            pcl::PointCloud<PointT>::Ptr filtered_map(new pcl::PointCloud<PointT>());
 
            for (const auto& point : scan->points) {
                if (pcl::isFinite(point) && 
                    std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
                    std::abs(point.x) < 1000.0 && std::abs(point.y) < 1000.0 && std::abs(point.z) < 1000.0) {
                    filtered_scan->points.push_back(point);
                }
            }
            
            for (const auto& point : map->points) {
                if (pcl::isFinite(point) && 
                    std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
                    std::abs(point.x) < 1000.0 && std::abs(point.y) < 1000.0 && std::abs(point.z) < 1000.0) {
                    filtered_map->points.push_back(point);
                }
            }
            
            if (filtered_map->empty() || filtered_scan->empty()) {
                ROS_WARN("No valid points after cleaning");
                return {Eigen::Matrix4f::Identity(), std::numeric_limits<double>::max()};
            }                                
            // 根据尺度参数对点云进行降采样处理
            // pcl::PointCloud<PointT>::Ptr scan_ds = voxelDownSample(scan, SCAN_VOXEL_SIZE * scale );
            pcl::PointCloud<PointT>::Ptr map_ds = voxelDownSample(filtered_map, MAP_VOXEL_SIZE * scale );
            // cout<<"voxelDownSample map size: "<<map_ds->points.size()<<endl;  

            double density = filtered_scan->size() / (M_PI * FOV_FAR * FOV_FAR);
            // double temp = 0.1 + 0.5 * (1.0 - std::min(1.0, density * 1000));
            cout<<"density: "<<density<<endl;
            // 配置ICP算法参数
            pcl::IterativeClosestPoint<PointT, PointT> icp;
            icp.setMaximumIterations(20);
            icp.setMaxCorrespondenceDistance(0.4 * scale);
            // icp.setTransformationEpsilon(1e-8);
            // icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setInputSource(filtered_scan);
            icp.setInputTarget(filtered_map);

            // 执行ICP配准
            pcl::PointCloud<PointT> Final;
            Eigen::Matrix4f init = initial;
            icp.align(Final, init);

            // 获取配准结果
            Eigen::Matrix4f transformation = icp.getFinalTransformation();
            double fitness = icp.getFitnessScore();
            return {transformation, fitness};
    }

std::pair<Eigen::Matrix4f, double> registrationAtScale(const pcl::PointCloud<PointT>::Ptr& scan,
                                                       const pcl::PointCloud<PointT>::Ptr& map,
                                                       const Eigen::Matrix4f& initial,
                                                       double scale) {
         // 检查输入点云
    if (!map || map->points.empty() || !scan || scan->points.empty()) {
        ROS_WARN("Invalid input point clouds");
        return {Eigen::Matrix4f::Identity(), std::numeric_limits<double>::max()};
    }

    // static pcl::PointCloud<PointT>::Ptr filtered_scan(new pcl::PointCloud<PointT>());
    // static pcl::PointCloud<PointT>::Ptr filtered_map(new pcl::PointCloud<PointT>());
    static pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    static pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    
    // filtered_scan->clear();
    // filtered_map->clear();
    normals->clear();

    // // 向量化过滤 - 使用 reserve 预分配内存
    // filtered_scan->reserve(scan->size());
    // filtered_map->reserve(map->size());

    // // 过滤扫描点云
    // for (const auto& point : scan->points) {
    //     if (pcl::isFinite(point) && 
    //         std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
    //         std::abs(point.x) < 1000.0 && std::abs(point.y) < 1000.0 && std::abs(point.z) < 1000.0) {
    //         filtered_scan->push_back(point);
    //     }
    // }
    
    // // 过滤地图点云
    // for (const auto& point : map->points) {
    //     if (pcl::isFinite(point) && 
    //         std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
    //         std::abs(point.x) < 1000.0 && std::abs(point.y) < 1000.0 && std::abs(point.z) < 1000.0) {
    //         filtered_map->push_back(point);
    //     }
    // }
    
    // if (filtered_map->empty() || filtered_scan->empty()) {
    //     ROS_WARN("No valid points after cleaning");
    //     return {Eigen::Matrix4f::Identity(), std::numeric_limits<double>::max()};
    // }

    // 降采样处理
    pcl::PointCloud<PointT>::Ptr map_ds = voxelDownSample(map, MAP_VOXEL_SIZE * scale);

    // 计算法向量 - 优化搜索方法
    pcl::NormalEstimation<PointT, pcl::Normal> norm_est;
    norm_est.setInputCloud(map_ds);
    norm_est.setSearchMethod(tree);  // 重用搜索树
    norm_est.setKSearch(10);
    norm_est.compute(*normals);

    // 更新法向量
    for (size_t i = 0; i < map_ds->size() && i < normals->size(); ++i) {
        map_ds->points[i].normal_x = normals->points[i].normal_x;
        map_ds->points[i].normal_y = normals->points[i].normal_y;
        map_ds->points[i].normal_z = normals->points[i].normal_z;
    }

    // ICP配准优化
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setMaximumIterations(15); 
    icp.setMaxCorrespondenceDistance(0.4 * scale);
    icp.setTransformationEpsilon(1e-6);  // 设置收敛条件
    icp.setEuclideanFitnessEpsilon(1e-4);  // 设置适应度阈值
    
    // 设置点对面距离度量
    typedef pcl::registration::TransformationEstimationPointToPlaneLLS<PointT, PointT> PointToPlane;
    boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
    icp.setTransformationEstimation(point_to_plane);
    
    icp.setInputSource(scan);
    icp.setInputTarget(map_ds);

    // 执行ICP配准
    pcl::PointCloud<PointT> Final;
    icp.align(Final, initial);

    // 获取配准结果
    Eigen::Matrix4f transformation = icp.getFinalTransformation();
    double fitness = icp.getFitnessScore();
    return {transformation, fitness};

}

    bool flag = true;
    bool globalLocalization(const Eigen::Matrix4f& pose_estimation) {
        auto tic = std::chrono::high_resolution_clock::now();
        // pcl::PointCloud<PointT>::Ptr global_map_in_FOV =  cropGlobalMapInFOV(global_map, pose_estimation, cur_odom.front());

        
        Eigen::Matrix4f best_transformation = Eigen::Matrix4f::Identity();
        double best_fitness = std::numeric_limits<double>::max();

        // for (const auto& candidate : candidates) {
            auto coarse_result = registrationAtScale(cur_scan, global_map, pose_estimation, 10.0);
            auto fine_result = registrationAtScale(cur_scan, global_map, coarse_result.first, 1.0);
            
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
    
    void initializeGlobalMap(const sensor_msgs::PointCloud2ConstPtr& pc_msg) {
        pcl::fromROSMsg(*pc_msg, *global_map);
        //global_map = voxelDownSample(global_map, MAP_VOXEL_SIZE);
        ROS_INFO("Global map received.");
    }
    
    void cbSaveCurOdom(const nav_msgs::OdometryConstPtr& odom_msg) {
        cur_odom=*odom_msg;
        odom_received = true;
    }
    
    void cbSaveCurScan(const sensor_msgs::PointCloud2ConstPtr& pc_msg) {
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
    
    void threadLocalization() {
        ros::Rate rate(FREQ_LOCALIZATION);
        while (ros::ok()) {
            rate.sleep();
            if (initialized) {
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
    void initialPoseCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg) {
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


    void run(std::string map_file_path ) {
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
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZINormal>());
            pcl::io::loadPCDFile(map_file_path, *map_cloud);


            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*map_cloud, *map_cloud, indices);

            pcl::PointCloud<pcl::PointXYZINormal>::Ptr final_cleaned_map(new pcl::PointCloud<pcl::PointXYZINormal>);
            int invalid_count = 0;
            
            for (const auto& point : map_cloud->points) {
                if (pcl::isFinite(point) && 
                    std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
                    std::abs(point.x) < 1000.0 && std::abs(point.y) < 1000.0 && std::abs(point.z) < 1000.0) {
                    final_cleaned_map->points.push_back(point);
                } else {
                    invalid_count++;
                }
            }
            
            final_cleaned_map->width = final_cleaned_map->points.size();
            final_cleaned_map->height = 1;
            final_cleaned_map->is_dense = true;

            global_map = voxelDownSample(final_cleaned_map, MAP_VOXEL_SIZE);

            map_cloud.reset();
            final_cleaned_map.reset();

            cout<<"Global map size: "<<global_map->points.size()<<endl;

            // 为全局地图计算法向量
            pcl::PointCloud<pcl::Normal>::Ptr map_normals(new pcl::PointCloud<pcl::Normal>);
            pcl::NormalEstimation<PointT, pcl::Normal> map_norm_est;
            map_norm_est.setInputCloud(global_map);
            pcl::search::KdTree<PointT>::Ptr map_tree(new pcl::search::KdTree<PointT>);
            map_norm_est.setSearchMethod(map_tree);
            map_norm_est.setKSearch(10);
            map_norm_est.compute(*map_normals);

            // 将法向量赋值给全局地图点
            for (size_t i = 0; i < global_map->size() && i < map_normals->size(); ++i) {
                global_map->points[i].normal_x = map_normals->points[i].normal_x;
                global_map->points[i].normal_y = map_normals->points[i].normal_y;
                global_map->points[i].normal_z = map_normals->points[i].normal_z;
            }
        
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
            std::thread localization_thread(&GlobalLocalization::threadLocalization, this);
            localization_thread.detach();
        }
        
        ros::spin();
    }
};

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


    pcl_pub = nh.advertise<sensor_msgs::PointCloud2> ("pcl_output", 10000);

    ROS_INFO("Localization Node Inited...");
    
    cout<<"map_file_path: "<<map_file_path<<endl;
 
    global_localization.run(map_file_path);
    ros::spin(); 
    return 0;
}



