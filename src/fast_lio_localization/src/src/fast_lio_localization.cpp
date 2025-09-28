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

using namespace std;
using PointT = pcl::PointXYZINormal;
// ROS相关
ros::Publisher pub_pc_in_map;
ros::Publisher pub_submap;
ros::Publisher pub_map_to_odom;
ros::Subscriber sub_cloud_registered;
ros::Subscriber sub_odometry;
ros::Publisher pcl_pub;
// 参数
double MAP_VOXEL_SIZE = 0.4;
double SCAN_VOXEL_SIZE = 0.1;
double FREQ_LOCALIZATION = 5;
double LOCALIZATION_TH = 0.3;
double FOV = M_PI*2;
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
    nav_msgs::Odometry cur_odom;
    pcl::PointCloud<PointT>::Ptr cur_scan;
    deque<pcl::PointCloud<PointT>::Ptr> lidar_buffer = deque<pcl::PointCloud<PointT>::Ptr>();

    boost::shared_ptr<geometry_msgs::PoseWithCovarianceStamped const> initial_pose_msg;
    deque<Eigen::Matrix4f> initial_pose_buffer = deque<Eigen::Matrix4f>();

    std::mutex pose_mutex;
    bool pose_received = false;
    
    bool odom_received = false;
    bool scan_received = false;

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
        pcl::PointCloud<PointT>::Ptr cloud_downsampled(new pcl::PointCloud<PointT>);
        pcl::VoxelGrid<PointT> voxel_grid;
        voxel_grid.setInputCloud(cloud);
        voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
        voxel_grid.filter(*cloud_downsampled);
        return cloud_downsampled;
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
        Eigen::Matrix4f T_odom_to_base_link = poseToMatrix(cur_odom);
        Eigen::Matrix4f T_map_to_base_link = pose_estimation * T_odom_to_base_link;
        Eigen::Matrix4f T_base_link_to_map = inverseSE3(T_map_to_base_link);
        
        // 将全局地图点云变换到LiDAR坐标系下
        pcl::PointCloud<PointT>::Ptr global_map_in_base_link(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*global_map, *global_map_in_base_link, T_base_link_to_map);
        
        // 根据LiDAR的视场角筛选在视角范围内的点
        pcl::PointCloud<PointT>::Ptr global_map_in_FOV(new pcl::PointCloud<PointT>);
        
        for (const auto& point : global_map_in_base_link->points) {
            bool in_fov = false;
            if (FOV > 3.14) {
                if (point.x < FOV_FAR && 
                    std::abs(std::atan2(point.y, point.x)) < FOV / 2.0) {
                    in_fov = true;
                }
            } else {
                // 非环状LiDAR：保留前方扇形区域内的点
                if (point.x > 0 && point.x < FOV_FAR && 
                    std::abs(std::atan2(point.y, point.x)) < FOV / 2.0) {
                    in_fov = true;
                }
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
std::pair<Eigen::Matrix4f, double> registrationAtScale1(const pcl::PointCloud<PointT>::Ptr& scan,
                                                       const pcl::PointCloud<PointT>::Ptr& map,
                                                       const Eigen::Matrix4f& initial,
                                                       double scale) {
            cout<<"map size: "<<map->points.size()<<endl;                                    
            // 根据尺度参数对点云进行降采样处理
            pcl::PointCloud<PointT>::Ptr scan_ds = voxelDownSample(scan, SCAN_VOXEL_SIZE * scale );
            pcl::PointCloud<PointT>::Ptr map_ds = voxelDownSample(map, MAP_VOXEL_SIZE * scale );
            // cout<<"voxelDownSample map size: "<<map_ds->points.size()<<endl;  

            // 配置ICP算法参数
            pcl::IterativeClosestPoint<PointT, PointT> icp;
            icp.setMaximumIterations(30);
            icp.setMaxCorrespondenceDistance(0.4 * scale);
            // icp.setTransformationEpsilon(1e-8);
            // icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setInputSource(scan_ds);
            icp.setInputTarget(map_ds);

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
    cout<<"scan size: "<<scan->points.size()<<endl;            
    
    // 根据尺度参数对点云进行降采样处理
    pcl::PointCloud<PointT>::Ptr scan_ds = voxelDownSample(scan, SCAN_VOXEL_SIZE * scale);
    pcl::PointCloud<PointT>::Ptr map_ds = voxelDownSample(map, MAP_VOXEL_SIZE * scale );
    if (scan_ds->empty()) {
        ROS_WARN("Downsampled scan is empty");
        return {Eigen::Matrix4f::Identity(), std::numeric_limits<double>::max()};
    }                      

    // 使用NDT算法替代ICP
    pcl::NormalDistributionsTransform<PointT, PointT> ndt;
    ndt.setMaximumIterations(30);
    ndt.setResolution(0.8 * scale);  // 设置NDT分辨率
    ndt.setStepSize(0.1);            // 设置牛顿法步长
    ndt.setInputSource(scan_ds);
    ndt.setInputTarget(map_ds);

    // 执行NDT配准
    pcl::PointCloud<PointT> output_cloud;
    Eigen::Matrix4f init = initial;
    ndt.align(output_cloud, init);

    // 获取配准结果
    Eigen::Matrix4f transformation = ndt.getFinalTransformation();
    double fitness = ndt.getFitnessScore();
    cout<<"fitness: "<<fitness<<endl;
    return {transformation, fitness};
}

    bool flag = true;
    bool globalLocalization(const Eigen::Matrix4f& pose_estimation) {

        if(lidar_buffer.empty() || lidar_buffer.size() < 1) {
            return false;
        }
        ROS_INFO("Global localization by scan-to-map matching......");
        
        auto tic = std::chrono::high_resolution_clock::now();
        
        // pcl::PointCloud<PointT>::Ptr global_map_in_FOV = 
        //     cropGlobalMapInFOV(global_map, pose_estimation, cur_odom);
        
        // 粗配准
        auto coarse_result = registrationAtScale1(lidar_buffer.front(), global_map, pose_estimation, 10.0);
        Eigen::Matrix4f transformation = coarse_result.first;
        double coarse_fitness = coarse_result.second;
        
        // 精配准
        auto fine_result = registrationAtScale1(lidar_buffer.front(), global_map, transformation, 1.0);
        transformation = fine_result.first;
        double fitness = fine_result.second;
        
        auto toc = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
        // ROS_INFO("Time: %ld ms", duration.count());
        
        // 当全局定位成功时才更新map2odom
        if (fitness < LOCALIZATION_TH) { 
            T_map_to_odom = transformation;
            
            // 发布map_to_odom
            nav_msgs::Odometry map_to_odom;
            Eigen::Affine3f affine(T_map_to_odom);
            tf::poseEigenToMsg(Eigen::Affine3d(affine.cast<double>()), map_to_odom.pose.pose);
            map_to_odom.header.stamp = cur_odom.header.stamp;
            map_to_odom.header.frame_id = "map";
            pub_map_to_odom.publish(map_to_odom);
            cout<<"T_map_to_odom: \n"<<T_map_to_odom<<endl;
            ROS_WARN("!!! Global localization success !!!");
            return true;
        } else {
            // ROS_WARN("Not match!!!!");
            ROS_INFO("fitness score: %f", fitness);
            ROS_INFO("coarse_fitness score: %f", coarse_fitness);
            return false;
        }
    }
    
    void initializeGlobalMap(const sensor_msgs::PointCloud2ConstPtr& pc_msg) {
        pcl::fromROSMsg(*pc_msg, *global_map);
        //global_map = voxelDownSample(global_map, MAP_VOXEL_SIZE);
        ROS_INFO("Global map received.");
    }
    
    void cbSaveCurOdom(const nav_msgs::OdometryConstPtr& odom_msg) {
        cur_odom = *odom_msg;
        odom_received = true;
    }
    
    void cbSaveCurScan(const sensor_msgs::PointCloud2ConstPtr& pc_msg) {
        // std::cout << "Debug: File=" << __FILE__ << ", Line=" << __LINE__ << ", Function=" << __FUNCTION__ << std::endl;
        sensor_msgs::PointCloud2 modified_msg = *pc_msg;
        modified_msg.header.frame_id = "camera_init";
        modified_msg.header.stamp = ros::Time::now();
        pub_pc_in_map.publish(modified_msg);
        
        pcl::PointCloud<PointT>::Ptr cur_lidar_scan (new pcl::PointCloud<PointT>());
        pcl::fromROSMsg(*pc_msg, *cur_lidar_scan);
        lidar_buffer.push_back(cur_lidar_scan);
        scan_received = true;
    }
    
    void threadLocalization() {
        ros::Rate rate(FREQ_LOCALIZATION);
        while (ros::ok()) {
            rate.sleep();
            if (initialized) {
                globalLocalization(T_map_to_odom);
            }
        }
    }
    // 回调函数
    void initialPoseCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg) {
        std::lock_guard<std::mutex> lock(pose_mutex);
        // cout<<"Initial pose received."<<endl;
        // initial_pose_msg = msg;
        Eigen::Matrix4f initial_pose = poseToMatrix(*msg);
        initial_pose_buffer.push_back(initial_pose);
        pose_received = true;
    }


    void run(std::string map_file_path ) {
        ros::NodeHandle nh;
        
        // // 等待全局地图
        // ROS_WARN("Waiting for global map......");
        // boost::shared_ptr<sensor_msgs::PointCloud2 const> map_msg = 
        //     ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/map");

        //     int retry_count = 0;
        //     const int max_retries = 5;
            
        //     while (retry_count < max_retries && ros::ok()) {
        //         map_msg = ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/map", ros::Duration(10.0));
                
        //         if (map_msg) {
        //             break;
        //         }
                
        //         retry_count++;
        //         ROS_WARN("Failed to receive global map, retrying... (%d/%d)", retry_count, max_retries);
        //     }
            map_file_path = "/home/edy/code/lidarcode/src/PCD/"+map_file_path;
            
            // 检查地图文件是否存在
            std::ifstream file(map_file_path);
            if (!file.good()) {
                std::cout << "No existing map found at: " << map_file_path << std::endl;
                return;
            }
            file.close();
            
            // 加载点云地图
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZINormal>());
            sensor_msgs::PointCloud2 rviz_output;

            if (pcl::io::loadPCDFile<pcl::PointXYZINormal>(map_file_path, *map_cloud) != 0) {
                std::cout << "Failed to load map from: " << map_file_path << std::endl;
                return;
            }else{
                pcl::toROSMsg(*map_cloud, rviz_output);
                rviz_output.header.frame_id = "map";

                ros::spinOnce();
                pcl_pub.publish(rviz_output);
                //global_map = map_cloud;
                global_map = voxelDownSample(map_cloud, MAP_VOXEL_SIZE);;
            }
        
            int number = 0;
        // initializeGlobalMap(map_msg);
        ros::Rate rate(5000);  
        // 初始化
        while (!initialized && ros::ok()) {
            ros::spinOnce();
            // ROS_WARN("Waiting for initial pose....");
            
            // boost::shared_ptr<geometry_msgs::PoseWithCovarianceStamped const> pose_msg = 
            //     ros::topic::waitForMessage<geometry_msgs::PoseWithCovarianceStamped>("/initialpose");

            // if (scan_received) {
            //     std::cout << "Debug: File=" << __FILE__ << ", Line=" << __LINE__ << ", Function=" << __FUNCTION__ << std::endl;
            //     Eigen::Matrix4f initial_pose = poseToMatrix(*pose_msg);
            //     initialized = globalLocalization(initial_pose);
            // } else {
            //     ROS_WARN("First scan not received!!!!!");
            // }

            std::lock_guard<std::mutex> lock(pose_mutex);
            // if (pose_received && scan_received) {
            //     pose_received = false; // 重置标志
            //     scan_received = false; // 重置标志
                // Eigen::Matrix4f initial_pose = poseToMatrix(*initial_pose_msg);
            if(initial_pose_buffer.size() > 0 && lidar_buffer.size() > 0) {
                auto t1 = std::chrono::high_resolution_clock::now();
                bool temp = globalLocalization(initial_pose_buffer.front());
                initial_pose_buffer.pop_front();
                lidar_buffer.pop_front();
                if (temp) {
                    number++;
                } 
                cout<<"number: "<<number<<endl;
                auto t2 = std::chrono::high_resolution_clock::now();
                auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
                ROS_INFO("initialization Time: %ld ms", t3.count());
            }

            // }

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



#if 0
// Converted from provided Python script to C++
//#include <ros/ros.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>


using PointT = pcl::PointXYZINormal;

// 全局点云地图指针，用于存储构建的全局地图数据
static pcl::PointCloud<PointT>::Ptr global_map(new pcl::PointCloud<PointT>());

// 全局互斥锁，用于保护多线程环境下对共享数据的访问
static std::mutex g_mutex;

// 原子布尔变量，标识系统是否已完成初始化
static std::atomic<bool> initialized(false);

// 4x4变换矩阵，表示从地图坐标系到里程计坐标系的变换关系
static Eigen::Matrix4f T_map_to_odom = Eigen::Matrix4f::Identity();

// 当前扫描点云数据指针，存储最新的激光雷达扫描数据
static pcl::PointCloud<PointT>::Ptr cur_scan(new pcl::PointCloud<PointT>());

// 地图体素大小，用于地图构建时的体素滤波器尺寸
static double MAP_VOXEL_SIZE = 0.4;

// 扫描体素大小，用于扫描数据处理时的体素滤波器尺寸
static double SCAN_VOXEL_SIZE = 0.3;

// 定位阈值，用于判断定位是否成功的置信度阈值  (值越小，要求越严格)
static double LOCALIZATION_TH = 0.3;

// 视场角，定义传感器的水平视场角范围(单位: 弧度)
static double FOV =  M_PI; // 180度

// 远距离视场范围，定义传感器的最大探测距离(单位: 米)
static double FOV_FAR = 50.0;

// 如果为 true，则在成功定位后将地图原点重定位到传入的 pose_estimation（即把世界坐标系原点设置为 pose_estimation）
static bool USE_POSE_AS_ORIGIN = true;
// 标记是否已执行过一次重定位，避免重复变换地图
static bool map_rebased = true;

Eigen::Matrix4f inverseSE3(const Eigen::Matrix4f& T) {
    Eigen::Matrix4f inv = Eigen::Matrix4f::Identity();
    inv.block<3,3>(0,0) = T.block<3,3>(0,0).transpose();
    inv.block<3,1>(0,3) = -inv.block<3,3>(0,0) * T.block<3,1>(0,3);
    return inv;
}
/**
 * @brief 将全局地图的原点重定位到给定的 pose_estimation
 * 
 * 实现思路：将全局地图中的所有点使用 pose_estimation 的逆变换进行变换，
 * 使得在新的地图坐标系中，原本由 pose_estimation 指定的位置变为坐标原点。
 * 同时更新内部保存的 T_map_to_odom 以保持坐标变换一致性。
 */
void setMapOriginFromPose(const Eigen::Matrix4f& pose_estimation) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!global_map || global_map->empty()) return;

    // 将地图点云从旧地图系变换到新的地图系（新的地图系使 pose_estimation 对应的点位于原点）
    Eigen::Matrix4f inv = inverseSE3(pose_estimation);
    pcl::transformPointCloud(*global_map, *global_map, inv);

    // 更新 T_map_to_odom：如果原来地图到里程计的变换为 T_map_to_odom_old，那么新的地图到里程计变换为
    // T_map'_to_odom = T_map_to_odom_old * pose_estimation
    T_map_to_odom = T_map_to_odom * pose_estimation;
}

/**
 * @brief 对点云进行体素下采样
 * 
 * 使用pcl::VoxelGrid滤波器对输入点云进行下采样处理，通过将点云划分为规则的三维体素网格，
 * 并用每个体素内的重心点代替该体素内的所有点，从而减少点云数据量
 * 
 * @param pc 输入的点云指针
 * @param voxel_size 体素网格的大小（立方体边长），单位为米
 * @return pcl::PointCloud<PointT>::Ptr 下采样后的点云指针
 */
pcl::PointCloud<PointT>::Ptr voxelDownSample(const pcl::PointCloud<PointT>::Ptr& pc, double voxel_size) {
    // 创建下采样后的点云对象
    pcl::PointCloud<PointT>::Ptr down(new pcl::PointCloud<PointT>());
    
    // 创建体素网格滤波器并执行下采样
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud(pc);
    sor.setLeafSize(voxel_size, voxel_size, voxel_size);
    sor.filter(*down);
    
    return down;
}

// publish pcl cloud as PointCloud2
void publishPointCloud(ros::Publisher& pub, const std_msgs::Header& header, const pcl::PointCloud<PointT>::Ptr& pc) {
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*pc, msg);
    msg.header = header;
    pub.publish(msg);
}

// inverse transform


// crop global map in FOV similar to python function
/**
 * @brief 从全局地图中裁剪出当前传感器视场（FOV）范围内的点云子图
 * 
 * 该函数首先将全局地图变换到机器人当前位姿的base_link坐标系下，
 * 然后根据设定的视场角和距离筛选出在视场内的点，
 * 最后将这些点变换回地图坐标系并返回。
 * 同时会发布一个降采样后的子图用于可视化。
 *
 * @param global_map_ptr 全局地图点云的指针
 * @param pose_estimation 估计的位姿变换矩阵 T_map_to_odom
 * @param cur_odom_ptr 当前里程计信息，包含机器人在odom坐标系下的位姿
 * @return 裁剪出的视场内子图点云指针（在地图坐标系下）
 */
pcl::PointCloud<PointT>::Ptr cropGlobalMapInFOV(const pcl::PointCloud<PointT>::Ptr& global_map_ptr,
                                                const Eigen::Matrix4f& pose_estimation,
                                                state_ikfom & state_point, geometry_msgs::Quaternion & geoQuat) {
    // 构造从odom到base_link的变换矩阵T_odom_to_base_link
    Eigen::Matrix4f T_odom_to_base_link = Eigen::Matrix4f::Identity();
    T_odom_to_base_link.block<3,3>(0,0) = Eigen::Quaternionf(
        geoQuat.w,
        geoQuat.x,
        geoQuat.y,
        geoQuat.z).toRotationMatrix();
    T_odom_to_base_link(0,3) = state_point.pos(0);
    T_odom_to_base_link(1,3) = state_point.pos(1);
    T_odom_to_base_link(2,3) = state_point.pos(2);

    // 计算从map到base_link的变换矩阵，并求其逆变换（即base_link到map）
    Eigen::Matrix4f T_map_to_base_link = pose_estimation * T_odom_to_base_link;
    Eigen::Matrix4f T_base_link_to_map = inverseSE3(T_map_to_base_link);

    // 将全局地图变换到base_link坐标系下
    pcl::PointCloud<PointT>::Ptr map_in_base(new pcl::PointCloud<PointT>());
    pcl::transformPointCloud(*global_map_ptr, *map_in_base, T_base_link_to_map);

    // 在base_link坐标系下筛选视场范围内的点
    pcl::PointCloud<PointT>::Ptr in_fov(new pcl::PointCloud<PointT>());
    for (const auto& pt : map_in_base->points) {
        double x = pt.x;
        double y = pt.y;
        double r = hypot(x, y);
        double ang = atan2(y, x);
        if (FOV > 3.14) {
            if (r < FOV_FAR && std::abs(ang) < FOV / 2.0) {
                in_fov->points.push_back(pt);
            }
        } else {
            if (x > 0 && r < FOV_FAR && std::abs(ang) < FOV / 2.0) {
                in_fov->points.push_back(pt);
            }
        }
    }

    // 将筛选出的点变换回地图坐标系
    pcl::PointCloud<PointT>::Ptr submap_in_map(new pcl::PointCloud<PointT>());
    pcl::transformPointCloud(*in_fov, *submap_in_map, T_map_to_base_link);

    return submap_in_map;
}

/**
 * @brief 在指定尺度下执行点云配准
 * 
 * 该函数通过对输入点云进行降采样，然后使用ICP算法进行配准，
 * 返回配准变换矩阵和配准质量评分。
 * 
 * @param scan 待配准的扫描点云
 * @param map 作为参考的地图点云
 * @param initial 初始变换矩阵
 * @param scale 配准尺度参数，用于调整降采样体素大小和最大对应距离
 * @return std::pair<Eigen::Matrix4f, double> 
 *         - first: 配准后的变换矩阵
 *         - second: 配准适应度评分，值越小表示配准质量越好
 */
std::pair<Eigen::Matrix4f, double> registrationAtScale(const pcl::PointCloud<PointT>::Ptr& scan,
                                                       const pcl::PointCloud<PointT>::Ptr& map,
                                                       const Eigen::Matrix4f& initial,
                                                       double scale) {
    // 根据尺度参数对点云进行降采样处理
    pcl::PointCloud<PointT>::Ptr scan_ds = voxelDownSample(scan, SCAN_VOXEL_SIZE * scale);
    pcl::PointCloud<PointT>::Ptr map_ds = voxelDownSample(map, MAP_VOXEL_SIZE * scale);

    // 配置ICP算法参数
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setMaximumIterations(20);
    icp.setMaxCorrespondenceDistance(0.4 * scale);
    icp.setInputSource(scan_ds);
    icp.setInputTarget(map_ds);

    // 执行ICP配准
    pcl::PointCloud<PointT> Final;
    Eigen::Matrix4f init = initial;
    icp.align(Final, init);

    // 获取配准结果
    Eigen::Matrix4f transformation = icp.getFinalTransformation();
    double fitness = icp.getFitnessScore();
    return {transformation, fitness};
}

bool globalLocalization(const Eigen::Matrix4f& pose_estimation,state_ikfom & state_point, geometry_msgs::Quaternion & geoQuat) {
    std::lock_guard<std::mutex> lock(g_mutex);
    cout << global_map->size() << " " << cur_scan->size() << endl;
    if (!global_map || global_map->empty() || cur_scan->empty() ) return false;
    ROS_INFO("Global localization by scan-to-map matching......");

    pcl::PointCloud<PointT>::Ptr scan_copy(new pcl::PointCloud<PointT>());
    *scan_copy = *cur_scan;

    ros::Time tic = ros::Time::now();
    // 提取当前估计位姿附近的局部子地图，以提高配准效率
    auto submap = cropGlobalMapInFOV(global_map, pose_estimation, state_point,geoQuat);

    // coarse
    Eigen::Matrix4f transformation;
    double fitness;
    // 粗配准：使用较大尺度进行初步匹配
    std::tie(transformation, std::ignore) = registrationAtScale(scan_copy, submap, pose_estimation, 5.0);

    // 精配准：在粗配准结果基础上使用较小尺度进行精细匹配
    std::tie(transformation, fitness) = registrationAtScale(scan_copy, submap, transformation, 1.0);

    ros::Time toc = ros::Time::now();
    ROS_INFO("Time: %f", (toc - tic).toSec());

    if (fitness < LOCALIZATION_TH) {
        // 当定位成功，根据设置决定是否将地图原点重定位到 pose_estimation
        T_map_to_odom = transformation;

        if (USE_POSE_AS_ORIGIN && !map_rebased) {
            // 把世界坐标系的原点设置为 pose_estimation 所指的位置
            setMapOriginFromPose(pose_estimation);
            map_rebased = true;
            ROS_INFO("Map origin rebased to pose_estimation.");
        }

        nav_msgs::Odometry map_to_odom;
        Eigen::Matrix3f R = T_map_to_odom.block<3,3>(0,0);
        Eigen::Quaternionf q(R);
        Eigen::Vector3f t = T_map_to_odom.block<3,1>(0,3);
        // 更新旋转部分
        state_point.rot = Eigen::Quaternionf(R);
        state_point.pos(0) = t.x();
        state_point.pos(1) = t.y();
        state_point.pos(2) = t.z();
        geoQuat.x = q.x();
        geoQuat.y = q.y();
        geoQuat.z = q.z();
        geoQuat.w = q.w();
        return true;
    } else {
        ROS_WARN("Poor fitness: %f", fitness);
        return false;
    }
}

void init_localization(const PointCloudXYZI::Ptr& feats_undistort,double map_voxel_size,double scan_voxel_size,float fov_degree,pcl::PointCloud<PointT>::Ptr&  down_map) {
    std::lock_guard<std::mutex> lock(g_mutex);

    cout << feats_undistort->size() << endl;
    pcl::PointCloud<PointT>::Ptr pc = feats_undistort;
    cout << pc->size() << endl;
    MAP_VOXEL_SIZE = map_voxel_size;
    SCAN_VOXEL_SIZE = scan_voxel_size;
    FOV = fov_degree*M_PI/180;                          //角度转弧度
    global_map = down_map;
    cur_scan = voxelDownSample(pc, SCAN_VOXEL_SIZE);
}

#endif