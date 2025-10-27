#ifndef GLOBALLOCALIZATION_H
#define GLOBALLOCALIZATION_H 

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
#include <nav_msgs/Path.h>

// PCL相关
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <thread>
#include <chrono>
#include <memory>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <deque>
#include <mutex>
#include "ikd_Tree.h"
// #include "GlobalOptimization.h"


#define MD(a,b)  Matrix<double, (a), (b)>
#define VD(a)    Matrix<double, (a), 1>
#define MF(a,b)  Matrix<float, (a), (b)>
#define VF(a)    Matrix<float, (a), 1>



using PointT = pcl::PointXYZINormal;//PointXYZINormal
typedef vector<PointT, Eigen::aligned_allocator<PointT>>  PointVector;

extern ros::Publisher pub_pc_in_map;
extern ros::Publisher pub_submap;
extern ros::Publisher pub_map_to_odom;
extern ros::Subscriber sub_cloud_registered;
extern ros::Subscriber sub_odometry;
extern ros::Publisher pcl_pub;
extern ros::Publisher path_pub;

// 参数
extern double MAP_VOXEL_SIZE;
extern double SCAN_VOXEL_SIZE;
extern double FREQ_LOCALIZATION;
extern double LOCALIZATION_TH;
extern double FOV;
extern double FOV_FAR;
extern std::string map_file_path;

#ifndef NUM_MATCH_POINTS
#define NUM_MATCH_POINTS 5
#endif

extern KD_TREE<PointT> kdtree;

class GlobalLocalization {
private:
    pcl::PointCloud<PointT>::Ptr global_map;
    
    bool initialized = false;
    
    Eigen::Matrix4f T_map_to_odom;


    nav_msgs::Odometry cur_odom;
    pcl::PointCloud<PointT>::Ptr cur_scan;

    Eigen::Matrix4f initial_pose;
    
    std::mutex pose_mutex;
    std::mutex data_mutex;
    bool pose_received = false;
    
    bool odom_received = false;
    bool scan_received = false;
    bool should_initialize = false;
    std::thread odom_path_thread; 
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr map_cloud;
public:
       // 构造函数和析构函数
    GlobalLocalization(ros::NodeHandle& nh);
    ~GlobalLocalization();
    
    // 坐标变换相关函数
    Eigen::Matrix4f poseToMatrix(const geometry_msgs::PoseWithCovarianceStamped& pose_msg);
    Eigen::Matrix4f poseToMatrix(const nav_msgs::Odometry& odom_msg);
    void threadOdomPath(const std::string& file_path);
    
    // 点云处理相关函数
    pcl::PointCloud<PointT>::Ptr voxelDownSample(pcl::PointCloud<PointT>::Ptr cloud, float voxel_size);
    pcl::PointCloud<PointT>::Ptr preprocessCloud(const pcl::PointCloud<PointT>::Ptr& cloud, float voxel_size);

    // 地图相关函数
    pcl::PointCloud<PointT>::Ptr cropGlobalMapInFOV(
        pcl::PointCloud<PointT>::Ptr global_map,
        const Eigen::Matrix4f& pose_estimation,
        const nav_msgs::Odometry& cur_odom);
    
    // ICP相关函数
    template<typename T>
    bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold);
    
    Eigen::Matrix<double, 1, 6> computeJacobianCentralDifference(
        const PointT& point,
        const Eigen::Matrix<float, 4, 1>& plane_params,
        const Eigen::Matrix4f& transformation,
        double step_size = 1e-6);
    
    std::pair<Eigen::Matrix4f, double> pointToPlaneICP(
        const pcl::PointCloud<PointT>::Ptr& source,
        const pcl::PointCloud<PointT>::Ptr& target,
        const Eigen::Matrix4f& initial,
        int max_iterations = 50,
        double transformation_epsilon = 0.005);
    
    // KD-Tree构建函数
    void kdtree_bulid(int scale, pcl::PointCloud<pcl::PointXYZINormal>::Ptr& map_cloud);
    
    // 回调函数
    void cbSaveCurOdom(const nav_msgs::OdometryConstPtr& odom_msg);
    void cbSaveCurScan(const sensor_msgs::PointCloud2ConstPtr& pc_msg);
    void initialPoseCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg);
    
    // 主要功能函数
    bool globalLocalization(const Eigen::Matrix4f& pose_estimation);
    void initializeGlobalMap(const sensor_msgs::PointCloud2ConstPtr& pc_msg);
    void threadLocalization();
    void run(std::string map_file_path);
};


        template<typename T>
        /**
         * @brief 估计一个点集是否构成平面
         * 
         * 该函数使用PCA方法拟合点集到平面，并检验所有点是否都在拟合的平面附近。
         * 平面方程为: ax + by + cz + d = 0
         * 
         * @param pca_result 输出参数，存储拟合得到的平面参数[a, b, c, d]
         * @param point 输入参数，用于拟合平面的点集
         * @param threshold 输入参数，判断点到平面距离的阈值
         * @return bool 如果所有点到平面的距离都小于阈值则返回true，否则返回false
         */
        bool GlobalLocalization::esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold)
        {
            Eigen::Matrix<T, NUM_MATCH_POINTS, 3> A;
            Eigen::Matrix<T, NUM_MATCH_POINTS, 1> b;
            A.setZero();
            b.setOnes();
            b *= -1.0f;

            // 构建线性方程组的系数矩阵A和常数向量b
            // 方程形式为: ax + by + cz = -1
            for (int j = 0; j < NUM_MATCH_POINTS; j++)
            {
                A(j,0) = point[j].x;
                A(j,1) = point[j].y;
                A(j,2) = point[j].z;
            }

            // 使用QR分解求解线性方程组，得到平面的法向量
            Eigen::Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

            // 归一化法向量并计算平面参数
            T n = normvec.norm();
            pca_result(0) = normvec(0) / n;
            pca_result(1) = normvec(1) / n;
            pca_result(2) = normvec(2) / n;
            pca_result(3) = 1.0 / n;

            // 检查所有点到拟合平面的距离是否都在阈值范围内
            for (int j = 0; j < NUM_MATCH_POINTS; j++)
            {
                if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
                {
                    return false;
                }
            }
            return true;
        }


#endif