#ifndef LRELOC_FUNCTION_H
#define LRELOC_FUNCTION_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Dense>
#include <thread>
#include <mutex>
#include <atomic>
#include <fstream>
#include <chrono>
#include "ikd-Tree/ikd_Tree.h"
#include "common/timed_queue.h"
#include "icp_3d.h"
// 类型定义
using PointT = pcl::PointXYZINormal;
typedef std::vector<PointT, Eigen::aligned_allocator<PointT>> PointVector;

#ifndef NUM_MATCH_POINTS
#define NUM_MATCH_POINTS 5
#endif

class lreloc_function : public Icp3d {
private:
    double localization_th;
    std::string map_file_path;
    std::string g_map_root_dir;
    double lidar_d;

    
    pcl::PointCloud<PointT>::Ptr global_map;
    utils::TimedQueue<Eigen::Matrix4f> initial_odom_queue;
    utils::TimedQueue<Eigen::Matrix4f> cur_odom_queue;
    
    Eigen::Matrix4f T_map_to_odom;
    pcl::PointCloud<PointT>::Ptr cur_scan;

    std::mutex initial_pose_mutex;
    std::mutex map_mutex;
    std::mutex cur_scan_mutex;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr map_cloud;
    std::atomic<bool> calculating = {false};
    std::atomic<bool> odom_received = {false};
    std::atomic<bool> scan_received = {false};
    std::atomic<bool> pose_received = {false};
    std::atomic<double> cur_scan_time = {0.0};
    std::atomic<bool> initialized = {false};
    std::atomic<double> offsetTs = {0.0};
    std::atomic<int> is_load_map{0};

    std::function<void(std::shared_ptr<Eigen::Matrix4f>)> cb_pub_map_to_odom;

public:
    // 构造函数和析构函数
    lreloc_function(double localization_th_ = 0.3,
                    std::string map_file_path_ = "",
                    std::string g_map_root_dir_ = "/userdata/RobotData/map/",
                    double map_voxel_size_ = 0.2,
                    double lidar_d_ = -6.7);


    virtual ~lreloc_function();

    // 主要功能函数
    bool loadMap();
    bool globalLocalization(const Eigen::Matrix4f& pose_estimation);
    void run();
    void init(double localization_th_,
                std::string map_file_path_,
                std::string g_map_root_dir_,
                double map_voxel_size_,double lidar_d_);
    
    void regPubMapToOdomCallback(const std::function<void(std::shared_ptr<Eigen::Matrix4f>)> pub_map_to_odom_);

    void insert_initial_odom_queue(Eigen::Matrix4f & initial_odom,double time);
    void set_cur_scan(pcl::PointCloud<PointT>::Ptr & cur_scan_);
    bool getCalculating() const { return calculating.load(); }
    double getCurScanTime() const { return cur_scan_time.load(); }
    int get_isLoadMap() const { return is_load_map.load(); }


    void set_isLoadMap(int val) { is_load_map.store(val); }
    void setPoseReceived(bool value) { pose_received.store(value); }
    void setCurScanTime(double time) { cur_scan_time.store(time); }
    void setOdomReceived(bool value) { odom_received.store(value); }
    void setScanReceived(bool value) { scan_received.store(value); }
    void setOffsetTs(double value) { offsetTs.store(value); }
    void setMapFilePath(std::string value) { map_file_path = value; }

    // ICP相关函数 (继承自 Icp3d)
    using Icp3d::pointToPlaneICP;
    using Icp3d::kdtree_bulid;
    using Icp3d::preprocessCloud;
    using Icp3d::computeJacobianCentralDifference;
    using Icp3d::esti_plane;
};

#endif // LRELOC_FUNCTION_H