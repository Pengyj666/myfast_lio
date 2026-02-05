#ifndef LRELOC_NODE_H
#define LRELOC_NODE_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float64.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <tf_conversions/tf_eigen.h>
#include <Eigen/Dense>
#include <mutex>
#include <atomic>
#include <string>
#include <std_msgs/Bool.h>
#include <pcl/filters/voxel_grid.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h> 
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <eigen_conversions/eigen_msg.h>

#include "mower_msgs/Trigger.h"
#include "common/timed_queue.h"
#include "lreloc/lreloc_function.h"
#include "common/sysutils.h"

#include <tf2_eigen/tf2_eigen.h>  
#include <nav_msgs/Odometry.h>    
#include <geometry_msgs/Pose.h>  
#include <Eigen/Geometry>       


// PCL相关类型定义
using PointT = pcl::PointXYZINormal;
typedef std::vector<PointT, Eigen::aligned_allocator<PointT>> PointVector;

#ifndef NUM_MATCH_POINTS
#define NUM_MATCH_POINTS 5
#endif

class lreloc_node {
private:
    // ROS相关成员
    ros::Publisher pub_pc_in_map;
    ros::Publisher pub_submap;
    ros::Publisher pub_map_to_odom;
    ros::Publisher path_pub;
    ros::Subscriber sub_cloud_registered;
    ros::Subscriber sub_odometry;
    ros::Subscriber subOffsetTs;
    ros::Subscriber initial_pose_sub;
    ros::ServiceServer serv_load_mapping_;
    ros::ServiceServer serv_onOroff_relocation_;

    utils::TimedQueue<Eigen::Matrix4f> cur_odom_queue;
    std::unique_ptr<std::thread> odom_path_thread_ptr;
    pcl::PointCloud<PointT>::Ptr cur_scan;

    std::mutex cur_odom_mutex;

    double lidar_d;
    double FREQ_LOCALIZATION;

    std::shared_ptr<lreloc_function> lreloc;
public:
    // 构造函数和析构函数
    lreloc_node(ros::NodeHandle& nh);
    ~lreloc_node();

    bool init(ros::NodeHandle& nh);
    void run();
    void reset();
    // 回调函数
    void cbSaveCurOdom(const nav_msgs::OdometryConstPtr& odom_msg);
    void callback_lio_offset_ts(const std_msgs::Float64::ConstPtr &msg);
    void cbSaveCurScan(const sensor_msgs::PointCloud2ConstPtr& pc_msg);
    void initialPoseCallback(const nav_msgs::OdometryConstPtr& msg);
    bool loadMapCallback(mower_msgs::TriggerRequest &req, mower_msgs::TriggerResponse &res);
    bool onoroff_relocation(mower_msgs::TriggerRequest &req, mower_msgs::TriggerResponse &res);


    // 发布函数
    void threadOdomPath(const std::string& file_path);

    void pub_mapToOdom(Eigen::Matrix4f T_map_to_odom);


};

#endif // LRELOC_NODE_H