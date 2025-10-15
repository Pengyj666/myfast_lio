#ifndef LASERMAPPING_ROS_H
#define LASERMAPPING_ROS_H


#include <omp.h>
#include <math.h>
#include <thread>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <Eigen/Core>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>

#include <fstream>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include "laserMapping_help.h"
#include "laserMapping_mapping.h"

extern int lidar_type;

void publish_path();
void publish_odometry();
void publish_map();
void publish_effect_world();
void publish_frame_body();
void publish_frame_world( );
void init_param( ros::NodeHandle & nh);
void init_subAndpub( ros::NodeHandle & nh);

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) ;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) ;
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) ;
void publish_this();
void set_geoQuat();
void downSizeFilter();
#endif
