#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

int main(int argc, char** argv)
{
    // 检查参数数量
    if (argc != 7) {
        ROS_ERROR("Usage: publish_initial_pose x y z yaw pitch roll");
        return 1;
    }

    // 解析命令行参数
    double x = atof(argv[1]);
    double y = atof(argv[2]);
    double z = atof(argv[3]);
    double yaw = atof(argv[4]);
    double pitch = atof(argv[5]);
    double roll = atof(argv[6]);

    // 初始化ROS节点
    ros::init(argc, argv, "publish_initial_pose");
    ros::NodeHandle nh;
    ros::Publisher pub_pose = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/initialpose", 1);

    // 转换欧拉角为四元数
    tf2::Quaternion quat;
    quat.setRPY(roll, pitch, yaw);

    // 创建位姿消息
    geometry_msgs::PoseWithCovarianceStamped initial_pose;
    initial_pose.header.stamp = ros::Time::now();
    initial_pose.header.frame_id = "map";
    
    // 设置位置
    initial_pose.pose.pose.position.x = x;
    initial_pose.pose.pose.position.y = y;
    initial_pose.pose.pose.position.z = z;
    
    // 设置方向（四元数）
    initial_pose.pose.pose.orientation.x = quat.x();
    initial_pose.pose.pose.orientation.y = quat.y();
    initial_pose.pose.pose.orientation.z = quat.z();
    initial_pose.pose.pose.orientation.w = quat.w();

    // 等待1秒确保发布器建立连接
    ros::Duration(1.0).sleep();
    
    // 打印初始位姿信息
    ROS_INFO("Initial Pose: %f %f %f %f %f %f", x, y, z, yaw, pitch, roll);
    
    // 发布位姿
    pub_pose.publish(initial_pose);
    
    return 0;
}