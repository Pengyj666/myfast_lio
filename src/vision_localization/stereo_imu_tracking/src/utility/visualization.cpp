/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "visualization.h"
#include <sensor_msgs/point_cloud_conversion.h>

#include "common/log_filters.h"
#include "common/offset_timer.h"
#include "common/sysutils.h"
#include "common/timed_queue.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"

const bool k_b_pub_point_cloud = false;

using namespace utils; 
using namespace ros;
using namespace Eigen;
ros::Publisher pub_odometry, pub_latest_odometry, pub_filtered_odometry;
ros::Publisher pub_vio_pose_result;
ros::Publisher pub_path;
ros::Publisher pub_point_cloud, pub_margin_cloud;
ros::Publisher pub_point_cloud2, pub_margin_cloud2;
ros::Publisher pub_key_poses;
ros::Publisher pub_camera_pose;
ros::Publisher pub_camera_pose_visual;
nav_msgs::Path path;

ros::Publisher pub_keyframe_pose;
ros::Publisher pub_keyframe_point, pub_keyframe_point2;
ros::Publisher pub_extrinsic;

ros::Publisher pub_image_track;

CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);

size_t pub_counter = 0;

std::mutex last_imu_poses_mutex;
utils::TimedQueue<nav_msgs::Odometry> last_imu_poses;
std::mutex last_vio_poses_mutex;
utils::TimedQueue<nav_msgs::Odometry> last_vio_poses;

class Vio2ASImpl {
public:
Vio2ASImpl() {
    // q_l = Eigen::Quaterniond((Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ())).toRotationMatrix());// 冗余 251209
    // Eigen 内部提供了从 AngleAxis 直接构造四元数的构造函数，会自动调用 Quaterniond(const AngleAxisd&) 构造函数  251209
    q_l = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ());
    // static Eigen::Matrix3d ric; // 同相机配置文件 body_T_cam.R
    // static Eigen::Matrix3d q_as2vio;
    // q_as2vio << 0, -1, 0, 0, 0, -1, 1, 0, 0;
    // ric << -1, 0, 0, 0, 1, 0, 0, 0, -1;
    // static Eigen::Quaterniond qq(q_as2vio); 
    // static Eigen::Quaterniond qric(ric);
    static Eigen::Matrix3d R_r;
    R_r << 0, 1, 0, 0, 0, -1, -1, 0, 0; // R_r = qric * qq;
    q_r = Eigen::Quaterniond(R_r);
}

Eigen::Vector3d Convert(const Eigen::Vector3d &vio) {
    return q_l * vio;
}

Eigen::Quaterniond Convert(const Eigen::Quaterniond &vio) {
    return q_l * vio * q_r;
}
private:
Eigen::Quaterniond q_l;
Eigen::Quaterniond q_r;
};

static Vio2ASImpl s_vio2as;
Eigen::Vector3d Vio2AS(const Eigen::Vector3d &vio) {
return s_vio2as.Convert(vio);
}

Eigen::Quaterniond Vio2AS(const Eigen::Quaterniond &vio) {
return s_vio2as.Convert(vio);
}

//  Eigen::Vector3d Vio2AS(const Eigen::Vector3d &vio) {
//     static Eigen::Quaterniond q = Eigen::Quaterniond((Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ())).toRotationMatrix());
//     return q * vio;
//  }
//  Eigen::Quaterniond Vio2AS(const Eigen::Quaterniond &vio) {
//     // static Eigen::Matrix3d ric; // 同相机配置文件 body_T_cam.R
//     // static Eigen::Matrix3d q_as2vio;
//     // q_as2vio << 0, -1, 0, 0, 0, -1, 1, 0, 0;
//     // ric << -1, 0, 0, 0, 1, 0, 0, 0, -1;
//     // static Eigen::Quaterniond qq(q_as2vio); 
//     // static Eigen::Quaterniond qric(ric);
//     static Eigen::Matrix3d q_r;
//     q_r << 0, 0, 1, -1, 0, 0, 0, -1, 0;
//     static Eigen::Quaterniond q_l = Eigen::Quaterniond((Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ())).toRotationMatrix());
//     return q_l * vio * qric * qq;
//  }

// 四元数外插值（支持 t < 0 或 t > 1）
Quaterniond slerpExtrap(const Quaterniond& q0, const Quaterniond& q1, double t) {
    // 归一化输入
    Quaterniond q0n = q0.normalized();
    Quaterniond q1n = q1.normalized();

    // 计算点积
    double dot = q0n.dot(q1n);

    // 如果点积为负，反转一个四元数以选择最短路径
    if (dot < 0.0) {
        q1n = Quaterniond(-q1n.w(), -q1n.x(), -q1n.y(), -q1n.z());
        dot = -dot;
    }

    // 如果接近线性，使用 LERP + 归一化（避免数值不稳定）
    const double DOT_THRESHOLD = 0.9995;
    if (dot > DOT_THRESHOLD) {
        Quaterniond result = Quaterniond(
            q0n.w() + t * (q1n.w() - q0n.w()),
            q0n.x() + t * (q1n.x() - q0n.x()),
            q0n.y() + t * (q1n.y() - q0n.y()),
            q0n.z() + t * (q1n.z() - q0n.z())
        );
        return result.normalized();
    }

    // 计算 SLERP（支持外插值）
    double theta = std::acos(dot);       // 两四元数夹角
    double theta_t = theta * t;          // 插值角度（可能超出 θ）
    double sin_theta = std::sin(theta);
    double sin_theta_t = std::sin(theta_t);

    double s0 = std::cos(theta_t) - dot * sin_theta_t / sin_theta;
    double s1 = sin_theta_t / sin_theta;

    Quaterniond result = Quaterniond(
        s0 * q0n.w() + s1 * q1n.w(),
        s0 * q0n.x() + s1 * q1n.x(),
        s0 * q0n.y() + s1 * q1n.y(),
        s0 * q0n.z() + s1 * q1n.z()
    );

    return result.normalized();
}

void registerPub(ros::NodeHandle &n)
{
    pub_latest_odometry = n.advertise<nav_msgs::Odometry>("imu_propagate", 10);
    pub_filtered_odometry = n.advertise<nav_msgs::Odometry>("filtered_odometry", 10);
    pub_path = n.advertise<nav_msgs::Path>("path", 10);
    pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 10);
    pub_vio_pose_result = n.advertise<nav_msgs::Odometry>("vio_pose_result", 10);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud", 10);
    pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("margin_cloud", 10);
    pub_point_cloud2 = n.advertise<sensor_msgs::PointCloud2>("point_cloud2", 10);
    pub_margin_cloud2 = n.advertise<sensor_msgs::PointCloud2>("margin_cloud2", 10);
    pub_key_poses = n.advertise<visualization_msgs::Marker>("key_poses", 10);
    pub_camera_pose = n.advertise<nav_msgs::Odometry>("camera_pose", 10);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 10);
    pub_keyframe_pose = n.advertise<nav_msgs::Odometry>("keyframe_pose", 10);
    pub_keyframe_point = n.advertise<sensor_msgs::PointCloud>("keyframe_point", 10);
    pub_keyframe_point2 = n.advertise<sensor_msgs::PointCloud2>("keyframe_point2", 10);
    pub_extrinsic = n.advertise<nav_msgs::Odometry>("extrinsic", 10);
    pub_image_track = n.advertise<sensor_msgs::Image>("image_track", 10);

    last_imu_poses.reset(128);
    last_vio_poses.reset(16);

    cameraposevisual.setScale(0.5);
    cameraposevisual.setLineWidth(0.05);
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, const Eigen::Vector3d &angV, double t)
{
    double offset_ts = offset_timer.GetEmb_dt();

    nav_msgs::Odometry odometry;
    odometry.header.stamp.fromSec(t + offset_ts);
    odometry.header.frame_id = "world";

    Eigen::Vector3d asP = Vio2AS(P);
    Eigen::Vector3d asV = Vio2AS(V);
    Eigen::Quaterniond asQ = Vio2AS(Q);
    Eigen::Vector3d asAngV(-angV.z(), angV.x(), -angV.y());

    odometry.pose.pose.position.x = asP.x();
    odometry.pose.pose.position.y = asP.y();
    odometry.pose.pose.position.z = asP.z();
    odometry.pose.pose.orientation.x = asQ.x();
    odometry.pose.pose.orientation.y = asQ.y();
    odometry.pose.pose.orientation.z = asQ.z();
    odometry.pose.pose.orientation.w = asQ.w();
    odometry.twist.twist.linear.x = asV.x();
    odometry.twist.twist.linear.y = asV.y();
    odometry.twist.twist.linear.z = asV.z();
    odometry.twist.twist.angular.x = asAngV.x();
    odometry.twist.twist.angular.y = asAngV.y();
    odometry.twist.twist.angular.z = asAngV.z();
    pub_latest_odometry.publish(odometry);
    {
        std::lock_guard<std::mutex> lock(last_imu_poses_mutex);
        last_imu_poses.emplace_back(odometry, odometry.header.stamp.toSec());
    }
}

void pubTrackImage(const cv::Mat &imgTrack, const double t)
{
    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(t);
    sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", imgTrack).toImageMsg();
    pub_image_track.publish(imgTrackMsg);
}


void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    //printf("position: %f, %f, %f\r", estimator.Ps[WINDOW_SIZE].x(), estimator.Ps[WINDOW_SIZE].y(), estimator.Ps[WINDOW_SIZE].z());
    ROS_DEBUG_STREAM("visualization::printStatistics(): position: " << estimator.Ps[WINDOW_SIZE].transpose());
    ROS_DEBUG_STREAM("visualization::printStatistics(): orientation: " << estimator.Vs[WINDOW_SIZE].transpose());
    if (ESTIMATE_EXTRINSIC)
    {
        cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            //ROS_DEBUG("calibration result for camera %d", i);
            ROS_DEBUG_STREAM("visualization::printStatistics(): extirnsic tic: " << estimator.tic[i].transpose());
            ROS_DEBUG_STREAM("visualization::printStatistics(): extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());

            Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
            eigen_T.block<3, 3>(0, 0) = estimator.ric[i];
            eigen_T.block<3, 1>(0, 3) = estimator.tic[i];
            cv::Mat cv_T;
            cv::eigen2cv(eigen_T, cv_T);
            if(i == 0)
                fs << "body_T_cam0" << cv_T ;
            else
                fs << "body_T_cam1" << cv_T ;
        }
        fs.release();
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    ROS_DEBUG("visualization::printStatistics(): vo solver costs: %f ms", t);
    ROS_DEBUG("visualization::printStatistics(): average of time %f ms", sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    ROS_DEBUG("sum of path %f", sum_of_path);
    if (ESTIMATE_TD) {
    static double pre_td = 0.0;
    static long long pre_log_ts = 0;
    double cur_td = estimator.get_td();
    if (GetNow_Steady() - pre_log_ts > 10000) {
        droslog(LogLevel::INFO, "visualization::printStatistics(): td=%.4f", cur_td, pre_td);
        pre_log_ts = GetNow_Steady();
    }
    if (std::abs(cur_td - pre_td) > 0.003) {
        ROS_WARN("visualization::printStatistics(): cam-imu td changed significantly: td=%.4f, pre_td=%.4f", cur_td, pre_td);
        droslog(LogLevel::WARN, "visualization::printStatistics(): cam-imu时间戳偏移估计变化较大: td=%.4f, pre_td=%.4f", cur_td, pre_td);
        pre_td = cur_td;
        }
    }
}

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header)
{
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        ros::Time cur_stamp;
        cur_stamp = ros::Time::now();

        double offset_ts = offset_timer.GetEmb_dt();
        double dts = cur_stamp.toSec() - header.stamp.toSec();
        // cur_stamp.fromSec(header.stamp.toSec() + offset_ts);

        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.stamp = cur_stamp;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "world";
        Vector3d tmp_T = estimator.Ps[WINDOW_SIZE];
        Quaterniond tmp_Q;
        tmp_Q = Quaterniond(estimator.Rs[WINDOW_SIZE]);
        Vector3d tmp_V = estimator.Vs[WINDOW_SIZE];
        Vector3d angV = estimator.latest_gyr_0 - estimator.latest_Bg;
        Eigen::Vector3d asAngV(-angV.z(), angV.x(), -angV.y());

        tmp_T = Vio2AS(tmp_T);
        tmp_Q = Vio2AS(tmp_Q);
        tmp_V = Vio2AS(tmp_V);

        odometry.pose.pose.position.x = tmp_T.x();
        odometry.pose.pose.position.y = tmp_T.y();
        odometry.pose.pose.position.z = tmp_T.z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        odometry.twist.twist.linear.x = tmp_V.x();
        odometry.twist.twist.linear.y = tmp_V.y();
        odometry.twist.twist.linear.z = tmp_V.z();
        odometry.twist.twist.angular.x = asAngV.x();
        odometry.twist.twist.angular.y = asAngV.y();
        odometry.twist.twist.angular.z = asAngV.z();
        pub_odometry.publish(odometry);
        {
            std::lock_guard<std::mutex> lock(last_vio_poses_mutex);
            last_vio_poses.emplace_back(odometry, cur_stamp.toSec());
        }

        nav_msgs::Odometry vio_pose_result_msg;
        // mower_msgs::VioPoseResult vio_pose_result_msg;
        vio_pose_result_msg.header.stamp = cur_stamp;
        vio_pose_result_msg.header.frame_id = "world";
        vio_pose_result_msg.child_frame_id = "world";
        // vio_pose_result_msg.pose_confidence = 3;
        vio_pose_result_msg.pose = odometry.pose;
        vio_pose_result_msg.pose.covariance[0] = double(VIO_FRAME_INDEX.load());
        vio_pose_result_msg.pose.covariance[1] = header.stamp.toSec();
        vio_pose_result_msg.pose.covariance[2] = dts;
        vio_pose_result_msg.pose.covariance[3] = offset_ts;
        VIO_FRAME_INDEX.store(VIO_FRAME_INDEX.load() + 1);
        vio_pose_result_msg.twist = odometry.twist;
        pub_vio_pose_result.publish(vio_pose_result_msg);

        {
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header = header;
            pose_stamped.header.stamp = cur_stamp;
            pose_stamped.header.frame_id = "world";
            pose_stamped.pose = odometry.pose.pose;
            path.header = header;
            path.header.stamp = cur_stamp;
            path.header.frame_id = "world";
            path.poses.push_back(pose_stamped);
            static SimpleLogFilter log_filter(200);
            if (log_filter.Output(GetNow_Steady())) {
                if (path.poses.size() > 15000) {
                    path.poses.erase(path.poses.begin(), path.poses.begin() + 100);
                }
                pub_path.publish(path);
            }
        }

        // write result to file
    //  ofstream foutC(VINS_RESULT_PATH, ios::app);
    //  foutC.setf(ios::fixed, ios::floatfield);
    //  foutC.precision(0);
    //  foutC << header.stamp.toSec() * 1e9 << ",";
    //  foutC.precision(5);
    //  foutC << tmp_T.x() << ","
    //        << tmp_T.y() << ","
    //        << tmp_T.z() << ","
    //        << tmp_Q.w() << ","
    //        << tmp_Q.x() << ","
    //        << tmp_Q.y() << ","
    //        << tmp_Q.z() << ","
    //        << estimator.Vs[WINDOW_SIZE].x() << ","
    //        << estimator.Vs[WINDOW_SIZE].y() << ","
    //        << estimator.Vs[WINDOW_SIZE].z() << "," << endl;
    //  foutC.close();

        static SimpleLogFilter log_filter(5000);
        if (log_filter.Output(GetNow_Steady())) {
            Eigen::Vector3d rpy = GetEulerRPY(tmp_Q);
            droslog(LogLevel::INFO, "visulization::pubOdometry(): time: %f, ros_ts-ts: %.3f, xyz: %f %f %f rpy: %f %f %f\n", 
                header.stamp.toSec(), dts, tmp_T.x(), tmp_T.y(), tmp_T.z(), rpy.x(), rpy.y(), rpy.z());
        }
    }
}

Quaterniond computeDeltaQuaternion(const Vector3d& angular_vel, double dt) {
    double angle = angular_vel.norm() * dt;
    if (angle < 1e-6) {
        return Quaterniond::Identity();
    }
    Vector3d axis = angular_vel.normalized();
    return Quaterniond(AngleAxisd(angle, axis));
}

void pubVioPoseResult()
{
    // return;
    nav_msgs::Odometry vio0, vio1;
    {
        std::lock_guard<std::mutex> lock(last_vio_poses_mutex);
        if (last_vio_poses.size() > 1) {
            vio0 = last_vio_poses[0];
            vio1 = last_vio_poses[1];
        } else {
            return;
        }
    }
    
    double vio0_ts = vio0.header.stamp.toSec();
    double vio1_ts = vio1.header.stamp.toSec();
    
    double cur_ts = ros::Time::now().toSec();
    if (cur_ts - vio1_ts > 0.5) {
        return;
    }
    
    std::vector<nav_msgs::Odometry> t_imu_poses;
    {
        std::lock_guard<std::mutex> lock(last_imu_poses_mutex);
        int idx = 0;
        while (idx < last_imu_poses.size() && last_imu_poses(idx) > vio0_ts) { 
            t_imu_poses.push_back(last_imu_poses[idx]);
            idx++;
        }
    }

    if (t_imu_poses.size() == 0) {
        return;
    }

    // 采用vio外插值
    double time = t_imu_poses[0].header.stamp.toSec();
    double ex_pos_x = (vio0.pose.pose.position.x - vio1.pose.pose.position.x) * (time - vio0_ts) / (vio0_ts - vio1_ts) + vio0.pose.pose.position.x;
    double ex_pos_y = (vio0.pose.pose.position.y - vio1.pose.pose.position.y) * (time - vio0_ts) / (vio0_ts - vio1_ts) + vio0.pose.pose.position.y;
    double ex_pos_z = (vio0.pose.pose.position.z - vio1.pose.pose.position.z) * (time - vio0_ts) / (vio0_ts - vio1_ts) + vio0.pose.pose.position.z;
    Eigen::Quaterniond q0(vio0.pose.pose.orientation.w, vio0.pose.pose.orientation.x, vio0.pose.pose.orientation.y, vio0.pose.pose.orientation.z);
    Eigen::Quaterniond q1(vio1.pose.pose.orientation.w, vio1.pose.pose.orientation.x, vio1.pose.pose.orientation.y, vio1.pose.pose.orientation.z);
    Eigen::Quaterniond ex_q = slerpExtrap(q1, q0, (time - vio1_ts) / (vio0_ts - vio1_ts));
    Eigen::Vector3d ex_pos(ex_pos_x, ex_pos_y, ex_pos_z);    

    // 采用lv+av推算
    Eigen::Vector3d pos(vio0.pose.pose.position.x, vio0.pose.pose.position.y, vio0.pose.pose.position.z);
    Eigen::Quaterniond q(vio0.pose.pose.orientation.w, vio0.pose.pose.orientation.x, vio0.pose.pose.orientation.y, vio0.pose.pose.orientation.z);
    double pose_ts = vio0.header.stamp.toSec();
    for (int i = t_imu_poses.size() - 1; i >= 0; i--) {
        Eigen::Vector3d lv(t_imu_poses[i].twist.twist.linear.x, t_imu_poses[i].twist.twist.linear.y, t_imu_poses[i].twist.twist.linear.z);
        Eigen::Vector3d av(t_imu_poses[i].twist.twist.angular.x, t_imu_poses[i].twist.twist.angular.y, t_imu_poses[i].twist.twist.angular.z);
        double new_ts = t_imu_poses[i].header.stamp.toSec();
        pos += lv * (new_ts - pose_ts);
        Eigen::Quaterniond dq = computeDeltaQuaternion(av, new_ts - pose_ts);
        q = (q * dq).normalized();
        pose_ts = new_ts;
    }

    pos = 0.5 * (pos + ex_pos);
    q = q.slerp(0.5, ex_q);
    
    vio0.pose.pose.position.x = pos.x();
    vio0.pose.pose.position.y = pos.y();
    vio0.pose.pose.position.z = pos.z();
    vio0.pose.pose.orientation.w = q.w();
    vio0.pose.pose.orientation.x = q.x();
    vio0.pose.pose.orientation.y = q.y();
    vio0.pose.pose.orientation.z = q.z();
    vio0.header.stamp.fromSec(pose_ts);

    double offset_ts = offset_timer.GetEmb_dt();
    
    vio0.pose.covariance[0] = double(VIO_FRAME_INDEX2.load());
    vio0.pose.covariance[1] = pose_ts - offset_ts;
    vio0.pose.covariance[2] = cur_ts - pose_ts;
    vio0.pose.covariance[3] = offset_ts;
    VIO_FRAME_INDEX2.store(VIO_FRAME_INDEX2.load() + 1);
    
    pub_filtered_odometry.publish(vio0);

    // double imu0_ts = imu_pose0.header.stamp.toSec();
    // if (cur_ts - vio_ts < 0.5 && cur_ts - imu0_ts < 0.2) {
    //     double factor = std::min(0.95, std::max(0.5, std::abs(0.9 - imu0_ts + vio_ts)));

    //     vio.pose.pose.position.x = factor * vio.pose.pose.position.x + (1 - factor) * imu_pose0.pose.pose.position.x;
    //     vio.pose.pose.position.y = factor * vio.pose.pose.position.y + (1 - factor) * imu_pose0.pose.pose.position.y;
    //     vio.pose.pose.position.z = factor * vio.pose.pose.position.z + (1 - factor) * imu_pose0.pose.pose.position.z;

    //     Eigen::Quaterniond q_vio(vio.pose.pose.orientation.w, vio.pose.pose.orientation.x, vio.pose.pose.orientation.y, vio.pose.pose.orientation.z);
    //     Eigen::Quaterniond q_imu(imu_pose0.pose.pose.orientation.w, imu_pose0.pose.pose.orientation.x, imu_pose0.pose.pose.orientation.y, imu_pose0.pose.pose.orientation.z);
    //     q_vio = q_vio.slerp(1.0-factor, q_imu);
    //     vio.pose.pose.orientation.w = q_vio.w();
    //     vio.pose.pose.orientation.x = q_vio.x();
    //     vio.pose.pose.orientation.y = q_vio.y();
    //     vio.pose.pose.orientation.z = q_vio.z();

    //     vio.twist.twist.linear.x = factor * vio.twist.twist.linear.x + (1 - factor) * imu_pose0.twist.twist.linear.x;
    //     vio.twist.twist.linear.y = factor * vio.twist.twist.linear.y + (1 - factor) * imu_pose0.twist.twist.linear.y;
    //     vio.twist.twist.linear.z = factor * vio.twist.twist.linear.z + (1 - factor) * imu_pose0.twist.twist.linear.z;

    //     vio.pose.covariance[0] = factor;

    //     vio.header.stamp = imu_pose0.header.stamp;
    //     pub_filtered_odometry.publish(vio);
    // }    
}

void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header)
{
    if (estimator.key_poses.size() == 0)
        return;
    visualization_msgs::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "world";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = ros::Duration();

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        geometry_msgs::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = estimator.key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses.publish(key_poses);
}

void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header)
{
    int idx2 = WINDOW_SIZE - 1;

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        int i = idx2;
    //  Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
    //  Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);
        Vector3d P = estimator.Ps[i];
        Quaterniond R = Quaterniond(estimator.Rs[i]);

        P = Vio2AS(P);
        R = Vio2AS(R);

        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();

        pub_camera_pose.publish(odometry);

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        if(STEREO)
        {
            Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[1];
            Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[1]);
            P = Vio2AS(P);
            R = Vio2AS(R);
        //  cameraposevisual.add_pose(P, R);
        }
        cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
    }
}


void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header)
{
    sensor_msgs::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;


    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

        w_pts_i = Vio2AS(w_pts_i);

        geometry_msgs::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);
    }
    if (k_b_pub_point_cloud)
    {
        pub_point_cloud.publish(point_cloud);
    }
    sensor_msgs::PointCloud2 point_cloud2, loop_point_cloud2;
    sensor_msgs::convertPointCloudToPointCloud2(point_cloud, point_cloud2);
    pub_point_cloud2.publish(point_cloud2);

    // pub margined potin
    sensor_msgs::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &it_per_id : estimator.f_manager.feature)
    { 
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
        //        continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 
            && it_per_id.solve_flag == 1 )
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

            w_pts_i = Vio2AS(w_pts_i);

            geometry_msgs::Point32 p;
            p.x = w_pts_i(0);
            p.y = w_pts_i(1);
            p.z = w_pts_i(2);
            margin_cloud.points.push_back(p);
        }
    }
    if (k_b_pub_point_cloud)
    {
        pub_margin_cloud.publish(margin_cloud);
    }
    sensor_msgs::PointCloud2 margin_cloud2;
    sensor_msgs::convertPointCloudToPointCloud2(margin_cloud, margin_cloud2);
    pub_margin_cloud2.publish(margin_cloud2);
}


void pubTF(const Estimator &estimator, const std_msgs::Header &header)
{
    if( estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    // body frame
    Vector3d correct_t;
    Quaterniond correct_q;
    correct_t = estimator.Ps[WINDOW_SIZE];
    correct_q = estimator.Rs[WINDOW_SIZE];

    correct_t = Vio2AS(correct_t);
    correct_q = Vio2AS(correct_q);

    transform.setOrigin(tf::Vector3(correct_t(0),
                                    correct_t(1),
                                    correct_t(2)));
    q.setW(correct_q.w());
    q.setX(correct_q.x());
    q.setY(correct_q.y());
    q.setZ(correct_q.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "body"));

    // camera frame
    transform.setOrigin(tf::Vector3(estimator.tic[0].x(),
                                    estimator.tic[0].y(),
                                    estimator.tic[0].z()));
    q.setW(Quaterniond(estimator.ric[0]).w());
    q.setX(Quaterniond(estimator.ric[0]).x());
    q.setY(Quaterniond(estimator.ric[0]).y());
    q.setZ(Quaterniond(estimator.ric[0]).z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "camera"));

    
    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.tic[0].x();
    odometry.pose.pose.position.y = estimator.tic[0].y();
    odometry.pose.pose.position.z = estimator.tic[0].z();
    Quaterniond tmp_q{estimator.ric[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();
    pub_extrinsic.publish(odometry);

}

void pubKeyframe(const Estimator &estimator)
{
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0)
    {
        int i = WINDOW_SIZE - 2;
        //Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Vector3d P = Vio2AS(estimator.Ps[i]);
        Quaterniond R = Vio2AS(Quaterniond(estimator.Rs[i]));

        nav_msgs::Odometry odometry;
        odometry.header.stamp = ros::Time(estimator.Headers[WINDOW_SIZE - 2]);
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();
        //printf("time: %f t: %f %f %f r: %f %f %f %f\n", odometry.header.stamp.toSec(), P.x(), P.y(), P.z(), R.w(), R.x(), R.y(), R.z());

        pub_keyframe_pose.publish(odometry);


        sensor_msgs::PointCloud point_cloud;
        point_cloud.header.stamp = ros::Time(estimator.Headers[WINDOW_SIZE - 2]);
        point_cloud.header.frame_id = "world";
        for (auto &it_per_id : estimator.f_manager.feature)
        {
            int frame_size = it_per_id.feature_per_frame.size();
            if(it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag == 1)
            {

                int imu_i = it_per_id.start_frame;
                Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0])
                                    + estimator.Ps[imu_i];
                
                w_pts_i = Vio2AS(w_pts_i);
                
                geometry_msgs::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);
                point_cloud.points.push_back(p);

                int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
                sensor_msgs::ChannelFloat32 p_2d;
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.y());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.y());
                p_2d.values.push_back(it_per_id.feature_id);
                point_cloud.channels.push_back(p_2d);
            }

        }
        pub_keyframe_point.publish(point_cloud);
        // sensor_msgs::PointCloud2 point_cloud2;
        // sensor_msgs::convertPointCloudToPointCloud2(point_cloud, point_cloud2);
        // pub_keyframe_point2.publish(point_cloud2);
    }
}
