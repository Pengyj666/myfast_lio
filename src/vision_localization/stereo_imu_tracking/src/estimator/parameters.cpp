/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

#include "droslog/log.h"
using namespace utils;

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;

double vio_memory_threshold;
double imu_pose_time_offset;

std::atomic<int> VIO_FRAME_INDEX;  // 用来标识vio是否重置
std::atomic<int> VIO_FRAME_INDEX2;

utils::OffsetTimer offset_timer;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("readParameters(): config_file dosen't exist; wrong config_file path: %s", config_file.c_str());
        ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        ROS_ERROR("readParameters(): ERROR: Wrong path to settings");
        return;
    }

    fsSettings["vio_memory_threshold"] >> vio_memory_threshold;
    droslog(LogLevel::INFO, "readParameters(): vio memory threshold: %f", vio_memory_threshold);
    fsSettings["imu_pose_time_offset"] >> imu_pose_time_offset;
    droslog(LogLevel::INFO, "readParameters(): imu pose time offset: %f", imu_pose_time_offset);

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_IMU = fsSettings["imu"];
    // printf("USE_IMU: %d\n", USE_IMU);
    droslog(LogLevel::INFO, "readParameters(): USE_IMU: %d", USE_IMU);
    
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        // printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        droslog(LogLevel::INFO, "readParameters(): IMU_TOPIC: %s", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    // std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    droslog(LogLevel::INFO, "readParameters(): vio result path %s", VINS_RESULT_PATH.c_str());
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("readParameters(): have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN("readParameters():  Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
        {
            ROS_WARN("readParameters():  fix extrinsic param ");
        }
    }

    NUM_OF_CAM = fsSettings["num_of_cam"];
    // printf("camera number %d\n", NUM_OF_CAM);
    droslog(LogLevel::INFO, "readParameters(): camera number %d", NUM_OF_CAM);
    
    if(NUM_OF_CAM != 2)
    {
        // printf("num_of_cam must be 2\n");
        droslog(LogLevel::ERROR, "readParameters(): num_of_cam must be 2");
        ROS_BREAK();
        return;
    }
    STEREO = 1;
    
    std::vector<std::string> tmp_fps;
    std::string stereoPath;
    fsSettings["stereo_calib"] >> stereoPath;
    std::string cam0Path;
    fsSettings["cam0_calib"] >> cam0Path;
    std::string cam1Path;
    fsSettings["cam1_calib"] >> cam1Path;

    // 检查相机参数文件是否存在
    tmp_fps.push_back(cam0Path);
    tmp_fps.push_back(cam1Path);
    tmp_fps.push_back(stereoPath);
    for (size_t i = 0; i < tmp_fps.size(); i++)
    {
        FILE *fh = fopen(tmp_fps[i].c_str(),"r");
        if(fh == NULL){
            ROS_WARN("readParameters(): cam param file dosen't exist: %s", tmp_fps[i].c_str());
            ROS_BREAK();
            return;          
        }
        fclose(fh);
    }
    
    // 将左右目的相机参数文件路径存入CAM_NAMES, tracker自身会读取
    CAM_NAMES.push_back(cam0Path);
    CAM_NAMES.push_back(cam1Path);

    // 读取双目外参
    cv::FileStorage stereoFS(stereoPath, cv::FileStorage::READ);
    if(!stereoFS.isOpened())
    {
        // std::cerr << "ERROR: Wrong path to stereo calibration file: " << stereoPath << std::endl;
        ROS_ERROR("readParameters(): Wrong path to stereo calibration file: %s", stereoPath.c_str());
        droslog(LogLevel::ERROR, "readParameters(): Wrong path to stereo calibration file: %s", stereoPath.c_str());
        ROS_BREAK();
        return;
    }
    Eigen::Matrix3d body_R_cam0, body_R_cam1;
    Eigen::Vector3d body_t_cam0, body_t_cam1;
    // 左目使用结构外参
    // 右目通过左目结构外参和双目标定外参计算得到
    body_R_cam0 << -1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, -1.0;
    body_t_cam0 << 0.05143, -0.00453, -0.01503;
    RIC.push_back(body_R_cam0);
    TIC.push_back(body_t_cam0);
    cv::FileNode stereoNode = stereoFS["stereo_params"];
    double roll = static_cast< double > (stereoNode["Rx"]);
    double pitch = static_cast< double > (stereoNode["Ry"]);
    double yaw = static_cast< double > (stereoNode["Rz"]);
    double tx = static_cast< double > (stereoNode["Tx"]) * 0.001;
    double ty = static_cast< double > (stereoNode["Ty"]) * 0.001;
    double tz = static_cast< double > (stereoNode["Tz"]) * 0.001;
    // printf("stereo extrinsic param: rpy=%.8f,%.8f,%.8f, xyz=%.8f,%.8f,%.8f\n", roll, pitch, yaw, tx, ty, tz);
    droslog(LogLevel::INFO, "readParameters(): stereo extrinsic param: rpy=%.8f,%.8f,%.8f, xyz=%.8f,%.8f,%.8f", roll, pitch, yaw, tx, ty, tz);
    Eigen::Matrix3d Rx, Ry, Rz;
    Rx << 1.0, 0.0, 0.0,
        0.0, cos(roll), -sin(roll),
        0.0, sin(roll), cos(roll);
    Ry << cos(pitch), 0.0, sin(pitch),
        0.0, 1.0, 0.0,
        -sin(pitch), 0.0, cos(pitch);
    Rz << cos(yaw), -sin(yaw), 0.0,
        sin(yaw), cos(yaw), 0.0,
        0.0, 0.0, 1.0;
    Eigen::Matrix3d R_rl = Rz * Ry * Rx;
    Eigen::Vector3d t_rl(tx, ty, tz);

    body_R_cam1 = body_R_cam0 * R_rl.inverse();
    body_t_cam1 = body_R_cam0 * (-R_rl.inverse() * t_rl) + body_t_cam0;
    RIC.push_back(body_R_cam1);
    TIC.push_back(body_t_cam1);
    
    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        // printf("no imu, fix extrinsic param; no time offset calibration\n");
        droslog(LogLevel::INFO, "readParameters(): no imu, fix extrinsic param; no time offset calibration");
    }

    fsSettings.release();
}

// void readParameters(std::string config_file)
// {
//     FILE *fh = fopen(config_file.c_str(),"r");
//     if(fh == NULL){
//         ROS_WARN("config_file dosen't exist; wrong config_file path");
//         ROS_BREAK();
//         return;          
//     }
//     fclose(fh);

//     cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
//     if(!fsSettings.isOpened())
//     {
//         std::cerr << "ERROR: Wrong path to settings" << std::endl;
//     }

//     fsSettings["image0_topic"] >> IMAGE0_TOPIC;
//     fsSettings["image1_topic"] >> IMAGE1_TOPIC;
//     MAX_CNT = fsSettings["max_cnt"];
//     MIN_DIST = fsSettings["min_dist"];
//     F_THRESHOLD = fsSettings["F_threshold"];
//     SHOW_TRACK = fsSettings["show_track"];
//     FLOW_BACK = fsSettings["flow_back"];

//     MULTIPLE_THREAD = fsSettings["multiple_thread"];

//     USE_IMU = fsSettings["imu"];
//     printf("USE_IMU: %d\n", USE_IMU);
//     if(USE_IMU)
//     {
//         fsSettings["imu_topic"] >> IMU_TOPIC;
//         printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
//         ACC_N = fsSettings["acc_n"];
//         ACC_W = fsSettings["acc_w"];
//         GYR_N = fsSettings["gyr_n"];
//         GYR_W = fsSettings["gyr_w"];
//         G.z() = fsSettings["g_norm"];
//     }

//     SOLVER_TIME = fsSettings["max_solver_time"];
//     NUM_ITERATIONS = fsSettings["max_num_iterations"];
//     MIN_PARALLAX = fsSettings["keyframe_parallax"];
//     MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

//     fsSettings["output_path"] >> OUTPUT_FOLDER;
//     VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
//     std::cout << "result path " << VINS_RESULT_PATH << std::endl;
//     std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
//     fout.close();

//     ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
//     if (ESTIMATE_EXTRINSIC == 2)
//     {
//         ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
//         RIC.push_back(Eigen::Matrix3d::Identity());
//         TIC.push_back(Eigen::Vector3d::Zero());
//         EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
//     }
//     else 
//     {
//         if ( ESTIMATE_EXTRINSIC == 1)
//         {
//             ROS_WARN(" Optimize extrinsic param around initial guess!");
//             EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
//         }
//         if (ESTIMATE_EXTRINSIC == 0)
//             ROS_WARN(" fix extrinsic param ");

//         cv::Mat cv_T;
//         fsSettings["body_T_cam0"] >> cv_T;
//         Eigen::Matrix4d T;
//         cv::cv2eigen(cv_T, T);
//         RIC.push_back(T.block<3, 3>(0, 0));
//         TIC.push_back(T.block<3, 1>(0, 3));
//     } 
    
//     NUM_OF_CAM = fsSettings["num_of_cam"];
//     printf("camera number %d\n", NUM_OF_CAM);

//     if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
//     {
//         printf("num_of_cam should be 1 or 2\n");
//         assert(0);
//     }


//     int pn = config_file.find_last_of('/');
//     std::string configPath = config_file.substr(0, pn);
    
//     std::string cam0Calib;
//     fsSettings["cam0_calib"] >> cam0Calib;
//     std::string cam0Path = configPath + "/" + cam0Calib;
//     CAM_NAMES.push_back(cam0Path);

//     if(NUM_OF_CAM == 2)
//     {
//         STEREO = 1;
//         std::string cam1Calib;
//         fsSettings["cam1_calib"] >> cam1Calib;
//         std::string cam1Path = configPath + "/" + cam1Calib; 
//         //printf("%s cam1 path\n", cam1Path.c_str() );
//         CAM_NAMES.push_back(cam1Path);
        
//         cv::Mat cv_T;
//         fsSettings["body_T_cam1"] >> cv_T;
//         Eigen::Matrix4d T;
//         cv::cv2eigen(cv_T, T);
//         RIC.push_back(T.block<3, 3>(0, 0));
//         TIC.push_back(T.block<3, 1>(0, 3));
//     }

//     INIT_DEPTH = 5.0;
//     BIAS_ACC_THRESHOLD = 0.1;
//     BIAS_GYR_THRESHOLD = 0.1;

//     TD = fsSettings["td"];
//     ESTIMATE_TD = fsSettings["estimate_td"];
//     if (ESTIMATE_TD)
//         ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
//     else
//         ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

//     ROW = fsSettings["image_height"];
//     COL = fsSettings["image_width"];
//     ROS_INFO("ROW: %d COL: %d ", ROW, COL);

//     if(!USE_IMU)
//     {
//         ESTIMATE_EXTRINSIC = 0;
//         ESTIMATE_TD = 0;
//         printf("no imu, fix extrinsic param; no time offset calibration\n");
//     }

//     fsSettings.release();
// }
