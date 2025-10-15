// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "laserMapping_pch.h"

/**
 * @brief 主函数，负责初始化ROS节点、加载参数、订阅传感器数据并执行激光雷达SLAM主循环。
 * 
 * 该函数完成以下主要任务：
 * 1. 初始化ROS节点 "laserMapping"
 * 2. 从参数服务器加载配置参数
 * 3. 初始化IMU处理模块和卡尔曼滤波器
 * 4. 订阅激光雷达和IMU数据
 * 5. 执行主循环进行点云去畸变、特征匹配、状态估计和地图更新
 * 6. 发布里程计、路径和点云数据
 * 7. 保存轨迹日志和点云地图
 * 
 * @param argc 命令行参数个数
 * @param argv 命令行参数数组
 * @return int 程序退出状态码，正常退出返回0
 */
int main(int argc, char** argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    init_param(nh);
    
     // 设置激光雷达与IMU的外参和协方差
    init_imu_extrin();  

    // 设置视场角相关参数
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    // _featsArray.reset(new PointCloudXYZI());

    /*** ROS订阅和发布初始化 ***/
    init_subAndpub(nh);

    /*** 调试日志文件初始化 ***/
    init_pos_log();
    
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    
    // 主循环
    while (status)
    {   
        if (flg_exit) break;
        ros::spinOnce();
        // 同步激光雷达和IMU数据包
        if(sync_packages()) //拿到雷达结束前时间段内的所有imu数据
        {
            //imu预处理 积分
            if(!imu_pretreatment()) continue;

            double t0,t1,t2,t3,t4,t5,match_start, solve_start;
            
            t0 = omp_get_wtime();

            /*** 根据激光雷达视场角分割地图 ***/
            lasermap_fov_segment();

            /*** 对扫描中的特征点进行降采样 ***/
            downSizeFilter();

            feats_down_size = feats_down_body->points.size();
            
            if(!init_kdtree()) continue;
            
            /*** ICP和迭代卡尔曼滤波更新 ***/
            // ROS_INFO("Downsampled points: %d", feats_down_size);
            if (feats_down_size < 5)
            {
                ROS_WARN("ICP No point, skip this scan {feats_down_size}!\n");
                //ROS_WARN("Original undistorted points: %zu", feats_undistort->points.size());
                //ROS_WARN("Original undistorted points: %zu", feats_undistort->points.size());
                continue;
            }
            
            t1 = omp_get_wtime();
            resizePointCloud(); 
            t2 = omp_get_wtime();
            
            /*** 迭代状态估计 ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;

            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            set_geoQuat();


            double t_update_end = omp_get_wtime();
            // cout<< "t_update_end - t_update_start: "<< static_cast<long long>((t_update_end -t_update_start)* 1e6) <<endl;

            /******* 发布里程计信息 *******/
            publish_odometry();

            /*** 将特征点添加到地图kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* 发布点云数据 *******/
            publish_this();

            //保存地图    这个是按照开关标志  save_map
            if(save_map)            exportStaticMapExample();
            save_map_accumulated_cloud(); 

            /*** 调试变量记录 ***/
            if (runtime_pos_log)    debug_runtime_pos_log(solve_H_time);

            // cout << "t1 at: " << static_cast<long long>((t1 -t0) * 1e6) << " microseconds" << endl;
            // cout << "t2 at: " << static_cast<long long>((t2-t1) * 1e6) << " microseconds" << endl;
            // cout << "t3 at: " << static_cast<long long>((t3 -t2)* 1e6) << " microseconds" << endl;
            // cout << "t4 at: " << static_cast<long long>(t4 * 1e6) << " microseconds" << endl;
            // cout << "t5 at: " << static_cast<long long>((t5 -t3) * 1e6) << " microseconds" << endl;
            double t6 = omp_get_wtime() - t0;
            cout << "t6 at: " << static_cast<long long>(t6 * 1e6) << " microseconds" << endl;
            // cout << "s_plot11 at: " << static_cast<long long>(s_plot11[0]* 1e6) << " microseconds" << endl;
        }

        status = ros::ok();
        rate.sleep();
    }

    /* 1. 确保有足够的内存  这个是按照帧率数量保存
    /* 2. pcd保存会严重影响实时性能 **/ 
    save_map_PclWaitSave();
        // 保存时间日志
    if (runtime_pos_log)  save_runtime_pos_log();

    
    close_pos_log();   
    return 0;
}
