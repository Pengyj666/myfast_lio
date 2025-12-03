#include "laserMapping_controller.h"

std::atomic<int> is_running_={0};  // 算法运行状态  0-停止 1-运行 2-重置算法 3-重置中 
mutex is_running_mutex;
FastLIOController::FastLIOController() {
    algorithm_thread_ = std::make_unique<std::thread>(&FastLIOController::algorithmLoop, this);
    ROS_INFO("LIO controller initialized. Ready to start/stop.");
}

FastLIOController::~FastLIOController() {
    stopAlgorithm();  // 析构时确保算法已停止
    if(algorithm_thread_ && algorithm_thread_->joinable()){
        algorithm_thread_->join();
    }
}



// 算法主循环
void FastLIOController::algorithmLoop() {
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    
    ROS_INFO("LIO controller started.-----------------------------------------");
    // 主循环
    while (status)
    {   
        if (flg_exit) break;
        ros::spinOnce();
        {
            std::lock_guard<std::mutex> lock(is_running_mutex);
            int expected = 2;
            if (is_running_.load() == 0 || is_running_.load() == 3) {   //算法未开启，或者正在重置中
                // 如果算法未运行，休眠等待
                rate.sleep();
                continue;
            }else if(is_running_.compare_exchange_strong(expected, 3)) {
                ROS_INFO("LIO controller reset ...");
                // 停止算法并清理资源
                FastLIOController::stopAlgorithm();

                is_running_.store(1);            //重新打开算法
                rate.sleep();
                continue;
            }
        }

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
            map_incremental();      //将特征点添加到地图kdtree 的时候   同步ikdtree到地图
            t5 = omp_get_wtime();
            
            /******* 发布点云数据 *******/
            publish_this();

            //保存地图    这个是按照开关标志  save_map
            // if(save_map)            exportStaticMapExample();  //不用kdtree保存地图
            // save_map_accumulated_cloud();                      //保存地图    已经放入到服务

            /*** 调试变量记录 ***/
            if (runtime_pos_log)    debug_runtime_pos_log(solve_H_time);

            // cout << "t1 at: " << static_cast<long long>((t1 -t0) * 1e6) << " microseconds" << endl;
            // cout << "t2 at: " << static_cast<long long>((t2-t1) * 1e6) << " microseconds" << endl;
            // cout << "t3 at: " << static_cast<long long>((t3 -t2)* 1e6) << " microseconds" << endl;
            // cout << "t4 at: " << static_cast<long long>(t4 * 1e6) << " microseconds" << endl;
            // cout << "t5 at: " << static_cast<long long>((t5 -t3) * 1e6) << " microseconds" << endl;
            double t6 = omp_get_wtime() - t0;
            cout << "t6 at: " << static_cast<long long>(t6 * 1e6) << " us" << endl;
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
}

//释放内存
void FastLIOController::stopAlgorithm() {
    ROS_INFO("Stopping LIO controller...");

    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    p_imu->Reset();
    p_imu-> set_b_first_frame_ (true);
    //重置state_point
    state_point.pos = MTK::vect<3, double>::Zero();     
    state_point.vel = MTK::vect<3, double>::Zero();       
    state_point.bg = MTK::vect<3, double>::Zero();        
    state_point.ba = MTK::vect<3, double>::Zero();       
    state_point.offset_T_L_I = Lidar_T_wrt_IMU;

    state_point.rot = MTK::SO3<double>(Eigen::Matrix3d::Identity()); 
    state_point.offset_R_L_I = Lidar_R_wrt_IMU;       

    Eigen::Vector3d grav_dir(0.0, 0.0, -1.0);  
    state_point.grav = MTK::S2<double, 98090, 10000, 1>(grav_dir);
    kf.change_x(state_point);  
    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P;
    init_P.setIdentity(); // 初始化为单位矩阵

    kf.change_P(init_P);

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    flg_first_scan = true;
    flg_EKF_inited = false;
    lidar_pushed = false;
    scan_count = 0;
    time_log_counter = 0;
    first_lidar_time = 0.0;
    lidar_mean_scantime = 0.0;
    scan_num = 0;

    lidar_buffer.clear();
    imu_buffer.clear();
    time_buffer.clear();
    Measures.imu.clear();
    Measures.lidar.reset();
    save_map = false;
    feats_down_size = 0;

    Localmap_Initialized = false;
    cub_needrm.clear();
    laserCloudOri->clear();
    corr_normvect->clear();
    normvec->clear();
    if (ikdtree.Root_Node != nullptr) {
        ikdtree.clear();
    }
    if(!feats_undistort->empty() || (feats_undistort != NULL)){
        feats_undistort->clear();
    }

    if (accumulated_cloud) {
        accumulated_cloud->clear();
    }
    

    if (!flg_exit){
        init_imu_extrin();
        feats_down_body.reset(new pcl::PointCloud<PointType>());
        path.header.stamp    = ros::Time::now();
        path.header.frame_id ="camera_init";
    }
}