#include "lio_code.h"

LioCode* LioCode::instance_ = nullptr;
mutex is_running_mutex;
LioCode::LioCode(std::shared_ptr<LioHelper>& lio_helper_) {
    instance_ = this;
    lio_helper = lio_helper_;
    lio_helper->init();
}

LioCode::~LioCode() {
    stopAlgorithm();  // 析构时确保算法已停止
    if(algorithm_thread_ && algorithm_thread_->joinable()){
        algorithm_thread_->join();
    }
    if(lio_helper) {
        lio_helper.reset();
    }
}
void LioCode::start(){
    if (!algorithm_thread_ && lio_helper) {
        algorithm_thread_ = std::make_unique<std::thread>(&LioCode::algorithmLoop, this);
        ROS_INFO("LIO controller thread started.");
    }
}
void LioCode::SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}
void LioCode::StaticSigHandle(int sig) {
    if (instance_) {
        instance_->SigHandle(sig);
    }
}

// 算法主循环
void LioCode::algorithmLoop() {
    signal(SIGINT,StaticSigHandle);
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
                stopAlgorithm();

                is_running_.store(1);            //重新打开算法
                rate.sleep();
                continue;
            }
        }
        // 同步激光雷达和IMU数据包
        if(lio_helper->sync_packages()) //拿到雷达结束前时间段内的所有imu数据
        {
            //imu预处理 积分
            if(!lio_helper->imu_pretreatment()) { ROS_INFO("lio_helper->imu_pretreatment()"); continue;}

            double t0,t1,t2,t3,t4,t5,match_start, solve_start;
            
            t0 = omp_get_wtime();

            /*** 根据激光雷达视场角分割地图 ***/
            lio_helper->lasermap_fov_segment();

            /*** 对扫描中的特征点进行降采样 ***/
            lio_helper->downSizeFilter();

            lio_helper->feats_down_size = lio_helper->feats_down_body->points.size();
            
            if(!lio_helper->init_kdtree())  continue;
            
            /*** ICP和迭代卡尔曼滤波更新 ***/
            if (lio_helper->feats_down_size < 5)
            {
                ROS_WARN("ICP No point, skip this scan {feats_down_size}!\n");
                //ROS_WARN("Original undistorted points: %zu", feats_undistort->points.size());
                continue;
            }
            
            t1 = omp_get_wtime();
            lio_helper->resizePointCloud(); 
            t2 = omp_get_wtime();
            
            /*** 迭代状态估计 ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;

            lio_helper->kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            lio_helper->state_point = lio_helper->kf.get_x();
            SO3ToEuler(lio_helper->state_point.rot);
            lio_helper->pos_lid = lio_helper->state_point.pos + lio_helper->state_point.rot * lio_helper->state_point.offset_T_L_I;

            if(lio_helper->cb_set_geoQuat)      lio_helper->cb_set_geoQuat(lio_helper->state_point);


            double t_update_end = omp_get_wtime();
            /******* 发布里程计信息 *******/
            if(lio_helper->cb_pub_odom) lio_helper->cb_pub_odom();

            /*** 将特征点添加到地图kdtree ***/
            t3 = omp_get_wtime();
            lio_helper->map_incremental();      //将特征点添加到地图kdtree 的时候   同步ikdtree到地图
            t5 = omp_get_wtime();
            
            /******* 发布点云数据 *******/
            if(lio_helper->cb_pub_point_cloud) lio_helper->cb_pub_point_cloud();

            // cout << "t1 at: " << static_cast<long long>((t1 -t0) * 1e6) << " microseconds" << endl;
            // cout << "t2 at: " << static_cast<long long>((t2-t1) * 1e6) << " microseconds" << endl;
            // cout << "t3 at: " << static_cast<long long>((t3 -t2)* 1e6) << " microseconds" << endl;
            // cout << "t4 at: " << static_cast<long long>(t4 * 1e6) << " microseconds" << endl;
            // cout << "t5 at: " << static_cast<long long>((t5 -t3) * 1e6) << " microseconds" << endl;
            // double t6 = omp_get_wtime() - t0;
            // cout << "t6 at: " << static_cast<long long>(t6 * 1e6) << " us" << endl;
            // cout << lio_helper->ikdtree.size() << " map points in ikdtree." << endl;
            // cout << lio_helper->ikdtree.validnum() << " validnum points in ikdtree." << endl;
            // cout << lio_helper->imu_buffer.size() << " IMU points in buffer." << endl;
            // cout << lio_helper->lidar_buffer.size() << " lidar_buffer points in first buffer element." << endl;
            // cout << "s_plot11 at: " << static_cast<long long>(s_plot11[0]* 1e6) << " microseconds" << endl;
        }


        status = ros::ok();
        rate.sleep();
    }
}

//释放内存
void LioCode::stopAlgorithm() {
    if(lio_helper)  lio_helper->reset();
}