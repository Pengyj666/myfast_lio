#ifndef LAERMAPPING_CONTROLLER_H
#define LAERMAPPING_CONTROLLER_H 

#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "IMU_Processing.h"
#include "laserMapping_help.h"


extern std::atomic<int> is_running_ ;  // 算法运行状态

class FastLIOController {
private:
    std::unique_ptr<std::thread> algorithm_thread_;            // 算法主线程

    // 配置参数
    std::string map_save_path_;  // 地图保存路径

public:
    /**
     * @brief 构造函数，初始化ROS服务
     * @param nh ROS节点句柄
     */
    FastLIOController();
    /**
     * @brief 停止算法并清理资源
     */
    static void stopAlgorithm();

    /**
     * @brief 析构函数，确保算法停止运行
     */
    ~FastLIOController();

    /**
     * @brief 算法主循环函数
     */
    void algorithmLoop();


    
};


#endif