#ifndef LIO_CODE_H
#define LIO_CODE_H 

#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "IMU_Processing.h"
#include "lio_helper.h"

class LioCode {
private:
    static LioCode* instance_;  

    bool flg_exit = false;
    condition_variable sig_buffer;
    
    std::unique_ptr<std::thread> algorithm_thread_;            // 算法主线程
    std::atomic<int> is_running_={0};  // 算法运行状态  0-停止 1-运行 2-重置算法 3-重置中 

    std::shared_ptr<LioHelper> lio_helper;
public:
    LioCode(std::shared_ptr<LioHelper>& lio_helper_);
    ~LioCode();

    void SigHandle(int sig);
    static void StaticSigHandle(int sig);
    void stopAlgorithm();
    void algorithmLoop();
    void start();

    void sig_notify(){sig_buffer.notify_all();};
    int get_is_running(){return is_running_.load();};
    
    void set_is_running(int val){is_running_.store(val);};

};


#endif