#include <stdio.h>
#include <atomic>
#include <queue>
#include <map>
#include <memory>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"

#include "version.h"
#include "droslog/log.h"
#include "droslog/logclient.h"
#include "common/log_filters.h"
#include "common/sysutils.h"
#include "common/mem_monitor.h"
#include "common/stereo_monitor.h"
#include "common/offset_timer.h"

#include <sensor_msgs/Image.h>
// #include <sensor_msgs/CompressedImage.h>

#include "mower_msgs/Trigger.h"

// 单位: 字节
// 总程序大小:  mem_info[0]
// 驻留集大小(RSS):  mem_info[1]
// 共享页面:  mem_info[2]
// 文本(代码):  mem_info[3]
// 数据/栈:  mem_info[4]
std::vector<size_t> getProcessMemoryUsage(pid_t pid) {
  std::vector<size_t> memory_info;
  std::string path = "/proc/" + std::to_string(pid) + "/statm";
  std::ifstream statm_file(path);
  
  if (statm_file.is_open()) {
      std::string line;
      std::getline(statm_file, line);
      std::istringstream iss(line);
      
      size_t value;
      while (iss >> value) {
          memory_info.push_back(value * sysconf(_SC_PAGESIZE)); // 转换为字节
      }
  }
  
  return memory_info;
}

namespace utils {

p_log_func dros_log_func_ptr;

} // namespace utils

using namespace utils;

// Estimator estimator;

std::mutex vio_mutex;
Estimator* VioIns(bool reset = false) {
    static std::shared_ptr<Estimator> vio = std::make_shared<Estimator>();
    std::lock_guard<std::mutex> lock(vio_mutex);
    if (reset) {
        vio.reset();
        Sleep(300);
        vio = std::make_shared<Estimator>();
    }
    return vio.get();
}

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;

std::atomic_bool active_vio;
std::atomic_bool to_shutdown;
std::atomic_bool reseting_vio;
std::atomic<long long> pre_reset_ts;

ros::Publisher pub_vio_offset_ts;   // sys_ts - vio_ts
ros::Publisher pub_vio_HB;

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    static double pre_ts = 0.0;
    double cur_ts = img_msg->header.stamp.toSec();
    if (cur_ts < pre_ts) {
        droslog(LogLevel::WARN,  "VIO::img0_callback(): img0 时间戳倒退, 丢弃该帧, dts=%.3f, cur_ts=%.3f, pre_ts=%.3f", cur_ts - pre_ts, cur_ts, pre_ts);
        return;
    }
    if (pre_ts > 0.0 && (cur_ts > pre_ts + 0.2)) {
        droslog(LogLevel::WARN,  "VIO::img0_callback(): img0 时间戳跳跃较大, dts=%.3f, cur_ts=%.3f, pre_ts=%.3f", cur_ts - pre_ts, cur_ts, pre_ts);
    }
    pre_ts = cur_ts;

    StereoMonitor::Instance()->count_update(cur_ts, 1);
    
    if (active_vio.load()) {
        static SimpleLogFilter log_filter(10000);
        if (log_filter.Output(GetNow_Steady())) {
            droslog(LogLevel::INFO, "VIO::img0_callback()0508: cur img0 timestamp: %.3f", cur_ts);
        }
        m_buf.lock();
        img0_buf.push(img_msg);
        m_buf.unlock();
    } else {
        static SimpleLogFilter log_filter2(10000);
        if (log_filter2.Output(GetNow_Steady())) {
            droslog(LogLevel::INFO, "VIO::img0_callback() vio暂停中: %.3f", cur_ts);
        }
    }
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    static double pre_ts = 0.0;
    double cur_ts = img_msg->header.stamp.toSec();
    if (cur_ts < pre_ts) {
        droslog(LogLevel::WARN, "VIO::img1_callback(): img1 时间戳倒退, 丢弃该帧, dts=%.3f, cur_ts=%.3f, pre_ts=%.3f", cur_ts - pre_ts, cur_ts, pre_ts);
        return;
    }
    if (pre_ts > 0.0 && (cur_ts > pre_ts + 0.2 || cur_ts < pre_ts)) {
        droslog(LogLevel::WARN, "VIO::img1_callback(): img1 时间戳跳跃较大, dts=%.3f, cur_ts=%.3f, pre_ts=%.3f", cur_ts - pre_ts, cur_ts, pre_ts);
    }
    pre_ts = cur_ts;

    StereoMonitor::Instance()->count_update(cur_ts, 2);
    
    if (active_vio.load()) {
        static SimpleLogFilter log_filter(10000);
        if (log_filter.Output(GetNow_Steady())) {
            droslog(LogLevel::INFO, "VIO::img1_callback()0508: cur img1 timestamp: %.3f", cur_ts);
        }
        m_buf.lock();
        img1_buf.push(img_msg);
        m_buf.unlock();
    } else {
        static SimpleLogFilter log_filter2(10000);
        if (log_filter2.Output(GetNow_Steady())) {
            droslog(LogLevel::INFO, "VIO::img1_callback() vio暂停中: %.3f", cur_ts);
        }
    }
}


cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv::Mat img;
    try {
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat image = cv_ptr->image;
      img = image.clone();
      cv::cvtColor(img, img, CV_RGB2GRAY);
    } catch (cv_bridge::Exception& e) {
      std::printf("Could not convert from '%s' to 'bgr8'.\n", img_msg->encoding.c_str());
    }

    return img;
}

// extract images with same timestamp from two topics
void sync_process()
{
    long long start_ts = GetNow_Steady();
    droslog(LogLevel::INFO, "VIO::sync_process() ++++++");
    while(ros::ok() && !to_shutdown.load())
    {
        if(STEREO)
        {
            cv::Mat image0, image1;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();
                // 0.003s sync tolerance
                if(time0 < time1 - 0.003)
                {
                    img0_buf.pop();
                    printf("throw img0\n");
                }
                else if(time0 > time1 + 0.003)
                {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                else
                {
                    time = img0_buf.front()->header.stamp.toSec();
                    header = img0_buf.front()->header;
                    image0 = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image1 = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();
                    //printf("find img0 and img1\n");
                }
            }
            m_buf.unlock();
            if(!image0.empty()) {
                if (!reseting_vio.load()) {
                    static long long pre_time = GetNow_Steady();
                    static double pre_img_time = 0.0;
                    if (GetNow_Steady() > pre_time + 50 && time > pre_img_time + 0.01) {
                        pre_time = GetNow_Steady();
                        pre_img_time = time;
                        VioIns()->inputImage(time, image0, image1);
                    }
                }
            }
        }

        {
            static SimpleLogFilter fps_filter(50);
            if (fps_filter.Output(GetNow_Steady())) {
                pubVioPoseResult();
            }
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
    droslog(LogLevel::INFO, "VIO::sync_process() ------");
}


void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    static double pre_ts = 0.0;
    static int jump_cnt = 0;
    double cur_ts = imu_msg->header.stamp.toSec();
    if (pre_ts > 0.0 && cur_ts > pre_ts + 0.038) {
        jump_cnt++;
        static SimpleLogFilter log_filter1(5000);
        if (log_filter1.Output(GetNow_Steady())) {
            droslog(LogLevel::WARN, "VIO::imu_callback(): IMU timestamp jump too large, dts=%.3f, cur_ts=%.3f, jump_cnt=%d", cur_ts - pre_ts, cur_ts, jump_cnt);
        }
    }
    pre_ts = cur_ts;

    StereoMonitor::Instance()->count_update(cur_ts, 0);
    offset_timer.FeedEmb_ts(imu_msg->angular_velocity_covariance[1], cur_ts);
    
    if (active_vio.load()) {
        static SimpleLogFilter log_filter(10000);
        if (log_filter.Output(GetNow_Steady())) {
            droslog(LogLevel::INFO, "VIO::imu_callback(): 时间戳确认 emb_ts: %.3f, sys_ts: %.3f, offset_ts: %.3f, 温度: %.1f", 
                cur_ts, imu_msg->angular_velocity_covariance[1], imu_msg->angular_velocity_covariance[2], imu_msg->angular_velocity_covariance[0]);
        }
        double t = imu_msg->header.stamp.toSec();
        double dx = imu_msg->linear_acceleration.x;
        double dy = imu_msg->linear_acceleration.y;
        double dz = imu_msg->linear_acceleration.z;
        double rx = imu_msg->angular_velocity.x;
        double ry = imu_msg->angular_velocity.y;
        double rz = imu_msg->angular_velocity.z;
        Vector3d acc(dx, dy, dz);
        Vector3d gyr(rx, ry, rz);
        if (!reseting_vio.load()) {
            VioIns()->inputIMU(t, acc, gyr);
        }
    } else {
        static SimpleLogFilter log_filter2(10000);
        if (log_filter2.Output(GetNow_Steady())) {
            droslog(LogLevel::INFO, "VIO::imu_callback() vio暂停中: %.3f", cur_ts);
        }
    }
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if(feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.toSec();
    if (!reseting_vio.load()) {
        VioIns()->inputFeature(t, featureFrame);
    }
    return;
}

// void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
// {
//     if (restart_msg->data == true)
//     {
//         ROS_WARN("restart the estimator!");
//         estimator.clearState();
//         estimator.setParameter();
//     }
//     return;
// }
bool ctrl_service(mower_msgs::Trigger::Request &req,
                    mower_msgs::Trigger::Response &rep)
{
    if (req.arg == "reset_vio")
    {
        long long dts = GetNow_Steady() - pre_reset_ts.load();
        if (dts < 5000) {
            ROS_WARN("VIO::ctrl_service(): gap time of two reset is too short, wait for %lldms", 5000-dts);
            droslog(LogLevel::WARN, "VIO::ctrl_service(): 两次重置间隔太短, 等待一会儿, dts=%lldms", 5000-dts);
            Sleep(5000 - dts);
        }
        pre_reset_ts.store(GetNow_Steady());

        ROS_WARN("VIO::ctrl_service(): restart the estimator!");
        droslog(LogLevel::WARN, "VIO::ctrl_service(): 收到重置VIO指令, 将重置VIO");
        active_vio.store(true);
        reseting_vio.store(true);
        Sleep(200);
        VioIns(true);
        VioIns()->setParameter();
        VIO_FRAME_INDEX.store(0);
        VIO_FRAME_INDEX2.store(0);
        reseting_vio.store(false);
        rep.result = true;
        rep.message = "ok";
    } else if (req.arg == "shutdown") {
        ROS_WARN("VIO::ctrl_service(): shutdown VIO");
        droslog(LogLevel::WARN, "VIO::ctrl_service(): 收到结束VIO进程指令, 将结束VIO");
        to_shutdown.store(true);
        ros::shutdown();
        rep.result = true;
        rep.message = "ok";
    } else if (req.arg == "vio_on") {
        ROS_WARN("VIO::ctrl_service(): switch on VIO");
        droslog(LogLevel::WARN, "VIO::ctrl_service(): 收到激活VIO进程指令, 将激活VIO");
        active_vio.store(true);
        reseting_vio.store(true);
        Sleep(200);
        VioIns(true);
        VioIns()->setParameter();
        VIO_FRAME_INDEX.store(0);
        VIO_FRAME_INDEX2.store(0);
        reseting_vio.store(false);
        rep.result = true;
        rep.message = "ok";
    } else if (req.arg == "vio_off") {
        ROS_WARN("VIO::ctrl_service(): switch off VIO");
        droslog(LogLevel::WARN, "VIO::ctrl_service(): 收到暂停VIO进程指令, 将暂停VIO");
        active_vio.store(false);
        rep.result = true;
        rep.message = "ok";
    } else {
        ROS_WARN("VIO::ctrl_service(): unknown command, arg=%s", req.arg.c_str());
        droslog(LogLevel::WARN, "VIO::ctrl_service(): 收到未知指令, arg=%s", req.arg.c_str());
        rep.result = false;
        rep.message = "unknown command";
    }
    return true;
}

void monitor_thread() {
    long long start_ts = GetNow_Steady();
    droslog(LogLevel::INFO, "VIO::monitor_thread() ++++++");
    std::map<int, int> pre_count = StereoMonitor::Instance()->get_all_count();
    while(ros::ok() && !to_shutdown.load())
    {
        long long cur_ts = GetNow_Steady();
        static SimpleLogFilter mem_filter(10000);
        if (mem_filter.Output(cur_ts)) {
          pid_t pid = getpid();
          auto mem_info = getProcessMemoryUsage(pid);
          double used = 0.0;
          if (mem_info.size() >= 5) {
            used = mem_info[1] / (1024.0 * 1024.0);
            if (used > vio_memory_threshold) {
              ROS_WARN("VIO::monitor_thread(): mem used=%.1f MB, too large, shutdown", used);
              ros::shutdown();
            }
          } else {
            ROS_WARN("VIO::monitor_thread(): getProcessMemoryUsage failed, pid=%d", pid);
          }
          droslog(LogLevel::INFO, "VIO::monitor_thread(): vio已运行 %.3f sec, mem used=%.1f MB", (cur_ts - start_ts) / 1000.0, used);
        }

        static SimpleLogFilter moni_filter(1000);
        if (moni_filter.Output(cur_ts)) {
            // 发布心跳
            std_msgs::Float64 HB_ts;
            HB_ts.data = (cur_ts - start_ts) * 0.001;
            pub_vio_HB.publish(HB_ts);

            // 发布as_vio/offset_ts
            std_msgs::Float64 offset_ts_msg;
            offset_ts_msg.data = offset_timer.GetEmb_dt();
            pub_vio_offset_ts.publish(offset_ts_msg);

            // 检查帧率
            auto cur_count = StereoMonitor::Instance()->get_all_count();
            int right_dc = cur_count[2] - pre_count[2];
            int imu_dc = cur_count[0] - pre_count[0];
            pre_count = cur_count;

            static long long pre_log_ts = 0;
            if (right_dc < 10 || imu_dc < 100) {
                if (cur_ts > pre_log_ts + 2000) {
                    droslog(LogLevel::WARN, "VIO::monitor_thread() 帧率异常: imu: %d, image: %d", imu_dc, right_dc);
                    pre_log_ts = cur_ts;
                }
            } else {
                if (cur_ts > pre_log_ts + 10000) {
                    droslog(LogLevel::INFO, "VIO::monitor_thread() 帧率监控: imu: %d, image: %d", imu_dc, right_dc);
                    pre_log_ts = cur_ts;
                }
            }
        }

        Sleep(100);
    }
    droslog(LogLevel::INFO, "VIO::monitor_thread() ------");
}

int main(int argc, char **argv)
{
    active_vio.store(true);
    reseting_vio.store(false);
    to_shutdown.store(false);
    // 配置log记录器
    dros_log_func_ptr = utils::LogClient_Log;
    LogClientConfig cfg;
    cfg.log_root_dir = "john_logs";
    cfg.log_sub_dir = "vio_logs";
    cfg.log_file_interval = 2 * 3600 * 1000;        // 2 hours
    cfg.log_keep_time = 7 * 24 * 3600 * 1000;       // 7 days
    cfg.log_prefix = "vio_";
    LogClient_Init(cfg);
    ROS_WARN("VIO::main() start as_vio_node");
    droslog(LogLevel::INFO, "VIO::main() version_info: %s, build time: %s", NODE_VERSION_DATE, COMPILE_TIME);

    auto start_time = GetNow_Steady();
    ROS_WARN("VIO::main() system boot time: %lld sec", start_time/1000);
    droslog(LogLevel::INFO, "VIO::main() 系统已启动时间: %lld sec", start_time/1000);

    StereoMonitor::Instance();
    offset_timer.Hello();

    ros::init(argc, argv, "as_vio");
    ros::NodeHandle n("~");
    // ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if(argc != 3)
    {
        droslog(LogLevel::ERROR, "VIO::main() 输入参数有误, usage: rosrun as_stereo_imu_tracking as_vio_node -d [config file]");
        return 1;
    }

    string config_file = argv[2];
    droslog(LogLevel::INFO, "VIO::main() config_file: %s", config_file.c_str());

    readParameters(config_file);
    VIO_FRAME_INDEX.store(0);
    VIO_FRAME_INDEX2.store(0);
    VioIns()->setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    // 取消对双目的启动控制
    // ros::ServiceClient stereo_ctrl_clt;
    // stereo_ctrl_clt = n.serviceClient<mower_msgs::Trigger>("/perception/ctrl");
    // if (stereo_ctrl_clt.waitForExistence(ros::Duration(5.0))) {
    //     mower_msgs::Trigger start_stereo;
    //     start_stereo.request.arg = "start";
    //     if (!stereo_ctrl_clt.call(start_stereo)) {
    //         droslog(LogLevel::ERROR, "VIO::main(): stereo_ctrl_clt 调用启动双目服务失败, rep=%d, err_msg=%s", start_stereo.response.result, start_stereo.response.message.c_str());
    //     } else {
    //         droslog(LogLevel::INFO, "VIO::main(): stereo_ctrl_clt 调用启动双目服务成功");
    //     }
    //     std::chrono::milliseconds dura(1000);
    //     std::this_thread::sleep_for(dura);
    // } else {
    //     ROS_ERROR("VIO::main(): stereo_ctrl_clt wait for existence timeout");
    //     droslog(LogLevel::ERROR, "VIO::main(): stereo_ctrl_clt 等待双目服务超时, 双目节点可能未启动或者异常");
    // }

    ROS_WARN("VIO::main(): waiting for image and imu...");
    droslog(LogLevel::INFO, "VIO::main(): waiting for image and imu...");

    registerPub(n);
    {
        long long start_ts = GetNow_Steady();
        while (GetNow_Steady() < start_ts + 15000) {
            float cpu_usage = GetCpuUsageRatio();
            if (cpu_usage > 0.8) {
                droslog(LogLevel::INFO, "VIO::main(): 当前cpu占用: %.3f 高, 等待cpu空闲启动", cpu_usage);
            } else {
                droslog(LogLevel::INFO, "VIO::main(): 当前cpu占用: %.3f 低, 准备启动", cpu_usage);
                break;
            }
            Sleep(1000);
        }
        Sleep(1000);
        droslog(LogLevel::INFO, "VIO::main(): 启动订阅");
    }

    droslog(LogLevel::INFO, "VIO::main(): IMAGE0_TOPIC: %s, IMAGE1_TOPIC: %s", IMAGE0_TOPIC.c_str(), IMAGE1_TOPIC.c_str());

    ros::Subscriber sub_imu  = n.subscribe(IMU_TOPIC, 5, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 2, img0_callback);
    ros::Subscriber sub_img1 = n.subscribe(IMAGE1_TOPIC, 2, img1_callback);
    
    ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2, feature_callback);
    ros::ServiceServer ctrl_server = n.advertiseService("/as_vio/ctrl", ctrl_service);

    pub_vio_offset_ts = n.advertise<std_msgs::Float64>("/as_vio/offset_ts", 1);
    pub_vio_HB = n.advertise<std_msgs::Float64>("/as_vio/heartbeat", 1);

    std::thread sync_thread{sync_process};
    std::thread moni_thread{monitor_thread};
    ros::spin();

    while (ros::ok()) {
        Sleep(1000);
    }

    std::printf("VIO::main(): will exit ......1\n");
    droslog(LogLevel::INFO, "VIO::main(): will exit ......1");
    
    if (sync_thread.joinable()) {
        sync_thread.join();
    }
    if (moni_thread.joinable()) {
        moni_thread.join();
    }
    
    VioIns(true);

    std::printf("VIO::main(): ------\n");
    droslog(LogLevel::INFO, "VIO::main(): ------");

    return 0;
}
