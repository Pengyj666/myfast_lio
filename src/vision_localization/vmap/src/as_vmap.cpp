#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include "mower_msgs/Trigger.h"
#include "mower_msgs/MowerSensorInfo.h"
// #include "mower_msgs/VioPoseResult.h"

#include <atomic>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "vmap_version.h"
#include "common/log_filters.h"
#include "common/sysutils.h"
#include "common/timed_queue.h"
#include "droslog/log.h"
#include "droslog/logclient.h"
#include "geo_utils/geo_utils.h"
#include "geo_utils/tf_helper.h"
#include "vreloc_tracker.h"

#include "sensor_monitor.h"
#include "parameters.h"
#include "simple_pose_graph.h"
#include "spatial_map_manager.h"  // 2025-12-10 添加空间地图管理器
#include "segment_optimizer.h"   // 2025-12-15 添加段优化器
// 2026-01-07: submap_cache.h 不再需要，已合并到 spatial_map_manager.h
#include "vreloc_tracker.h"
#include "utility/CameraPoseVisualization.h"
#include "utility/map_drawer.h"  // 2025-12-11 添加空间分布可视化
#include <malloc.h>

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

#define SKIP_FIRST_CNT 10

std::mutex gps_xyz_q_mutex;
TimedQueue<geometry_msgs::PoseWithCovarianceStamped::ConstPtr> gps_xyz_q;

std::queue<sensor_msgs::ImageConstPtr> image_buf;
std::queue<sensor_msgs::PointCloudConstPtr> point_buf;
std::queue<nav_msgs::Odometry::ConstPtr> pose_buf;
std::queue<Eigen::Vector3d> odometry_buf;
std::mutex m_buf;
std::mutex m_process;
int frame_index  = 0;
int sequence = 1;
const double SKIP_DIS_MAPPING = 0.35;
const double SKIP_DIS_RELOC = 0.08;

std::atomic<double> offset_ts(0.0);

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE;

double loop_pos_cov = 0.01;
double loop_quat_cov = 0.01;

double reloc_filter_pos_factor = 0.5;
double reloc_filter_quat_factor = 0.5;

camodocal::CameraPtr m_camera;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;
ros::Publisher pub_match_img;
ros::Publisher pub_reloc_odom, pub_reloc_result;
ros::Publisher pub_kp2;
ros::Publisher pub_vmap_state;
std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string VINS_RESULT_PATH;
Eigen::Vector3d last_t(-100, -100, -100);

std::atomic<bool> is_saving(false);
std::atomic<bool> to_stop(false);

std::string g_vocabulary_file;
std::string g_map_root_dir = "/userdata/RobotData/map/";

std::atomic<int> vmap_mode_(0); // 0: idle, 1: mapping, 2: localization
std::atomic<int> pub_vreloc_cnt(0);
std::atomic_bool is_mapping_kf1(false);
std::atomic<bool> reloc_pose_valid(false);

// 空间索引统计（需要在 VmapReset 时清零）
std::atomic<int> spatial_insert_cnt(0);
std::atomic<int> spatial_reject_cnt(0);
Eigen::Vector3d reloc_d_pos;
Eigen::Quaterniond reloc_d_quat;

// 2026-01-07: submap_cache_mutex 不再需要，已合并到 SpatialMapManager

// ========== 后台线程管理器 - 2025-12-25 ==========
// 用于安全管理异步全局优化线程，避免 detach 导致的 use-after-free
// 改进：使用 condition_variable 替代忙等，支持任务名字，正确的超时语义
class BackgroundTaskManager {
public:
    static BackgroundTaskManager& instance() {
        static BackgroundTaskManager mgr;
        return mgr;
    }
    
    // 提交后台任务（带名字，用于调试）
    void submit(const std::string& task_name, std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            
            // 检查是否处于关闭状态
            if (shutdown_forced_) {
                droslog(LogLevel::WARN, "BackgroundTaskManager: 系统已关闭，拒绝任务 [%s]", task_name.c_str());
                return;
            }
            
            // 清理已完成的线程
            cleanupFinishedLocked();
            
            // 创建新任务
            auto wrapper = std::make_shared<TaskWrapper>();
            wrapper->name = task_name;
            wrapper->done.store(false);
            wrapper->thread = std::thread([this, wrapper, task]() {
                droslog(LogLevel::INFO, "BackgroundTaskManager: 任务开始 [%s]", wrapper->name.c_str());
                try {
                    task();
                } catch (const std::exception& e) {
                    droslog(LogLevel::ERROR, "BackgroundTaskManager: 任务异常 [%s]: %s", 
                            wrapper->name.c_str(), e.what());
                }
                
                // 标记完成并通知等待者
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    wrapper->done.store(true);
                    droslog(LogLevel::INFO, "BackgroundTaskManager: 任务完成 [%s]", wrapper->name.c_str());
                }
                cv_.notify_all();  // 通知所有等待者
            });
            
            tasks_.push_back(wrapper);
            droslog(LogLevel::INFO, "BackgroundTaskManager: 提交任务 [%s]，当前任务数=%zu", 
                    task_name.c_str(), tasks_.size());
        }
    }
    
    // 等待所有任务完成（节点关闭时调用）
    // 超时后进入 shutdown-forced 状态，不再接受新任务
    void waitAll(int timeout_sec = 60) {
        droslog(LogLevel::INFO, "BackgroundTaskManager: 等待所有后台任务完成 (超时=%ds)...", timeout_sec);
        
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
        
        std::unique_lock<std::mutex> lock(mutex_);
        
        // 使用 condition_variable 等待，直到所有任务完成或超时
        bool all_done = cv_.wait_until(lock, deadline, [this]() {
            for (const auto& wrapper : tasks_) {
                if (!wrapper->done.load()) return false;
            }
            return true;
        });
        
        if (!all_done) {
            // 超时 - 进入强制关闭状态
            shutdown_forced_ = true;
            
            // 打印未完成的任务
            std::string pending_tasks;
            int pending_count = 0;
            for (const auto& wrapper : tasks_) {
                if (!wrapper->done.load()) {
                    if (!pending_tasks.empty()) pending_tasks += ", ";
                    pending_tasks += wrapper->name;
                    pending_count++;
                }
            }
            
            droslog(LogLevel::ERROR, "BackgroundTaskManager: 等待超时！未完成任务(%d): [%s]", 
                    pending_count, pending_tasks.c_str());
            droslog(LogLevel::WARN, "BackgroundTaskManager: 进入 shutdown-forced 状态，不再接受新任务");
            
            // 仍然尝试 join 已完成的任务
            cleanupFinishedLocked();
            
            droslog(LogLevel::WARN, "BackgroundTaskManager: 放弃等待 %d 个未完成任务", pending_count);
        } else {
            // 全部完成
            cleanupFinishedLocked();
            droslog(LogLevel::INFO, "BackgroundTaskManager: 所有任务已完成");
        }
    }
    
    // 检查是否有正在运行的任务
    bool hasRunningTasks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& wrapper : tasks_) {
            if (!wrapper->done.load()) return true;
        }
        return false;
    }
    
    // 获取运行中任务数
    int getRunningCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        int count = 0;
        for (const auto& wrapper : tasks_) {
            if (!wrapper->done.load()) count++;
        }
        return count;
    }
    
    // 获取运行中任务名列表（用于调试）
    std::vector<std::string> getRunningTaskNames() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> names;
        for (const auto& wrapper : tasks_) {
            if (!wrapper->done.load()) {
                names.push_back(wrapper->name);
            }
        }
        return names;
    }
    
    // 是否处于强制关闭状态
    bool isShutdownForced() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return shutdown_forced_;
    }
    
private:
    BackgroundTaskManager() : shutdown_forced_(false) {}
    
    ~BackgroundTaskManager() {
        // 析构时确保没有运行中的任务
        if (hasRunningTasks()) {
            droslog(LogLevel::WARN, "BackgroundTaskManager: 析构时仍有任务运行，等待 5 秒...");
            waitAll(5);
        }
        
        // 断言检查（调试用）
        // assert(!hasRunningTasks() && "BackgroundTaskManager destroyed with running tasks!");
    }
    
    BackgroundTaskManager(const BackgroundTaskManager&) = delete;
    BackgroundTaskManager& operator=(const BackgroundTaskManager&) = delete;
    
    struct TaskWrapper {
        std::string name;           // 任务名（用于调试）
        std::thread thread;
        std::atomic<bool> done{false};
    };
    
    // 清理已完成的线程（必须在锁内调用）
    void cleanupFinishedLocked() {
        auto it = tasks_.begin();
        while (it != tasks_.end()) {
            if ((*it)->done.load()) {
                if ((*it)->thread.joinable()) {
                    (*it)->thread.join();
                }
                it = tasks_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    mutable std::mutex mutex_;
    std::condition_variable cv_;    // 用于等待任务完成
    std::vector<std::shared_ptr<TaskWrapper>> tasks_;
    bool shutdown_forced_;          // 超时后进入强制关闭状态
};
// 2026-01-07: SubMapCache 已合并到 SpatialMapManager，SubMapCacheIns 不再需要

ros::Publisher pub_reloc_path;
nav_msgs::Path reloc_path;
ros::Publisher pub_camera_pose_visual;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);

std::mutex vmap_mutex;
// 2025-12-04 更新：使用引用计数等待机制替代硬编码 Sleep
SimplePoseGraph* VmapIns(bool reset = false) {
  static std::shared_ptr<SimplePoseGraph> vmap = std::make_shared<SimplePoseGraph>();
  std::lock_guard<std::mutex> lock(vmap_mutex);
  if (reset) {
    // 等待外部引用释放，最多等待 500ms
    const int kMaxWaitMs = 500;
    const int kCheckIntervalMs = 10;
    int waited_ms = 0;
    
    while (vmap.use_count() > 1 && waited_ms < kMaxWaitMs) {
      vmap_mutex.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(kCheckIntervalMs));
      waited_ms += kCheckIntervalMs;
      vmap_mutex.lock();
    }
    
    if (vmap.use_count() > 1) {
      droslog(LogLevel::WARN, "VmapIns: 等待超时，仍有%ld个引用，强制重置", vmap.use_count());
    }
    
    vmap.reset();
    Sleep(50);  // 等待析构完成
    vmap = std::make_shared<SimplePoseGraph>();
  }
  return vmap.get();
}

// 空间地图管理器（动态地图加载） 2025-12-10 添加
std::mutex spatial_map_mutex;
SpatialMapManager* SpatialMapIns(bool reset = false) {
  static std::shared_ptr<SpatialMapManager> spatial_map = std::make_shared<SpatialMapManager>();
  std::lock_guard<std::mutex> lock(spatial_map_mutex);
  if (reset) {
    spatial_map.reset();
    spatial_map = std::make_shared<SpatialMapManager>();
    droslog(LogLevel::INFO, "SpatialMapIns: 已重置空间地图管理器");
  }
  return spatial_map.get();
}

// 段优化器（增量位姿优化） 2025-12-15 添加
std::mutex segment_opt_mutex;
SegmentOptimizer* SegmentOptIns(bool reset = false) {
  static std::shared_ptr<SegmentOptimizer> segment_opt = std::make_shared<SegmentOptimizer>();
  std::lock_guard<std::mutex> lock(segment_opt_mutex);
  if (reset) {
    segment_opt.reset();
    segment_opt = std::make_shared<SegmentOptimizer>();
    droslog(LogLevel::INFO, "SegmentOptIns: 已重置段优化器");
  }
  return segment_opt.get();
}

void VmapReset() {
  droslog(LogLevel::INFO, "VmapReset(): ========== 开始重置 ==========");
  
  vmap_mode_.store(0);
  is_mapping_kf1.store(false);
  reloc_pose_valid.store(false);
  pub_vreloc_cnt.store(0);
  
  // ======== 修复1: 彻底清空 reloc_path ========
  {
    std::lock_guard<std::mutex> lock(m_process);
    reloc_path.poses.clear();
    reloc_path.poses.shrink_to_fit();
    frame_index = 0;
  }
  
  // ======== 修复2: 彻底清空队列 ========
  {
    m_buf.lock();
    droslog(LogLevel::INFO, "VmapReset(): 清空缓存: pose_buf=%d, image_buf=%d, point_buf=%d", 
        (int)pose_buf.size(), (int)image_buf.size(), (int)point_buf.size());
    
    std::queue<sensor_msgs::ImageConstPtr>().swap(image_buf);
    std::queue<sensor_msgs::PointCloudConstPtr>().swap(point_buf);
    std::queue<nav_msgs::Odometry::ConstPtr>().swap(pose_buf);
    std::queue<Eigen::Vector3d>().swap(odometry_buf);
    
    m_buf.unlock();
  }
  
  // ======== 修复3: 重置 SimplePoseGraph 和 SpatialMapManager ========
  auto mem_before = getProcessMemoryUsage(getpid());
  size_t rss_before = mem_before.size() > 1 ? mem_before[1] / 1024 : 0;
  droslog(LogLevel::INFO, "VmapReset(): 重置前 RSS=%zu KB", rss_before);
  
  // 重置 SimplePoseGraph（原有的位姿图管理）
  VmapIns(true);
  droslog(LogLevel::INFO, "VmapReset(): SimplePoseGraph 已重置");
  
  // 重置 SpatialMapManager（新的空间索引管理）2025-12-10 添加
  SpatialMapIns(true);
  droslog(LogLevel::INFO, "VmapReset(): SpatialMapManager 已重置");
  
  // 重置 SegmentOptimizer（段优化器）2025-12-15 添加
  SegmentOptIns(true);
  droslog(LogLevel::INFO, "VmapReset(): SegmentOptimizer 已重置");
  
  // 2026-01-07: SubMapCache 已合并到 SpatialMapManager，无需单独重置
  // 预加载系统会在 SpatialMapManager 重置时一起关闭
  
  // 重置空间索引统计计数器
  spatial_insert_cnt.store(0);
  spatial_reject_cnt.store(0);
  droslog(LogLevel::INFO, "VmapReset(): 空间索引统计计数器已清零");
  
  // // 验证析构是否完成
  // int kf_cnt_after = get_KF_cnt();
  auto mem_after = getProcessMemoryUsage(getpid());
  size_t rss_after = mem_after.size() > 1 ? mem_after[1] / 1024 : 0;
  droslog(LogLevel::INFO, "VmapReset: 析构后 RSS=%zu KB (变化: %+ld KB)", 
           rss_after, (long)(rss_after - rss_before));
  
  // if (kf_cnt_after != 0) {
  //   droslog(LogLevel::WARN, "VmapReset: 警告！仍有 %d 个 KeyFrame 未析构，可能存在内存泄漏！", kf_cnt_after);
  // }
  
  // 强制归还内存给操作系统
  malloc_trim(0);
  auto mem_trim = getProcessMemoryUsage(getpid());
  size_t rss_trim = mem_trim.size() > 1 ? mem_trim[1] / 1024 : 0;
  droslog(LogLevel::INFO, "VmapReset: malloc_trim后 RSS=%zu KB (释放: %ld KB)", 
          rss_trim, (long)(rss_after - rss_trim));
  
  // // 判断内存状态
  // if (kf_cnt_after == 0 && (long)(rss_after - rss_trim) > 1000) {
  //   droslog(LogLevel::INFO, "VmapReset: 内存状态正常 - 析构完成，glibc缓存已归还");
  // } else if (kf_cnt_after == 0 && (long)(rss_before - rss_after) < 1000) {
  //   droslog(LogLevel::WARN, "VmapReset: 可疑 - 析构完成但RSS几乎没变化");
  // }
 
  // 词汇表只在节点启动时加载一次，这里只清空数据库
  if (VmapIns()->isVocabularyLoaded()) {
    VmapIns()->clearDatabase();
    droslog(LogLevel::INFO, "VmapReset(): 数据库已清空（词汇表保留）");
  } else {
    // 首次调用或异常情况，加载词汇表
    VmapIns()->loadVocabulary(g_vocabulary_file);
    droslog(LogLevel::INFO, "VmapReset(): 词汇表已加载");
  }

  VrelocTracker::Instance()->Reset();
  droslog(LogLevel::INFO, "VmapReset(): VrelocTracker 已重置");
  
  droslog(LogLevel::INFO, "VmapReset(): ========== 重置完成 ==========");
}

cv::Mat getImageFromMsg(const sensor_msgs::CompressedImage::ConstPtr &img_msg, bool to_gray = true)
{
  cv::Mat img;
  try {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg);
    cv::Mat image = cv_ptr->image;
    img = image.clone();
    if (to_gray)
    {
      cv::cvtColor(img, img, CV_RGB2GRAY);
    }
  } catch (cv_bridge::Exception& e) {
    std::printf("Could not convert from '%s' to 'bgr8'.\n", img_msg->format.c_str());
  }

  return img;
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg, bool to_gray = true)
{
  cv::Mat img;
  try {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat image = cv_ptr->image;
    img = image.clone();
    if (to_gray) {
      cv::cvtColor(img, img, CV_RGB2GRAY);
    }
  } catch (cv_bridge::Exception& e) {
    std::printf("Could not convert from '%s' to 'bgr8'.\n", img_msg->encoding.c_str());
  }
  return img;
}

void image_callback(const sensor_msgs::ImageConstPtr &image_msg)  
{
  static SimpleLogFilter log_filter(5000);
  if (log_filter.Output(GetNow_Steady())) {
    droslog(LogLevel::INFO, "VMAP::main() image_callback(): ts=%.3f", image_msg->header.stamp.toSec());
  }
  m_buf.lock();
  image_buf.push(image_msg);
  m_buf.unlock();
}

void point_callback(const sensor_msgs::PointCloudConstPtr &point_msg)
{
  static SimpleLogFilter log_filter(5000);
  if (log_filter.Output(GetNow_Steady())) {
    droslog(LogLevel::INFO, "VMAP::main() point_callback(): ts=%.3f", point_msg->header.stamp.toSec());
  }
  m_buf.lock();
  point_buf.push(point_msg);
  m_buf.unlock();

  sensor_msgs::PointCloud2 kp2_msg;
  sensor_msgs::convertPointCloudToPointCloud2(*point_msg, kp2_msg);
  pub_kp2.publish(kp2_msg);
}

void kf_pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)   // 回调函数，用于接收VIO的位姿信息
{
  offset_ts.store(pose_msg->pose.covariance[3]);
  static SimpleLogFilter log_filter(5000);
  if (log_filter.Output(GetNow_Steady())) {
    auto pose = pose_msg->pose.pose;
    droslog(LogLevel::INFO, "VMAP::main() kf_pose_callback(): ts=%.3f, vio_ts=%.3f, offset_ts=%.3f, pose=(%.3f,%.3f,%.3f)", 
        pose_msg->header.stamp.toSec(), pose_msg->pose.covariance[1], offset_ts.load(), pose.position.x, pose.position.y, pose.position.z);
  }
  m_buf.lock();
  pose_buf.push(pose_msg);
  m_buf.unlock();
}

void gps_xyz_callback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &gps_xyz_msg)
{
  static SimpleLogFilter log_filter(5000);
  if (log_filter.Output(GetNow_Steady())) {
    droslog(LogLevel::INFO, "VMAP::main() gps_xyz_callback(): ts=%.3f, type=%.0f, pos=(%.3f,%.3f,%.3f),sigma=%.3f,%.3f,%.3f", 
        gps_xyz_msg->header.stamp.toSec(), gps_xyz_msg->pose.covariance[0],
        gps_xyz_msg->pose.pose.position.x, gps_xyz_msg->pose.pose.position.y, gps_xyz_msg->pose.pose.position.z,
        gps_xyz_msg->pose.covariance[1], gps_xyz_msg->pose.covariance[2], gps_xyz_msg->pose.covariance[3]);
  }
  gps_xyz_q_mutex.lock();
  gps_xyz_q.emplace_back(gps_xyz_msg, gps_xyz_msg->header.stamp.toSec());
  gps_xyz_q_mutex.unlock();
}

void callback_vio(const nav_msgs::Odometry::ConstPtr msg) {
  offset_ts.store(msg->pose.covariance[3]);
  static SimpleLogFilter log_filter(5000);
  if (log_filter.Output(GetNow_Steady())) {
    droslog(LogLevel::INFO, "VMAP::main() callback_vio(): ts=%.3f, offset_ts=%.3f", msg->header.stamp.toSec(), offset_ts.load());
  }

  double msg_ts = msg->pose.covariance[1];
  Eigen::Vector3d vio_pos;
  Eigen::Quaterniond vio_quat;
  vio_pos << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
  vio_quat = Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
  
  if (vmap_mode_.load() == 2 && VrelocTracker::Instance()->IsTFValid()) {
    auto tf_pose = VrelocTracker::Instance()->GetVioTF();
    Eigen::Quaterniond reloc_quat = tf_pose.data.quat * vio_quat;
    Eigen::Vector3d reloc_pos = tf_pose.data.pos + tf_pose.data.quat * vio_pos;
    nav_msgs::Odometry reloc_odom;
    reloc_odom.header.stamp.fromSec(msg_ts);
    reloc_odom.header.frame_id = "world";
    reloc_odom.child_frame_id = "world";
    reloc_odom.pose.pose.position.x = reloc_pos.x();
    reloc_odom.pose.pose.position.y = reloc_pos.y();
    reloc_odom.pose.pose.position.z = reloc_pos.z();
    reloc_odom.pose.pose.orientation.w = reloc_quat.w();
    reloc_odom.pose.pose.orientation.x = reloc_quat.x();
    reloc_odom.pose.pose.orientation.y = reloc_quat.y();
    reloc_odom.pose.pose.orientation.z = reloc_quat.z();
    pub_reloc_odom.publish(reloc_odom);

    {
      cameraposevisual.reset();
      cameraposevisual.add_pose(reloc_pos, reloc_quat);
      cameraposevisual.publish_by(pub_camera_pose_visual, reloc_odom.header);
    }

    static Eigen::Vector3d last_reloc_pos = Eigen::Vector3d::Zero();
    if ((vio_pos-last_reloc_pos).norm() > 0.05) {
      last_reloc_pos = vio_pos;
      
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header = reloc_odom.header;
      pose_stamped.header.frame_id = "world";
      pose_stamped.pose = reloc_odom.pose.pose;
      reloc_path.header = reloc_odom.header;
      reloc_path.header.frame_id = "world";
      reloc_path.poses.push_back(pose_stamped);
  
      static SimpleLogFilter log_filter(2000);
      if (log_filter.Output(GetNow_Steady())) {
        if (reloc_path.poses.size() > 2000) {
          reloc_path.poses.erase(reloc_path.poses.begin(), reloc_path.poses.begin() + 50);
        }
        pub_reloc_path.publish(reloc_path);
      }
    }
  }
}

void callback_wheel_vel(const geometry_msgs::TwistStamped::ConstPtr msg) {
  double ts = msg->header.stamp.toSec();
  double lv_x = msg->twist.linear.x;
  double av_z = msg->twist.linear.z;

  // droslog(LogLevel::INFO, "VMAP::main() callback_wheel_vel(): ts=%.3f, vel=(%.3f,%.3f,%.3f)", ts, lv_x, lv_y, av_z);
  int state = (lv_x > 0.05 || av_z > 0.05) ? 1 : 0;
  SensorMonitor::Instance()->FeedMovingState(ts, state);
}

void callback_sensor_info(const mower_msgs::MowerSensorInfo::ConstPtr &msg) {
  double ts = msg->header.stamp.toSec();
  int state = (msg->is_docking_done) ? 1 : 0;

  SensorMonitor::Instance()->FeedCSState(ts, state);
}

void process()  
{
  droslog(LogLevel::INFO, "process() thread start");
  to_stop.store(false);
  while (ros::ok() && !to_stop.load())
  {
    sensor_msgs::ImageConstPtr image_msg = NULL;
    sensor_msgs::PointCloudConstPtr point_msg = NULL;
    nav_msgs::Odometry::ConstPtr pose_msg = NULL;

    // find out the messages with same time stamp
    m_buf.lock();
    {
      static SimpleLogFilter log_filter(5000);
      if (log_filter.Output(GetNow_Steady())) {
        droslog(LogLevel::INFO, "VMAP::main() process() img.size=%d, pts.size=%d, pose.size=%d", image_buf.size(), point_buf.size(), pose_buf.size());
      }
    }
    while (image_buf.size() > 30)  
      image_buf.pop();

    if(!image_buf.empty() && !point_buf.empty() && !pose_buf.empty())   
    {
      double pose_buf_ts0 = pose_buf.front()->pose.covariance[1];  
      if (image_buf.front()->header.stamp.toSec() > pose_buf_ts0)  
      {
        pose_buf.pop();
        // droslog(LogLevel::INFO, "VMAP::main() process(): throw pose at beginning");
      }
      else if (image_buf.front()->header.stamp.toSec() > point_buf.front()->header.stamp.toSec())  
      {
        point_buf.pop();
        // 降频日志：每 5 秒输出一次统计
        static int throw_point_cnt = 0;
        static SimpleLogFilter throw_point_filter(5000);
        throw_point_cnt++;
        if (throw_point_filter.Output(GetNow_Steady())) {
          droslog(LogLevel::INFO, "VMAP::main() process(): throw point at beginning, 最近累计 %d 次", throw_point_cnt);
          throw_point_cnt = 0;
        }
      }
      else if (image_buf.back()->header.stamp.toSec() >= pose_buf_ts0   
          && point_buf.back()->header.stamp.toSec() >= pose_buf_ts0) 
      {
        pose_msg = pose_buf.front();  
        double pose_msg_ts = pose_msg->pose.covariance[1];
        
        pose_buf.pop();         
        while (!pose_buf.empty()) 
          pose_buf.pop();
        while (image_buf.front()->header.stamp.toSec() < pose_msg_ts)
          image_buf.pop();
        image_msg = image_buf.front();
        image_buf.pop();

        while (point_buf.front()->header.stamp.toSec() < pose_msg_ts)
          point_buf.pop();
        point_msg = point_buf.front();
        point_buf.pop();
      }
    }
    m_buf.unlock();

    if (vmap_mode_.load() > 0 && pose_msg != NULL)
    {
      cv::Mat rgb_image = getImageFromMsg(image_msg, false);
      cv::Mat image;
      cv::cvtColor(rgb_image, image, cv::COLOR_RGB2GRAY);
      
      // build keyframe
      Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                            pose_msg->pose.pose.position.y,
                            pose_msg->pose.pose.position.z);
      Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                pose_msg->pose.pose.orientation.x,
                                pose_msg->pose.pose.orientation.y,
                                pose_msg->pose.pose.orientation.z).toRotationMatrix();
      
      double msg_ts = pose_msg->pose.covariance[1];
      int CS_state = SensorMonitor::Instance()->GetChargingStationState(msg_ts);
      static double pre_CS_ts = 0.0;
      static int pre_KF_state_cnt = 0;

      // 2025-12-11: 恢复距离限制（0.35m建图，0.08m重定位）+ 空间索引双重筛选
      // 距离限制：粗筛，避免创建过多关键帧对象
      // 空间索引：精筛，Cell+方向槽位控制最终存储的关键帧
      bool is_charging_station_area = (CS_state >= 1 && msg_ts > pre_CS_ts + 1.0 && pre_KF_state_cnt < 5);
      bool mapping_distance_ok = (vmap_mode_.load() == 1 && (T - last_t).norm() > SKIP_DIS_MAPPING);
      bool reloc_distance_ok = (vmap_mode_.load() == 2 && (T - last_t).norm() > SKIP_DIS_RELOC);
      if (mapping_distance_ok || reloc_distance_ok || is_charging_station_area)
      {
        if (CS_state >= 1) {
          pre_CS_ts = msg_ts;
          pre_KF_state_cnt++;
          droslog(LogLevel::INFO, "VMAP::main() 检测到在桩, 增加在桩关键帧, 本次已添加: %d", pre_KF_state_cnt);
        } else {
          if (pre_KF_state_cnt > 0) {
            droslog(LogLevel::INFO, "VMAP::main() 检测到离桩");
          }
          pre_KF_state_cnt = 0;
        }

        vector<cv::Point3f> point_3d; 
        vector<cv::Point2f> point_2d_uv; 
        vector<cv::Point2f> point_2d_normal;
        vector<double> point_id;

        for (unsigned int i = 0; i < point_msg->points.size(); i++)
        {
          cv::Point3f p_3d;
          p_3d.x = point_msg->points[i].x;
          p_3d.y = point_msg->points[i].y;
          p_3d.z = point_msg->points[i].z;
          point_3d.push_back(p_3d);

          cv::Point2f p_2d_uv, p_2d_normal;
          double p_id;
          p_2d_normal.x = point_msg->channels[i].values[0];
          p_2d_normal.y = point_msg->channels[i].values[1];
          p_2d_uv.x = point_msg->channels[i].values[2];
          p_2d_uv.y = point_msg->channels[i].values[3];
          p_id = point_msg->channels[i].values[4];
          point_2d_normal.push_back(p_2d_normal);
          point_2d_uv.push_back(p_2d_uv);
          point_id.push_back(p_id);

          //printf("u %f, v %f \n", p_2d_uv.x, p_2d_uv.y);
        }

        // 2026-01-14: 性能诊断 - 记录关键帧创建耗时
        long long t_kf_start = GetNow_Steady();
        std::shared_ptr<KeyFrame> keyframe = std::make_shared<KeyFrame>(msg_ts, frame_index, T, R, image,
                            point_3d, point_2d_uv, point_2d_normal, point_id, sequence);
        long long t_kf_create = GetNow_Steady() - t_kf_start;

        RefLocInfo rli;
        double msg_sys_ts = msg_ts + offset_ts.load();
        {
          // 查找gps_xyz
          std::lock_guard<std::mutex> lock(gps_xyz_q_mutex);
          int idx = gps_xyz_q.findAfter(msg_sys_ts);
          if (idx > 0) {
            auto pre_gps = gps_xyz_q[idx];
            auto next_gps = gps_xyz_q[idx - 1];
            double pre_gps_ts = pre_gps->header.stamp.toSec();
            double next_gps_ts = next_gps->header.stamp.toSec();

            if (next_gps_ts - pre_gps_ts < 0.3) {
              double alpha = (msg_sys_ts - pre_gps_ts) / (next_gps_ts - pre_gps_ts);
              rli.xyz[0] = pre_gps->pose.pose.position.x * (1 - alpha) + next_gps->pose.pose.position.x * alpha;
              rli.xyz[1] = pre_gps->pose.pose.position.y * (1 - alpha) + next_gps->pose.pose.position.y * alpha;
              rli.xyz[2] = pre_gps->pose.pose.position.z * (1 - alpha) + next_gps->pose.pose.position.z * alpha;
              rli.timestamp = msg_ts;
              rli.type = pre_gps->pose.covariance[0];
              double x_sig = pre_gps->pose.covariance[1];
              double y_sig = pre_gps->pose.covariance[2];
              double z_sig = pre_gps->pose.covariance[3];

              rli.cov << x_sig * x_sig, 0, 0,
                        0, y_sig * y_sig, 0,
                        0, 0, z_sig * z_sig;
            }
          } else if (idx == 0) {
            auto pre_gps = gps_xyz_q[idx];
            double pre_gps_ts = pre_gps->header.stamp.toSec();
            if (msg_sys_ts - pre_gps_ts < 0.1) {
              rli.xyz[0] = pre_gps->pose.pose.position.x;
              rli.xyz[1] = pre_gps->pose.pose.position.y;
              rli.xyz[2] = pre_gps->pose.pose.position.z;

              rli.timestamp = msg_ts;
              rli.type = pre_gps->pose.covariance[0];
              double x_sig = pre_gps->pose.covariance[1];
              double y_sig = pre_gps->pose.covariance[2];
              double z_sig = pre_gps->pose.covariance[3];

              rli.cov << x_sig * x_sig, 0, 0,
                        0, y_sig * y_sig, 0,
                        0, 0, z_sig * z_sig;
            }
          }
        }
        if (false) {
          // 检查在桩
          if (CS_state >= 1) {
            rli.type = 0; // 代表在桩
            rli.xyz << 0.098, 0.0, 0.0;
            rli.timestamp = msg_ts;
            rli.cov << 0.01, 0, 0,
                       0, 0.01, 0,
                       0, 0, 0.01;
          }
        }

        common::Data_ProbPose vio_kf;
        vio_kf.timestamp = msg_ts;
        vio_kf.ppose.pos = T;
        vio_kf.ppose.quat = R;
        VrelocTracker::Instance()->FeedData(vio_kf);
        
        // 2026-01-14: 性能诊断统计变量
        long long t_reloc = 0, t_insert = 0, t_loop = 0, t_segment = 0;
        
        int reloc_ret = -1;
        m_process.lock();
        if (vmap_mode_.load() == 1) {
          if (is_mapping_kf1.load()) {
            droslog(LogLevel::INFO, "VMAP::process() 记录首个建图关键帧");
            // 检查是否是建图第一个关键帧
            is_mapping_kf1.store(false);
            rli.type = 0; 
            rli.xyz << 0.098, 0.0, 0.0;
            rli.timestamp = msg_ts;
            rli.cov << 0.01, 0, 0,
                       0, 0.01, 0,  
                       0, 0, 0.01;
          }

          keyframe->SetRefLocInfo(rli);
          
          // 2025-12-15: 建图流程优化（增量段优化版本）
          // 1. 建图阶段：使用 VIO 位姿进行空间索引筛选（控制关键帧数量）
          //    - RTK 信息存入 ref_loc_info_，作为后续优化的约束
          //    - 不用 RTK 替换 VIO 位姿，因为 VIO 位姿虽有漂移但相对精度高
          // 2. 段优化：每收集到足够 RTK 约束后，执行局部 SPA 优化
          //    - 更新关键帧位姿，增量重建空间索引，增量存盘
          // 3. 保存时：loopCorrection() 进行全局优化（加入回环约束微调）
          //    - 对已局部优化的帧进行全局调整，修正量更小
          
          // 使用 VIO 位姿进行空间索引筛选
          bool inserted = SpatialMapIns()->insertKeyFrame(keyframe);
          if (inserted) {
            // 被空间索引接受，同时添加到 SimplePoseGraph（用于词袋匹配和回环检测）
            VmapIns()->addKeyFrame(keyframe, 1);
            spatial_insert_cnt++;
            
            // 2026-01-08: 更新最新关键帧索引（用于滑窗淘汰）
            SpatialMapIns()->updateLatestKeyFrameIndex(keyframe->index);
            
            // 2025-12-15: 添加到段优化器
            bool triggered = SegmentOptIns()->addKeyFrame(keyframe);
            if (triggered) {
              auto seg_stats = SegmentOptIns()->getStats();
              droslog(LogLevel::INFO, "VMAP::process() 段优化触发: segments=%d, total_kf=%d, moved=%d",
                  seg_stats.segment_count, seg_stats.total_keyframes, seg_stats.total_moved_indices);
            }
          } else {
            // 该 Cell 该方向槽位已被占用，拒绝此关键帧
            spatial_reject_cnt++;
          }
          
          static SimpleLogFilter mapping_log(5000);
          if (mapping_log.Output(GetNow_Steady())) {
            auto seg_stats = SegmentOptIns()->getStats();
            droslog(LogLevel::INFO, "VMAP::process() 建图中: 接受=%d, 拒绝=%d, 关键帧数=%d, 段优化=%d次", 
                spatial_insert_cnt.load(), spatial_reject_cnt.load(), VmapIns()->getKeyFrameCount(),
                seg_stats.segment_count);
          }
        } else if (vmap_mode_.load() == 2) {
          // 2026-01-09: 设置 GPS 信息，用于 findConnection() 中的 GPS 交叉验证
          keyframe->SetRefLocInfo(rli);
          
          // ========== 阶段5b：统一子图管理（预加载 + 淘汰）- 2026-01-07 ==========
          // 使用 SpatialMapManager 统一管理，无需 SubMapCache
          SpatialMapIns()->updatePosition(T);
          
          // ========== 阶段6：全局 DBoW2 重定位 ==========
          // 2026-01-11: 改回全局 DBoW2 搜索，因为 VIO 漂移经常很大
          // 空间索引依赖 VIO 位置，漂移大时会找不到候选帧
          Eigen::Vector3d reloc_t;
          Eigen::Quaterniond reloc_q;
          
          // 2026-01-13: 判断是否是首次重定位
          // 参考 VioTracker 的思路：首次重定位时 VIO 坐标系还未对齐到地图坐标系
          // 需要跳过 relative_t 验证，只依赖 GPS 验证
          bool is_first_reloc = !reloc_pose_valid.load();
          
          // 2026-01-14: 性能诊断 - 记录重定位耗时
          long long t_reloc_start = GetNow_Steady();
          // 使用全局 DBoW2 重定位
          reloc_ret = VmapIns()->relocalization(keyframe, reloc_t, reloc_q, 0, is_first_reloc);
          t_reloc = GetNow_Steady() - t_reloc_start;
          
          if (reloc_ret >= 0) {
            // 首次重定位成功，标记坐标系已对齐
            if (is_first_reloc) {
              reloc_pose_valid.store(true);
              droslog(LogLevel::INFO, "VMAP: 首次重定位成功，VIO 坐标系已对齐到地图坐标系");
            }
            common::Data_ProbPose vreloc_vio;
            vreloc_vio.timestamp = msg_ts;
            vreloc_vio.ppose.pos = reloc_t;
            vreloc_vio.ppose.quat = reloc_q;
            VrelocTracker::Instance()->FeedVreloc(vreloc_vio);
            // 注意：addRelocConstraint 移到 insertKeyFrame 成功后调用
            // 确保约束和关键帧同步添加到段优化器
          }
          
          // ========== 阶段7：边工作边建图 + 回环检测 ==========
          // 工作时同时创建新关键帧并添加到段优化器
          long long t_insert_start = GetNow_Steady();
          bool inserted = SpatialMapIns()->insertKeyFrame(keyframe);
          t_insert = GetNow_Steady() - t_insert_start;
          
          if (inserted) {
            spatial_insert_cnt++;
            
            // 2026-01-08: 更新最新关键帧索引（用于滑窗淘汰）
            SpatialMapIns()->updateLatestKeyFrameIndex(keyframe->index);
            
            // 2026-01-17: 重定位约束移到这里，确保与关键帧同步添加
            // 修复：之前 addRelocConstraint 在 insertKeyFrame 之前调用，
            // 导致约束被添加但关键帧可能被拒绝，造成约束与帧不匹配
            if (reloc_ret >= 0) {
              SegmentOptIns()->addRelocConstraint(keyframe->index, reloc_ret, reloc_t, reloc_q);
            }
            
            // 2026-01-12: 添加新帧到 DBoW2 数据库（与建图模式一致）
            
            VmapIns()->addKeyFrameIntoVoc(keyframe);
            
            // 2026-01-14: 性能诊断 - 记录回环检测耗时
            long long t_loop_start = GetNow_Steady();
            // 回环检测（与建图模式相同，使用 DBoW2）
            int loop_index = VmapIns()->detectLoop(keyframe, keyframe->index);
            if (loop_index >= 0) {
              auto loop_kf = VmapIns()->getKeyFrame(loop_index);
              if (loop_kf && keyframe->findConnection(loop_kf.get())) {
                // 回环检测成功，添加回环约束到段优化器
                Eigen::Matrix<double, 8, 1> loop_info;
                loop_info << keyframe->loop_info[0], keyframe->loop_info[1], keyframe->loop_info[2],
                             keyframe->loop_info[3], keyframe->loop_info[4], keyframe->loop_info[5],
                             keyframe->loop_info[6], keyframe->loop_info[7];
                SegmentOptIns()->addLoopConstraint(keyframe->index, loop_index, loop_info);
                
                droslog(LogLevel::INFO, "VMAP: 定位模式回环成功: %d -> %d", 
                        keyframe->index, loop_index);
              }
            }
            t_loop = GetNow_Steady() - t_loop_start;
            
            // 2026-01-14: 性能诊断 - 记录段优化耗时
            long long t_segment_start = GetNow_Steady();
            SegmentOptIns()->addKeyFrame(keyframe);
            t_segment = GetNow_Steady() - t_segment_start;
            
            // 2026-01-08: 定期执行滑窗淘汰（每50帧检查一次）
            static int sliding_evict_counter = 0;
            if (++sliding_evict_counter >= 50) {
              sliding_evict_counter = 0;
              SpatialMapIns()->evictBySlidingWindow();
            }
          } else {
            spatial_reject_cnt++;
          }
        }
        frame_index++;
        last_t = T;

        m_process.unlock();
        
        // 2026-01-14: 性能诊断 - 输出耗时统计
        // 当任一环节耗时超过100ms时输出警告，帮助定位性能瓶颈
        long long total_time = t_kf_create + t_reloc + t_insert + t_loop + t_segment;
        static long long max_total_time = 0;
        static long long max_kf_create = 0, max_reloc = 0, max_loop = 0, max_segment = 0;
        max_total_time = std::max(max_total_time, total_time);
        max_kf_create = std::max(max_kf_create, t_kf_create);
        max_reloc = std::max(max_reloc, t_reloc);
        max_loop = std::max(max_loop, t_loop);
        max_segment = std::max(max_segment, t_segment);
        
        if (total_time > 100 || t_kf_create > 50 || t_reloc > 80 || t_loop > 50 || t_segment > 30) {
          droslog(LogLevel::WARN, "VMAP::process() 性能警告: total=%lldms (kf=%lld, reloc=%lld, loop=%lld, seg=%lld), 可能影响VIO实时性",
              total_time, t_kf_create, t_reloc, t_loop, t_segment);
        }
        
        // 每10秒输出一次性能统计
        static SimpleLogFilter perf_log(10000);
        if (perf_log.Output(GetNow_Steady())) {
          droslog(LogLevel::INFO, "VMAP::process() 性能统计(最大值): total=%lldms, kf_create=%lldms, reloc=%lldms, loop=%lldms, segment=%lldms",
              max_total_time, max_kf_create, max_reloc, max_loop, max_segment);
          // 重置统计
          max_total_time = 0;
          max_kf_create = 0;
          max_reloc = 0;
          max_loop = 0;
          max_segment = 0;
        }
        
        keyframe.reset();
        
        if (reloc_ret >= 0 && vmap_mode_.load() == 2 && VrelocTracker::Instance()->IsVioValid()) {
          auto tf_pose = VrelocTracker::Instance()->GetVioTF();
          Eigen::Quaterniond reloc_quat = tf_pose.data.quat * Eigen::Quaterniond(R);
          Eigen::Vector3d reloc_pos = tf_pose.data.pos + tf_pose.data.quat * T;
          nav_msgs::Odometry reloc_result;
          reloc_result.header.stamp.fromSec(msg_ts);
          reloc_result.header.frame_id = "world";
          reloc_result.child_frame_id = "world";
          reloc_result.pose.pose.position.x = reloc_pos.x();
          reloc_result.pose.pose.position.y = reloc_pos.y();
          reloc_result.pose.pose.position.z = reloc_pos.z();
          reloc_result.pose.pose.orientation.w = reloc_quat.w();
          reloc_result.pose.pose.orientation.x = reloc_quat.x();
          reloc_result.pose.pose.orientation.y = reloc_quat.y();
          reloc_result.pose.pose.orientation.z = reloc_quat.z();
          pub_reloc_result.publish(reloc_result);
        }
      }
    }
    
    Sleep(1);
  }
  droslog(LogLevel::INFO, "VMAP::main() process() thread finished");
}

void monitor_thread() {
  long long start_ts = GetNow_Steady();
  droslog(LogLevel::INFO, "VMAP::monitor_thread() ++++++");
  while(ros::ok() && !to_stop.load())
  {
    long long cur_ts = GetNow_Steady();
    static SimpleLogFilter mem_filter(10000);
    if (mem_filter.Output(cur_ts)) {
      pid_t pid = getpid();
      auto mem_info = getProcessMemoryUsage(pid);
      double used = 0.0;
      if (mem_info.size() >= 5) {
        used = mem_info[1] / (1024.0 * 1024.0);
        if (used > 1024.0) {
          ROS_WARN("VMAP::monitor_thread(): mem used=%.1f MB, too large, shutdown", used);
          ros::shutdown();
        }
      } else {
        ROS_WARN("VMAP::monitor_thread(): getProcessMemoryUsage failed, pid=%d", pid);
      }
      droslog(LogLevel::INFO, "VMAP::monitor_thread(): vmap已运行 %.3f sec, mem used=%.1f MB, cur_mode=%d", (cur_ts - start_ts) / 1000.0, used, vmap_mode_.load());
    }

    static SimpleLogFilter state_filter(1000);
    if (state_filter.Output(cur_ts)) {
      std_msgs::String state_msg;
      state_msg.data = "idl";
      if (vmap_mode_.load() == 1) {
        state_msg.data = "map";
      } else if (vmap_mode_.load() == 2) {
        state_msg.data = "loc";
      }
      pub_vmap_state.publish(state_msg);
    }

    Sleep(100);
  }
  droslog(LogLevel::INFO, "VMAP::monitor_thread() ------");
}

bool ctrl_service(mower_msgs::Trigger::Request &req,
    mower_msgs::Trigger::Response &rep)
{
  droslog(LogLevel::WARN, "VMAP::ctrl_service(): 收到指令, arg=%s", req.arg.c_str());
  std::string ctrl_type = req.arg;

  if (ctrl_type == "LC") {
    droslog(LogLevel::WARN, "VMAP::ctrl_service(): 收到回环修正指令, 将进行回环修正");
    // LoopCorrection
    rep.result = true;
    rep.message = "ok";
  } else if (ctrl_type == "reset_vmap") {
    droslog(LogLevel::WARN, "VMAP::ctrl_service(): 收到重置指令, 将重置vmap");
    // reset_vmap
    VmapReset();
    rep.result = true;
    rep.message = "ok";
  } else if (ctrl_type == "start_mapping") {
    droslog(LogLevel::WARN, "VMAP::ctrl_service(): 收到开始视觉建图指令, 将重置vmap并开始建图");
    VmapReset();  // 会同时重置 SimplePoseGraph、SpatialMapManager 和 SegmentOptimizer
    is_mapping_kf1.store(true);
    Sleep(100);
    
    // 2025-12-15: 配置段优化器
    SegmentOptimizerConfig seg_config;
    seg_config.min_rtk_count = 3;           // 至少 3 个 RTK Fix 触发段优化
    seg_config.min_keyframe_count = 30;     // 至少 30 帧触发
    seg_config.min_distance = 5.0;          // 至少行驶 5 米
    seg_config.max_time_gap = 60.0;         // 超过 60 秒强制触发
    seg_config.auto_save_to_disk = true;    // 优化后自动存盘
    seg_config.map_dir = g_map_root_dir;    // 使用全局地图目录
    SegmentOptIns()->setConfig(seg_config);
    SegmentOptIns()->setSpatialMapManager(SpatialMapIns());
    droslog(LogLevel::INFO, "VMAP::ctrl_service(): 段优化器已配置: min_rtk=%d, min_kf=%d, min_dist=%.1fm",
        seg_config.min_rtk_count, seg_config.min_keyframe_count, seg_config.min_distance);
    
    vmap_mode_.store(1);
    
    droslog(LogLevel::INFO, "VMAP::ctrl_service(): 建图模式已启动, SimplePoseGraph/SpatialMapManager/SegmentOptimizer已重置");
    rep.result = true;
    rep.message = "ok";
  } else if (ctrl_type == "stop_mapping") {
    droslog(LogLevel::WARN, "VMAP::ctrl_service(): 收到停止视觉建图指令, 将停止输入建图数据");
    
    // 2025-12-15: 强制执行最后一段的优化
    if (SegmentOptIns()->hasPendingSegment()) {
      int pending = SegmentOptIns()->getPendingCount();
      droslog(LogLevel::INFO, "VMAP::ctrl_service(): 执行最后一段优化, pending=%d", pending);
      int optimized = SegmentOptIns()->forceOptimize();
      droslog(LogLevel::INFO, "VMAP::ctrl_service(): 最后一段优化完成, optimized=%d", optimized);
    }
    
    // 打印建图统计
    auto seg_stats = SegmentOptIns()->getStats();
    droslog(LogLevel::INFO, "VMAP::ctrl_service(): 建图统计 - SimplePoseGraph关键帧数=%d, SpatialMap(子图=%d, 关键帧=%d, 插入=%d, 拒绝=%d)",
        VmapIns()->getKeyFrameCount(),
        SpatialMapIns()->getSubMapCount(), 
        SpatialMapIns()->getTotalKeyFrameCount(),
        spatial_insert_cnt.load(), spatial_reject_cnt.load());
    droslog(LogLevel::INFO, "VMAP::ctrl_service(): 段优化统计 - segments=%d, total_kf=%d, rtk=%d, moved=%d, dist=%.1fm",
        seg_stats.segment_count, seg_stats.total_keyframes, seg_stats.total_rtk_frames,
        seg_stats.total_moved_indices, seg_stats.total_distance);
    vmap_mode_.store(0);
    rep.result = true;
    rep.message = "ok";
  } else if (ctrl_type == "start_reloc") {
    droslog(LogLevel::WARN, "VMAP::ctrl_service(): 收到开始重定位指令");
    vmap_mode_.store(2);
    rep.result = true;
    rep.message = "ok";
  } else if (ctrl_type == "flush_only") {
    // 2026-01-17: 仅刷盘，不做全局优化，用于加载新地图前快速保存增量数据
    droslog(LogLevel::INFO, "VMAP::ctrl_service(): 收到 flush_only 指令，仅刷盘");
    
    // 停止定位模式
    vmap_mode_.store(0);
    
    // 同步刷盘（阻塞直到完成）
    SegmentOptIns()->forceOptimize();
    SegmentOptIns()->flushDiskSync();
    SpatialMapIns()->flushAllDirty();
    
    droslog(LogLevel::INFO, "VMAP::ctrl_service(): flush_only 完成");
    rep.result = true;
    rep.message = "ok";
  } else if (ctrl_type == "stop_reloc") {
    droslog(LogLevel::WARN, "VMAP::ctrl_service(): 收到停止重定位指令");
    
    // 立即停止定位模式
    vmap_mode_.store(0);
    
    // ========== 工作结束后全局优化（异步执行）- 2025-12-24 ==========
    // 异步执行全局优化，避免阻塞服务返回，不影响机器人回充电桩等操作
    auto seg_stats = SegmentOptIns()->getStats();
    bool should_global_optimize = seg_stats.total_keyframes > 0 && 
                                  seg_stats.total_rtk_frames > 0;
    
    if (should_global_optimize) {
      droslog(LogLevel::INFO, "VMAP::ctrl_service(): 启动异步全局优化线程 (kf=%d, rtk=%d)",
              seg_stats.total_keyframes, seg_stats.total_rtk_frames);
      
      // 使用后台任务管理器安全地执行异步全局优化
      BackgroundTaskManager::instance().submit("WorkModeGlobalOptimization", [=]() {
        droslog(LogLevel::INFO, "VMAP::async_global_opt: 开始异步全局优化...");
        
        // 1. 强制刷盘
        SegmentOptIns()->forceOptimize();
        SegmentOptIns()->flushDiskSync();
        SpatialMapIns()->flushAllDirty();  // 2026-01-07: 使用统一接口
        
        // 2. 执行全局优化
        double opt_start = GetNow_Steady();
        int correction_ret = VmapIns()->loopCorrection();
        double opt_time = GetNow_Steady() - opt_start;
        
        if (correction_ret > 0) {
          droslog(LogLevel::INFO, "VMAP::async_global_opt: 全局优化完成，耗时 %.2f 秒", opt_time / 1000.0);
          
          // 3. 更新空间索引
          SpatialMapIns()->rebuildSpatialIndex();
          
          // 4. 保存优化后的地图
          std::string map_path = SpatialMapIns()->getMapPath();  // 2026-01-07: 使用统一接口
          if (!map_path.empty()) {
            int saved = SpatialMapIns()->saveToDirectory(map_path, 0.0, 0.0, 0.0);
            droslog(LogLevel::INFO, "VMAP::async_global_opt: 保存优化后地图: %d 个子图", saved);
          }
        } else {
          droslog(LogLevel::WARN, "VMAP::async_global_opt: 全局优化失败，约束不足");
        }
        
        droslog(LogLevel::INFO, "VMAP::async_global_opt: 异步全局优化线程结束");
      });
      
    } else {
      // 数据不足，只做刷盘
      SegmentOptIns()->forceOptimize();
      SegmentOptIns()->flushDiskSync();
      SpatialMapIns()->flushAllDirty();  // 2026-01-07: 使用统一接口
      droslog(LogLevel::INFO, "VMAP::ctrl_service(): 工作数据不足，跳过全局优化，仅刷盘");
    }
    
    rep.result = true;
    rep.message = "ok";
  } else {
    droslog(LogLevel::WARN, "VMAP::ctrl_service() 收到未知指令, arg=%s", ctrl_type.c_str());
    rep.result = false;
    rep.message = "unknown command";
  }

  return true;
}

// 异步保存地图的全局变量
static std::atomic<int> async_save_state(0);  // 0:空闲, 1:保存中, 2:完成, -1:失败
static std::string async_save_message;
static std::mutex async_save_mutex;

bool savemap_service(mower_msgs::Trigger::Request &req,
  mower_msgs::Trigger::Response &rep)
{
  ROS_WARN("VMAP::savemap_service(): received request, arg=%s", req.arg.c_str());
  droslog(LogLevel::WARN, "VMAP::savemap_service(): 收到指令, arg=%s", req.arg.c_str());
  std::string map_name = req.arg;
  if (map_name.empty()) {
    rep.result = false;
    rep.message = "map name is empty";
    return true;
  }
  
  // 检查异步保存状态（无论 is_saving 是什么值都要检查 state）
  // 修复：之前只在 is_saving=true 时检查 state，导致轮询时可能重复触发保存
  int state = async_save_state.load();
  if (state == 1) {
    // 正在保存中
    rep.result = true;
    rep.message = "saving_in_progress";
    return true;
  } else if (state == 2) {
    // 保存完成，重置状态并返回结果
    async_save_state.store(0);
    is_saving.store(false);
    rep.result = true;
    std::lock_guard<std::mutex> lock(async_save_mutex);
    rep.message = async_save_message;
    return true;
  } else if (state == -1) {
    // 保存失败，重置状态并返回错误
    async_save_state.store(0);
    is_saving.store(false);
    rep.result = false;
    std::lock_guard<std::mutex> lock(async_save_mutex);
    rep.message = async_save_message;
    return true;
  }
  
  // state == 0 时才允许启动新的保存
  if (is_saving.load()) {
    // 理论上不应该到这里，但作为防护
    rep.result = true;
    rep.message = "saving_in_progress";
    return true;
  }

  std::string map_path = g_map_root_dir + map_name;
  if (map_path.back() != '/')
    map_path += "/";
  
  is_saving.store(true);
  async_save_state.store(1);
  
  Sleep(200);
  if (!IsDirExisting(map_path.c_str())) {
    CreateDir(map_path.c_str());
  }

  droslog(LogLevel::INFO, "VMAP::savemap_service(): 将保存地图在: %s", map_path.c_str());
  
  // 2025-12-15: 打印段优化统计（保存前确认增量优化情况）
  auto seg_stats = SegmentOptIns()->getStats();
  droslog(LogLevel::INFO, "VMAP::savemap_service(): 段优化统计 - segments=%d, optimized_kf=%d, moved_idx=%d",
      seg_stats.segment_count, seg_stats.total_keyframes, seg_stats.total_moved_indices);
  
  // ========== 启动异步保存任务 - 2025-12-25 ==========
  // 全局优化和保存操作耗时较长，使用后台任务管理器避免阻塞服务调用
  // 使用 BackgroundTaskManager 而非 detach，确保节点关闭时正确 join
  BackgroundTaskManager::instance().submit("MappingSaveMap", [=]() {
    droslog(LogLevel::INFO, "VMAP::async_savemap: 开始异步保存地图...");
    double total_start = GetNow_Steady();
    
    // ========== Step 1: 执行全局位姿图优化 ==========
    // 在段优化基础上，加入回环约束进行全局调整
    // 由于段优化已经修正了大部分 VIO 漂移，全局优化的调整量会更小
    droslog(LogLevel::INFO, "VMAP::async_savemap: ========== Step 1: 全局位姿图优化 ==========");
    droslog(LogLevel::INFO, "VMAP::async_savemap: 优化前关键帧数=%d", VmapIns()->getKeyFrameCount());
    
    double opt_start = GetNow_Steady();
    int correction_ret = VmapIns()->loopCorrection();
    double opt_time = GetNow_Steady() - opt_start;
    
    if (correction_ret <= 0) {
      droslog(LogLevel::WARN, "VMAP::async_savemap: loopCorrection 返回 %d，可能RTK/回环约束不足", correction_ret);
    } else {
      droslog(LogLevel::INFO, "VMAP::async_savemap: loopCorrection 成功，耗时 %.2f 秒", opt_time / 1000.0);
    }
  
    // ========== Step 2: 增量式重建空间索引 ==========
    // 2025-12-15: 改用增量索引更新，只移动位姿变化导致 Cell/Slot 改变的帧
    droslog(LogLevel::INFO, "VMAP::async_savemap: ========== Step 2: 增量重建空间索引 ==========");
    droslog(LogLevel::INFO, "VMAP::async_savemap: 优化前空间索引: 子图=%d, 关键帧=%d",
        SpatialMapIns()->getSubMapCount(), SpatialMapIns()->getTotalKeyFrameCount());
    
    // 获取所有优化后的关键帧
    auto all_keyframes = VmapIns()->getAllKeyFrames();
    
    if (all_keyframes.empty()) {
      droslog(LogLevel::WARN, "VMAP::async_savemap: 没有关键帧，跳过保存");
      {
        std::lock_guard<std::mutex> lock(async_save_mutex);
        async_save_message = "no keyframes to save";
      }
      async_save_state.store(-1);
      is_saving.store(false);
      return;
    }
    
    // 标记所有帧为脏，检查是否需要重新索引
    SpatialMapIns()->markDirtyBatch(all_keyframes);
    
    // 执行增量索引重建（只移动真正需要移动的帧）
    int moved_count = SpatialMapIns()->rebuildDirtyIndices();
    
    // 统计结果
    int insert_success = SpatialMapIns()->getTotalKeyFrameCount();
    int insert_reject = static_cast<int>(all_keyframes.size()) - insert_success;
    
    droslog(LogLevel::INFO, "VMAP::async_savemap: 增量索引重建完成: 移动=%d, 总帧=%d, 拒绝=%d", 
        moved_count, insert_success, insert_reject);
    droslog(LogLevel::INFO, "VMAP::async_savemap: SpatialMapManager 统计: 子图数=%d, 总关键帧=%d",
        SpatialMapIns()->getSubMapCount(), SpatialMapIns()->getTotalKeyFrameCount());
    SpatialMapIns()->printStatistics();
    
    // ========== Step 3: 生成空间分布可视化图 ==========
    droslog(LogLevel::INFO, "VMAP::async_savemap: ========== Step 3: 空间分布可视化 ==========");
    {
    // 收集关键帧可视化信息
    std::vector<MapDrawer::KeyFrameVis> kf_vis_list;
    int rtk_count = 0, loop_count = 0;
    
    for (auto& kf : all_keyframes) {
      MapDrawer::KeyFrameVis kfv;
      kfv.x = kf->T_w_i.x();
      kfv.y = kf->T_w_i.y();
      kfv.yaw = SpatialMapManager::getYawFromRotation(kf->R_w_i);
      // // 使用原始 VIO 位姿绘制，因为优化后的 T_w_i 可能因坐标系对齐问题而失真
      // // 特别是在没有 RTK 约束的情况下
      // kfv.x = kf->vio_T_w_i.x();
      // kfv.y = kf->vio_T_w_i.y();
      // kfv.yaw = SpatialMapManager::getYawFromRotation(kf->vio_R_w_i);
      kfv.direction_slot = kf->direction_slot;
      kfv.cell_x = kf->cell_x;
      kfv.cell_y = kf->cell_y;
      kfv.submap_x = kf->submap_x;
      kfv.submap_y = kf->submap_y;
      kfv.has_rtk = (kf->ref_loc_info_.type == 1);
      kfv.has_loop = kf->has_loop;
      
      if (kfv.has_rtk) rtk_count++;
      if (kfv.has_loop) loop_count++;
      
      kf_vis_list.push_back(kfv);
    }
    
    // 计算地图范围 - 同时考虑优化后位姿和原始VIO位姿，确保画布足够大  2025-12-17 添加
    float min_x = 0, max_x = 0, min_y = 0, max_y = 0;
    for (const auto& kf : all_keyframes) {
      // 优化后位姿
      min_x = std::min(min_x, (float)kf->T_w_i.x());
      max_x = std::max(max_x, (float)kf->T_w_i.x());
      min_y = std::min(min_y, (float)kf->T_w_i.y());
      max_y = std::max(max_y, (float)kf->T_w_i.y());
      // 原始VIO位姿
      min_x = std::min(min_x, (float)kf->vio_T_w_i.x());
      max_x = std::max(max_x, (float)kf->vio_T_w_i.x());
      min_y = std::min(min_y, (float)kf->vio_T_w_i.y());
      max_y = std::max(max_y, (float)kf->vio_T_w_i.y());
    }
    
    droslog(LogLevel::INFO, "VMAP::async_savemap: 画布范围: x=[%.2f, %.2f], y=[%.2f, %.2f]",
        min_x, max_x, min_y, max_y);
    
    // 添加边距
    float margin = 5.0f;
    min_x -= margin; max_x += margin;
    min_y -= margin; max_y += margin;
    
    // 配置画布 - 让轨迹居中显示  2025-12-19 修正
    MapDrawer::CanvasParams canvas_params;
    canvas_params.resolution = 0.05f;  // 5cm/pixel
    
    // 计算轨迹范围
    float range_x = max_x - min_x;
    float range_y = max_y - min_y;
    
    // 画布尺寸 = 轨迹范围 + 边距
    int margin_pixels = 100;  // 边距像素
    canvas_params.width = int(range_y / canvas_params.resolution) + 2 * margin_pixels;
    canvas_params.height = int(range_x / canvas_params.resolution) + 2 * margin_pixels;
    canvas_params.width = std::max(800, std::min(8000, canvas_params.width));
    canvas_params.height = std::max(800, std::min(8000, canvas_params.height));
    
    // 原点位置：让轨迹居中
    // 变换公式: u = org_xy[0] - y/res, v = org_xy[1] - x/res
    // 要让 y=min_y 映射到 u=margin_pixels, y=max_y 映射到 u=width-margin_pixels
    // 要让 x=min_x 映射到 v=height-margin_pixels, x=max_x 映射到 v=margin_pixels
    // 所以: org_xy[0] = margin_pixels + max_y/res (让 y=max_y 时 u=margin_pixels)
    //       org_xy[1] = margin_pixels + max_x/res (让 x=max_x 时 v=margin_pixels)
    canvas_params.org_xy[0] = margin_pixels + max_y / canvas_params.resolution;
    canvas_params.org_xy[1] = margin_pixels + max_x / canvas_params.resolution;
    
    droslog(LogLevel::INFO, "VMAP::async_savemap: 画布配置: %dx%d, res=%.2f, org=(%.1f, %.1f), 轨迹范围: x=[%.1f,%.1f], y=[%.1f,%.1f]",
        canvas_params.width, canvas_params.height, canvas_params.resolution,
        canvas_params.org_xy[0], canvas_params.org_xy[1], min_x, max_x, min_y, max_y);
    
    MapDrawer drawer;
    drawer.InitCanvas(canvas_params);
    
    // 绘制网格和边界
    drawer.DrawSubMapGrid(5.0f);   // 5m SubMap 边界
    drawer.DrawCellGrid(0.25f);    // 0.25m Cell 网格
    drawer.DrawOrgP();             // 原点
    
    // 绘制关键帧
    drawer.DrawKeyFrames(kf_vis_list);
    
    // 绘制统计信息
    drawer.DrawStatistics(
        insert_success,
        SpatialMapIns()->getTotalKeyFrameCount(),
        SpatialMapIns()->getSubMapCount(),
        rtk_count, loop_count);
    
    // 保存图像
    std::string vis_path = map_path + "spatial_distribution.png";
    cv::imwrite(vis_path, drawer.GetMap());
    droslog(LogLevel::INFO, "VMAP::async_savemap: 空间分布图已保存: %s", vis_path.c_str());
    
    // 生成热力图版本
    MapDrawer drawer2;
    drawer2.InitCanvas(canvas_params);
    drawer2.DrawCellHeatmap(kf_vis_list, 0.25f);
    drawer2.DrawSubMapGrid(5.0f);
    drawer2.DrawOrgP();
    drawer2.DrawStatistics(
        insert_success,
        SpatialMapIns()->getTotalKeyFrameCount(),
        SpatialMapIns()->getSubMapCount(),
        rtk_count, loop_count);
    
    std::string heatmap_path = map_path + "cell_heatmap.png";
    cv::imwrite(heatmap_path, drawer2.GetMap());
    droslog(LogLevel::INFO, "VMAP::async_savemap: Cell热力图已保存: %s", heatmap_path.c_str());
    }
    
    // ========== Step 4: 保存地图 ==========
    droslog(LogLevel::INFO, "VMAP::async_savemap: ========== Step 4: 保存地图 ==========");
    
    // 4.1 保存视觉地图（位姿图 + 特征点/描述子）
    VmapIns()->saveMap(map_path);
    droslog(LogLevel::INFO, "VMAP::async_savemap: 视觉地图保存完成");
    
    // 4.2 保存空间索引到最终目录 spatial/submaps/（区别于增量存盘的 submaps_temp/）
    // 这里保存的是全局优化后的最终数据
    int spatial_saved = SpatialMapIns()->saveToDirectory(map_path, 0.0, 0.0, 0.0);
    droslog(LogLevel::INFO, "VMAP::async_savemap: 空间索引保存完成: submaps=%d (保存到 spatial/submaps/)", spatial_saved);
    
    droslog(LogLevel::INFO, "VMAP::async_savemap: 地图保存完成: %s", map_path.c_str());
    
    // ========== Step 5: 内存清理 ==========
    // 释放未被空间索引的关键帧，减少内存占用
    // 这一步对于长时间建图尤其重要，可防止内存持续增长
    droslog(LogLevel::INFO, "VMAP::async_savemap: ========== Step 5: 内存清理 ==========");
    
    // 获取空间索引中的有效关键帧
    auto indexed_keyframes = SpatialMapIns()->getAllIndexedKeyFrames();
    droslog(LogLevel::INFO, "VMAP::async_savemap: 空间索引中的有效帧数: %zu", indexed_keyframes.size());
    
    // 从 SimplePoseGraph 中移除未被索引的帧，并重建词袋
    int removed_count_inner = VmapIns()->cleanupUnindexedKeyFrames(indexed_keyframes);
    
    droslog(LogLevel::INFO, "VMAP::async_savemap: 内存清理完成: 移除冗余帧=%d, 当前总帧数=%d", 
        removed_count_inner, VmapIns()->getKeyFrameCount());
    
    // 计算总耗时
    double total_time = GetNow_Steady() - total_start;
    droslog(LogLevel::INFO, "VMAP::async_savemap: 异步保存完成，总耗时 %.2f 秒", total_time / 1000.0);
    
    // 设置完成状态
    {
      std::lock_guard<std::mutex> lock(async_save_mutex);
      async_save_message = map_name;
    }
    async_save_state.store(2);
    is_saving.store(false);
    vmap_mode_.store(0);
    
  });  // 使用 BackgroundTaskManager，节点关闭时会自动 join
  
  // 异步保存已启动，立即返回
  rep.result = true;
  rep.message = "save_started";  // 表示保存已开始，需要轮询状态
  return true;
}

bool loadmap_service(mower_msgs::Trigger::Request &req,
  mower_msgs::Trigger::Response &rep)
{
  ROS_WARN("VMAP::loadmap_service(): received request, arg=%s", req.arg.c_str());
  droslog(LogLevel::WARN, "VMAP::loadmap_service(): 收到指令, arg=%s", req.arg.c_str());
  std::string map_name = req.arg;
  if (map_name.empty()) {
    rep.result = false;
    rep.message = "map name is empty";
    return true;
  }

  std::string map_path = g_map_root_dir + map_name;
  if (map_path.back() != '/')
    map_path += "/";
  
  Sleep(200);
  if (!IsDirExisting(map_path.c_str())) {
    droslog(LogLevel::ERROR, "VMAP::loadmap_service(): map dir not exist: %s", map_path.c_str());
    rep.result = false;
    rep.message = "map dir not exist";
    return true;
  }

  droslog(LogLevel::INFO, "VMAP::loadmap_service(): 将加载地图: %s", map_path.c_str());
  
  VmapReset();  // 会清空数据库但保留词汇表

  vmap_mode_.store(2);
  
  // ========== 统一子图管理 - 2026-01-07 ==========
  // 词汇表已在 VmapReset 中处理（保留或加载），这里不需要重复加载
  
  // 2. 初始化 SpatialMapManager 预加载系统（原 SubMapCache）
  SpatialMapIns()->setWorkMode(true);                       // 启用 SubMap 淘汰
  SpatialMapIns()->initializeCache(map_path, 25);           // 最大缓存 25 个子图
  
  // 3. 加载空间索引元数据
  int loaded = SpatialMapIns()->loadFromDirectory(map_path);
  droslog(LogLevel::INFO, "VMAP::loadmap_service(): 空间索引加载完成: %d submaps", loaded);
  
  // 4. 初始化段优化器（工作模式）
  SegmentOptimizerConfig seg_config;
  seg_config.auto_save_to_disk = true;
  seg_config.map_dir = map_path;
  seg_config.work_mode = true;           // 工作模式
  seg_config.use_temp_dir = false;       // 不用 temp 目录
  seg_config.disk_batch_threshold = 10;  // 工作模式参数
  seg_config.disk_max_pending_sec = 60;
  seg_config.vio_only_max_time = 120.0;  // 纯 VIO 2分钟超时
  SegmentOptIns()->setConfig(seg_config);
  SegmentOptIns()->setSpatialMapManager(SpatialMapIns());
  SegmentOptIns()->setWorkMode(true);
  
  // 5. 设置空间索引管理器到 SimplePoseGraph（用于空间索引重定位）
  VmapIns()->setSpatialMapManager(SpatialMapIns());
  droslog(LogLevel::INFO, "VMAP::loadmap_service(): 空间索引重定位已启用");
  
  // 6. 设置起点位置（用于滑窗淘汰时保留起点附近的关键帧）
  // 起点通常是充电桩位置，割草机需要回到起点
  // 这里使用地图中第一个关键帧的位置作为起点
  auto all_kfs = SpatialMapIns()->getAllIndexedKeyFrames();
  if (!all_kfs.empty()) {
    // 2026-01-11: 过滤掉空指针，避免比较时崩溃
    std::vector<std::shared_ptr<KeyFrame>> valid_kfs;
    valid_kfs.reserve(all_kfs.size());
    for (const auto& kf : all_kfs) {
      if (kf) valid_kfs.push_back(kf);
    }
    
    if (!valid_kfs.empty()) {
      // 找到索引最小的关键帧作为起点
      auto min_it = std::min_element(valid_kfs.begin(), valid_kfs.end(),
          [](const std::shared_ptr<KeyFrame>& a, const std::shared_ptr<KeyFrame>& b) {
              return a->index < b->index;
          });
      if (min_it != valid_kfs.end()) {
          SpatialMapIns()->setOriginPosition((*min_it)->T_w_i);
          droslog(LogLevel::INFO, "VMAP::loadmap_service(): 设置起点位置 (%.2f, %.2f, %.2f)",
                  (*min_it)->T_w_i.x(), (*min_it)->T_w_i.y(), (*min_it)->T_w_i.z());
      }
      
      // 更新最新关键帧索引
      auto max_it = std::max_element(valid_kfs.begin(), valid_kfs.end(),
          [](const std::shared_ptr<KeyFrame>& a, const std::shared_ptr<KeyFrame>& b) {
              return a->index < b->index;
          });
      if (max_it != valid_kfs.end()) {
          SpatialMapIns()->updateLatestKeyFrameIndex((*max_it)->index);
          droslog(LogLevel::INFO, "VMAP::loadmap_service(): 最新关键帧索引 %d", (*max_it)->index);
      }
    }
  }
  
  // 7. 加载关键帧到词袋数据库（DBoW2）
  // 注意：loadFromDirectory() 已经加载了所有子图的完整数据
  // initialLoadByPosition() 设计用于按需加载场景，但当前是全量加载模式
  // 所以不需要调用 initialLoadByPosition()，它只会增加缓存命中计数
  VmapIns()->loadMap(map_path);
  
  rep.result = true;
  rep.message = map_name;
  droslog(LogLevel::INFO, "VMAP::loadmap_service(): 地图加载完成（工作模式）");
  return true;
}

int main(int argc, char** argv)   
{
  // 配置log记录器 
  dros_log_func_ptr = utils::LogClient_Log;  
  LogClientConfig cfg;
  cfg.log_root_dir = "john_logs";
  cfg.log_sub_dir = "vmap_logs";
  cfg.log_file_interval = 2 * 3600 * 1000;        // 2 hours
  cfg.log_keep_time = 7 * 24 * 3600 * 1000;       // 7 days
  cfg.log_prefix = "test_";
  LogClient_Init(cfg);
  ROS_WARN("VMAP::main() start vmap_test");
  droslog(LogLevel::INFO, "VMAP::main() version_info: %s, build time: %s", NODE_VERSION_DATE, COMPILE_TIME);

  if(argc < 3)
  {
    ROS_ERROR("VMAP::main() usage: rosrun vmap vmap_test -d [config file]");
    droslog(LogLevel::ERROR, "VMAP::main() 输入参数有误, usage: rosrun vmap vmap_test -d [config file], argc: %d, argv[1]: %s, argv[2]: %s", argc, argv[1], argv[2]);
    return 1;
  }

  ros::init(argc, argv, "vmap_test");
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

  double TF_vio2gps_x = 0.353, TF_vio2gps_y = -0.041, TF_vio2gps_z = -0.069;  
  double TF_imu2gps_x = 0.0, TF_imu2gps_y = 0.0, TF_imu2gps_z = -0.017;
  double TF_gps2base_x= 0.098, TF_gps2base_y = 0.0, TF_gps2base_z = 0.326;
  TFHelper::Instance()->SetParams_Vio2Gps(TF_vio2gps_x, TF_vio2gps_y, TF_vio2gps_z, 0.0, 0.0, 0.0);
  TFHelper::Instance()->SetParams_Imu2Gps(TF_imu2gps_x, TF_imu2gps_y, TF_imu2gps_z, 0.0, 0.0, 0.0);
  TFHelper::Instance()->SetParams_Gps2Base(TF_gps2base_x, TF_gps2base_y, TF_gps2base_z, 0.0, 0.0, 0.0);

  std::string config_file = argv[2];
  droslog(LogLevel::INFO, "VMAP::main() config_file: %s", argv[2]);

  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if(!fsSettings.isOpened())
  {
    droslog(LogLevel::ERROR, "VMAP::main() ERROR: Wrong path to settings, open failed!");
    return -1;
  }

  std::string IMAGE_TOPIC;

  ROW = fsSettings["image_height"];
  COL = fsSettings["image_width"];
  std::string pkg_path = ros::package::getPath("vmap");
  g_vocabulary_file = pkg_path + "/../support_files/brief_k10L6.bin";
  droslog(LogLevel::INFO, "VMAP::main() vocabulary_file: %s", g_vocabulary_file.c_str());

  VmapIns(true)->loadVocabulary(g_vocabulary_file);

  BRIEF_PATTERN_FILE = pkg_path + "/../support_files/brief_pattern.yml";
  droslog(LogLevel::INFO, "VMAP::main() BRIEF_PATTERN_FILE: %s", BRIEF_PATTERN_FILE.c_str());

  int pn = config_file.find_last_of('/');
  std::string configPath = config_file.substr(0, pn);
  std::string cam0Path;
  fsSettings["cam0_calib"] >> cam0Path;
  droslog(LogLevel::INFO, "VMAP::main() cam calib path: %s", cam0Path.c_str());
  m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam0Path.c_str());

  fsSettings["loop_pos_cov"] >> loop_pos_cov;
  fsSettings["loop_quat_cov"] >> loop_quat_cov;

  fsSettings["reloc_filter_pos_factor"] >> reloc_filter_pos_factor;
  fsSettings["reloc_filter_quat_factor"] >> reloc_filter_quat_factor;
  droslog(LogLevel::INFO, "VMAP::main() loop_pos_cov: %.4f, loop_quat_cov: %.4f", loop_pos_cov, loop_quat_cov);

  VrelocTracker::Config vreloc_cfg;
  fsSettings["reloc_align_vio_factor"] >> vreloc_cfg.vio_factor;
  fsSettings["reloc_align_vio_align_factor"] >> vreloc_cfg.vio_align_factor;
  fsSettings["reloc_align_vreloc_factor"] >> vreloc_cfg.vio_vreloc_factor;
  
  // 读取新增的优化参数
  fsSettings["reloc_tf_max_pos_jump"] >> vreloc_cfg.tf_max_pos_jump;
  fsSettings["reloc_tf_max_yaw_jump"] >> vreloc_cfg.tf_max_yaw_jump;
  fsSettings["reloc_tf_filter_enable"] >> vreloc_cfg.tf_filter_enable;
  fsSettings["reloc_tf_filter_alpha"] >> vreloc_cfg.tf_filter_alpha;
  
  VrelocTracker::Instance()->SetParams(vreloc_cfg);
  droslog(LogLevel::INFO, "VMAP::main() reloc_align_vio_factor: %.4f, reloc_align_vio_align_factor: %.4f, reloc_align_vreloc_factor: %.4f", 
      vreloc_cfg.vio_align_factor, vreloc_cfg.vio_factor, vreloc_cfg.vio_vreloc_factor);
  droslog(LogLevel::INFO, "VMAP::main() TF变化检查: max_pos_jump=%.2f, max_yaw_jump=%.2f", 
      vreloc_cfg.tf_max_pos_jump, vreloc_cfg.tf_max_yaw_jump);
  droslog(LogLevel::INFO, "VMAP::main() 渐进式校正: enable=%d, alpha=%.2f", 
      vreloc_cfg.tf_filter_enable, vreloc_cfg.tf_filter_alpha);

  fsSettings["image0_topic"] >> IMAGE_TOPIC;        
  fsSettings["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;
  fsSettings["output_path"] >> VINS_RESULT_PATH;
  DEBUG_IMAGE = 0;  // 关闭图像调试功能，节省 CPU 和内存

  tic << 0.0, 0.0, 0.0;
  qic << 1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0;

  VINS_RESULT_PATH = VINS_RESULT_PATH + "/vio_loop.csv";
  std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
  fout.close();

  fsSettings.release();

  gps_xyz_q.reset(256);

  ros::Subscriber sub_image = n.subscribe(IMAGE_TOPIC, 2, image_callback);
  ros::Subscriber sub_pose = n.subscribe("/as_vio/vio_pose_result", 2, kf_pose_callback);
  ros::Subscriber sub_point = n.subscribe("/as_vio/keyframe_point", 2, point_callback);
  ros::Subscriber sub_gps_xyz = n.subscribe("/gps_local_xyz", 2, gps_xyz_callback);
  ros::Subscriber sub_sensor_info = n.subscribe("/mower_sensor_info", 2, callback_wheel_vel);
  ros::Subscriber sub_wheel_vel = n.subscribe("/wheel_vel", 2, callback_wheel_vel);
  ros::Subscriber sub_vio = n.subscribe("/as_vio/vio_pose_result", 2, callback_vio);

  pub_kp2 = n.advertise<sensor_msgs::PointCloud2>("/as_vio/keyframe_point2", 2);

  pub_match_img = n.advertise<sensor_msgs::Image>("/as_vmap/match_image", 2);
  pub_reloc_odom = n.advertise<nav_msgs::Odometry>("/as_vmap/reloc_pose", 2);
  pub_reloc_result = n.advertise<nav_msgs::Odometry>("/as_vmap/reloc_result", 2);
  pub_reloc_path = n.advertise<nav_msgs::Path>("/as_vmap/reloc_path", 2);
  pub_vmap_state = n.advertise<std_msgs::String>("/as_vmap/vmap_state", 2);

  pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("/as_vmap/camera_pose_visual", 2);

  ros::ServiceServer savemap_server = n.advertiseService("/as_vmap/savemap", savemap_service);
  ros::ServiceServer loadmap_server = n.advertiseService("/as_vmap/loadmap", loadmap_service);

  ros::ServiceServer ctrl_server = n.advertiseService("/as_vmap/ctrl", ctrl_service);

  cameraposevisual.setScale(0.5);
  cameraposevisual.setLineWidth(0.05);

  std::thread process_thread{process};
  std::thread moni_thread{monitor_thread};

  ros::spin();

  while (ros::ok()) {
    Sleep(1000);
  }

  std::printf("VMAP::main(): will exit ......1\n");
  droslog(LogLevel::INFO, "VMAP::main(): will exit ......1");
  
  // 等待后台任务完成（全局优化、保存地图等）
  // 最多等待 60 秒，避免无限阻塞
  if (BackgroundTaskManager::instance().hasRunningTasks()) {
    droslog(LogLevel::INFO, "VMAP::main(): 等待后台任务完成 (运行中=%d)...", 
            BackgroundTaskManager::instance().getRunningCount());
    BackgroundTaskManager::instance().waitAll(60);
  }
  
  if (process_thread.joinable()) {
      process_thread.join();
  }
  if (moni_thread.joinable()) {
      moni_thread.join();
  }
  
  std::printf("VMAP::main(): ------\n");
  droslog(LogLevel::INFO, "VMAP::main(): 所有线程已结束，节点退出");

  return 0;
}