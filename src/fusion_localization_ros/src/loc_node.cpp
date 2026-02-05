#include "version.h"
#include "loc_node.h"

#include "common/sysutils.h"
#include "droslog/log.h"
#include "droslog/logclient.h"

#include <thread>

#include "common/vio_gnss_initor.h"
#include "common/vio_tracker.h"

namespace utils {

p_log_func dros_log_func_ptr;

} // namespace utils

using namespace utils;

int main(int argc, char *argv[])
{
  ROS_INFO("LOC::main() ++++++");
  // 配置log记录器
  dros_log_func_ptr = utils::LogClient_Log;
  LogClientConfig cfg;
  cfg.log_root_dir = "john_logs";
  cfg.log_sub_dir = "localization_logs";
  cfg.log_file_interval = 3600 * 1000;      // 1 hours
  cfg.log_keep_time = 7 * 24 * 3600 * 1000; // 7 days
  cfg.log_prefix = "localization_";
  LogClient_Init(cfg);
  droslog(LogLevel::INFO, "LOC::main() version_info: %s, build time: %s", NODE_VERSION_DATE, COMPILE_TIME);

  auto start_time = GetNow_Steady();
  droslog(LogLevel::INFO, "LOC::main() 系统已启动时间: %lld sec", start_time/1000);

  // 初始化所有单例
  {
    VioTracker::Instance()->Hello();
    VioGnssInitor::Instance()->Hello();
  }

  ros::init(argc, argv, NODE_NAME);

  ros::NodeHandle nh;
  ros::NodeHandle nh_param("~");
  loc_node my_node(nh,nh_param);

  std::thread monitor_thread(&loc_node::MonitorThread, &my_node);

  my_node.init();
  my_node.loop();

  if (monitor_thread.joinable())
    monitor_thread.join();

  droslog(LogLevel::INFO, "LOC::main() finish loc node");
  ROS_INFO("LOC::main() ------");
  return 0;
}