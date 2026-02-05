#include "common/sysutils.h"
#include "droslog/log.h"
#include "droslog/logclient.h"

#include <thread>

#include <ros/ros.h>

#include "test/ctrl_line_test.h"

namespace utils {

p_log_func dros_log_func_ptr;

} // namespace utils

using namespace utils;

int main(int argc, char *argv[])
{
  ROS_INFO("Test::main() ++++++");
  // 配置log记录器
  dros_log_func_ptr = utils::LogClient_Log;
  LogClientConfig cfg;
  cfg.log_root_dir = "john_logs";
  cfg.log_sub_dir = "test_logs";
  cfg.log_file_interval = 3600 * 1000;      // 1 hours
  cfg.log_keep_time = 7 * 24 * 3600 * 1000; // 7 days
  cfg.log_prefix = "test_";
  LogClient_Init(cfg);

  auto start_time = GetNow_Steady();
  droslog(LogLevel::INFO, "Test::main() 系统已启动时间: %lld sec", start_time/1000);

  ros::init(argc, argv, "test_tool_node");

  ros::NodeHandle nh;
  ros::NodeHandle nh_param("~");

  ctrl_line_node test_node(nh, nh_param);

  test_node.init();
  test_node.loop();

  droslog(LogLevel::INFO, "Test::main() finish test node");
  ROS_INFO("Test::main() ------");
  return 0;
}