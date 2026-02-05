#include "version.h"
#include "loc_node.h"

#include <fstream>

#include "common/vmap_monitor.h"
#include "common/log_filters.h"
#include "common/sysutils.h"
#include "common/vio_reseter.h"
#include "common/vio_tracker.h"
#include "common/vio_gnss_initor.h"
#include "droslog/log.h"

using namespace utils;

namespace {

bool getProcPid(const std::string& proc_name, pid_t& pid) {
  std::string command = "pidof " + proc_name;
  std::string pid_str = utils::ExecWithStdout(command);
  if (pid_str.size() > 0) {
    if (std::all_of(pid_str.begin(), pid_str.end(), [](unsigned char c) { return std::isdigit(c) || c == '\n'; } )) {
      pid = static_cast<pid_t>(std::atoi(pid_str.c_str()));
      return true;
    } else {
      droslog(LogLevel::WARN, "getProcPid(): call cmd '%s' ret bad: %s!", command.c_str(), pid_str.c_str());
    }
  } else {
    droslog(LogLevel::WARN, "getProcPid(): call cmd '%s' Failed!", command.c_str());
  }
  return false;
}

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
  
} // namespace 

void loc_node::loop()
{  
  droslog(LogLevel::INFO, "LOC::loop() ++++++");
  ros::Rate loop_rate(100);
  while (ros::ok()) 
  {
    PubDebugInfo();
    PubLocalizationInfo();

    ros::spinOnce();
    loop_rate.sleep();
  }
  droslog(LogLevel::INFO, "LOC::loop() ------");
}

void loc_node::MonitorThread() {
  droslog(LogLevel::INFO, "LOC::MonitorThread() ++++++");
  while (ros::ok()) {

    static SimpleLogFilter fps_filter(3000);
    if (fps_filter.Output(GetNow_Steady())) {
      pid_t pid;
      float vio_used = 0.f;
      // 监控as_vio_node 进程
      if (getProcPid("as_vio_node", pid)) {
        auto mem_info = getProcessMemoryUsage(pid);
        if (mem_info.size() >= 5) {
          vio_used = mem_info[1] / (1024.0 * 1024.0);
        }
      }

      // 监控系统cpu和mem
      float cpu_usage = GetCpuUsageRatio();
      float mem_usage = GetMemoryUsageRatio();

      droslog(LogLevel::INFO, "LOC::MonitorThread() 资源监控: 系统cpu,mem=%.3f, %.1f, as_vio_node=%.1fMB", cpu_usage, mem_usage, vio_used);
    }

    // 这里监控vio是否发散
    {
      static SimpleLogFilter fps_filter(1000);
      bool is_vio_failed = VioReseter::Instance()->IsVioNeedReset();
      if (is_vio_failed) {
        droslog(LogLevel::WARN, "LOC::MonitorThread() vio发散, 需要重置vio");
        mower_msgs::Trigger reset_vio;
        reset_vio.request.arg = "reset_vio";
        if (vio_reset_clt_.waitForExistence(ros::Duration(2.0))) {
          if (!vio_reset_clt_.call(reset_vio)) {
            droslog(LogLevel::ERROR, "LOC::MonitorThread() vio_reset_clt_ 调用重置VIO服务失败, rep=%d, err_msg=%s", reset_vio.response.result, reset_vio.response.message.c_str());
          } else {
            droslog(LogLevel::INFO, "LOC::MonitorThread() vio_reset_clt_ 调用重置VIO服务成功");
          }
          VioReseter::Instance()->UpdateResetTime();
          // Sleep(200);
          // ProcVioReset();
        } else {
          droslog(LogLevel::ERROR, "LOC::MonitorThread() vio_reset_clt_ 等待重置VIO服务超时, VIO节点可能未启动或者异常");
        }
      }
      // 如果连续3次重置vio都是立马发散, 则重启vio
    }

    {
      static SimpleLogFilter fps_filter(1000);
      if (fps_filter.Output(GetNow_Steady())) {
        const int loc_mode = locator_.GetWorkMode();
        if (use_vmap_.load() && loc_mode == 1) {
          bool need_load = VmapMonitor::Instance()->NeedLoadMap();
          if (need_load) {
            VmapMonitor::Instance()->SetLoadMapTs(GetNow_Steady());
            mower_msgs::Trigger ctl_msg;
            ctl_msg.request.arg = map_name_;
            if (load_vmap_clt_.waitForExistence(ros::Duration(2.0))) {
              if (!load_vmap_clt_.call(ctl_msg)) {
                droslog(LogLevel::ERROR, "LOC::MonitorVioThread() load_vmap_clt_ 调用视觉加载地图服务失败, rep=%d, err_msg=%s", ctl_msg.response.result, ctl_msg.response.message.c_str());
              } else {
                droslog(LogLevel::INFO, "LOC::MonitorVioThread() load_vmap_clt_ 调用视觉加载地图服务成功");
              }
            } else {
              droslog(LogLevel::ERROR, "LOC::MonitorVioThread() load_vmap_clt_ 等待视觉加载地图服务超时, VMAP节点可能未启动或者异常");
            }
          }
        }
      }
    }

    Sleep(2000);
  }
  droslog(LogLevel::INFO, "LOC::MonitorThread() ------");
}

void loc_node::ProcVioReset() {
  VioTracker::Instance()->Reset();
  VioGnssInitor::Instance()->Reset();
  pre_vio_fid_.store(-1.0);
  pre_reset_ts_.store(GetNow_Steady());
}

void loc_node::ProcLioReset() {
  lidar_tracker_.Reset();
  pre_reset_ts_.store(GetNow_Steady());
}