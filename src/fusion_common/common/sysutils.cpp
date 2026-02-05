#include "common/sysutils.h"

#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <thread>

#if defined(_WIN32) || defined(_WIN64)
#include <io.h>
#include <direct.h>
#include <windows.h>
#elif defined(__linux__)
#include <dirent.h>
#include <unistd.h>
#include <net/if.h>  
#include <sys/ioctl.h>  
#include <sys/socket.h> 
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/sysinfo.h>
#else
#error "DO NOT support this system, only support Window or Linux"
#endif

namespace utils {

// *************************************************************************************** //
// **************************** system misc utilities ************************************ //
// *************************************************************************************** //

void Sleep(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

std::string ExecWithStdout(const std::string &cmd, const int &max_size) {
  std::string result;
#if defined(_WIN32) || defined(_WIN64)
  // no implemented
#elif defined(__linux__)
  if (cmd.empty())
    return result;
  std::unique_ptr<FILE, void(*)(FILE*)> pipe(popen(cmd.c_str(), "r"),
      [](FILE * f) -> void 
      {
        // wrapper to ignore the return value from pclose() is needed with newer versions of gnu g++
        std::ignore = pclose(f);
      });
  if (!pipe) {
    return result;
  }

  std::array<char, 128> buffer;
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
    result += buffer.data();
    if ((int)result.size() > max_size)
      break;
  }
#endif
  return result;
}

// *************************************************************************************** //
// ************************ system resource status *************************************** //
// *************************************************************************************** //

std::string GetMacAddress(const std::string& nic_name, std::string& error_info) {
  std::string result;
#if defined(_WIN32) || defined(_WIN64)
  error_info = "NO implemented";
#elif defined(__linux__)
  struct ifreq ifr;  
  int sd;  
    
  bzero(&ifr, sizeof(struct ifreq));  
  if( (sd = socket(AF_INET, SOCK_STREAM, 0)) < 0)  
  {  
    error_info = "get mac address socket creat error";
    return result;
  }
    
  strncpy(ifr.ifr_name, nic_name.c_str(), sizeof(ifr.ifr_name) - 1);  

  if(ioctl(sd, SIOCGIFHWADDR, &ifr) < 0)  
  {  
    error_info = "get mac address error";
    close(sd);  
    return result;  
  }  

  result.resize(13);
  snprintf(&result[0], 13, "%02x%02x%02x%02x%02x%02x",
      (unsigned char)ifr.ifr_hwaddr.sa_data[0],   
      (unsigned char)ifr.ifr_hwaddr.sa_data[1],  
      (unsigned char)ifr.ifr_hwaddr.sa_data[2],   
      (unsigned char)ifr.ifr_hwaddr.sa_data[3],  
      (unsigned char)ifr.ifr_hwaddr.sa_data[4],  
      (unsigned char)ifr.ifr_hwaddr.sa_data[5]);
  close(sd);
#endif
  return result;
}

long long GetTotalMemory() {
#if defined(_WIN32) || defined(_WIN64)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullTotalPhys;
#elif defined(__linux__)
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
#endif
}

float GetTotalMemory_GB() {
  return float(GetTotalMemory() / (1024 * 1024 * 1024.0));
}

float GetMemoryUsageRatio() {
#if defined(_WIN32) || defined(_WIN64)
  // TODO
#elif defined(__linux__)
  // ref1: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=34e431b0ae398fc54ea69ff85ec700722c9da773
  // ref2: https://stackoverflow.com/questions/349889/how-do-you-determine-the-amount-of-linux-system-ram-in-c
  std::string token;
  std::ifstream file("/proc/meminfo");
  double total = 0.0, available = 0.0;
  int ck = 0;
  while(file >> token) {
    if(token == "MemAvailable:") {
      if (file >> available) {
        ck++;
      } else {
        printf("GetMemoryUsageRatio() read MemAvailable error!\n");
        return -1.f;
      }
    } else if (token == "MemTotal:") {
      if (file >> total && total > 1e-3) {
        ck++;
      } else {
        printf("GetMemoryUsageRatio() read MemTotal error!\n");
        return -1.f;
      }
    }
    if (2 == ck) {
      return 1.0 - available / total;
    }
  }
  printf("GetMemoryUsageRatio() read meminfo failed!\n");
#endif
  return -1.f;
}

namespace {

typedef struct cpu_occupy_ {
  char name[20];
  unsigned int user;          // Time spent in user mode.
  unsigned int nice;          // Time spent in user mode with low priority (nice).
  unsigned int system;        // Time spent in system mode.
  unsigned int idle;          // Time spent in the idle task. This value should be USER_HZ times the second entry in the /proc/uptime pseudo-file.
  unsigned int iowait;        // Time waiting for I/O to complete. This value is not reliable, for the following reasons:
                              //  1.The CPU will not wait for I/O to complete; iowait is the time that a task is waiting for I/O to complete. When a CPU goes into idle state for outstanding task I/O, another task will be scheduled on this CPU.
                              //  2.On a multi-core CPU, the task waiting for I/O to complete is not running on any CPU, so the iowait of each CPU is difficult to calculate.
                              //  3.The value in this field may decrease in certain conditions.
  unsigned int irq;           // Time servicing interrupts.(since Linux 2.6.0-test4)
  unsigned int soft_irq;      // Time servicing softirqs.(since Linux 2.6.0-test4)
  unsigned int steal;         // Stolen time, which is the time spent in other operating systems when running in a virtualized environment(since Linux 2.6.11)
  unsigned int guest;         // Time spent running a virtual CPU for guest operating systems under the control of the Linux kernel.(since Linux 2.6.24)
  unsigned int guest_nice;    // Time spent running a niced guest (virtual CPU for guest operating systems under the control of the Linux kernel).(since Linux 2.6.33)
} cpu_occupy_t;

// return < 0: invalid
float cal_cpuoccupy(cpu_occupy_t *o, cpu_occupy_t *n) {
  double od = o->user + o->nice + o->system + o->idle + o->iowait + o->irq + o->soft_irq;
  double nd = n->user + n->nice + n->system + n->idle + n->iowait + n->irq + n->soft_irq;
  if (nd - od > 1e-3)
    return 1.0 - (n->idle - o->idle) / (nd - od);
  else 
    return -1.f;
}

bool get_cpuoccupy(cpu_occupy_t *cpust) {
#if defined(_WIN32) || defined(_WIN64)
  // TODO
  return false;
#elif defined(__linux__)
  FILE *fd;

  fd = fopen("/proc/stat", "r");
  if (fd == NULL) {
    printf("fopen /proc/stat failed!\n");
    return false;
  }
  char buff[256];
  if (fgets(buff, sizeof(buff), fd) &&
      sscanf(buff, "%s %u %u %u %u %u %u %u", cpust->name, &cpust->user, 
          &cpust->nice, &cpust->system, &cpust->idle, 
          &cpust->iowait, &cpust->irq, &cpust->soft_irq)) {
    // printf("cpu: %d %d %d %d %d %d %d\n", cpust->user, cpust->nice, 
    //     cpust->system, cpust->idle, cpust->iowait, cpust->irq, cpust->soft_irq);
    fclose(fd);
    return true;
  }
  fclose(fd);
#endif
  return true;
}

} // namespace 

float GetCpuUsageRatio() {
#if defined(_WIN32) || defined(_WIN64)
  // TODO
  return -1.f;
#elif defined(__linux__)
  static long long last_stat_ts = 0;
  static double last_ratio = -1.0;
  static cpu_occupy_t last_stat;

  long long now_ts = GetNow_Steady();
  if (now_ts > last_stat_ts + 1000) {
    cpu_occupy_t stat1, stat2;
    if (!get_cpuoccupy(&stat1))
      return -1.f;
    Sleep(200);
    if (!get_cpuoccupy(&stat2))
      return -1.f;
    
    last_ratio = cal_cpuoccupy(&stat1, &stat2);
    last_stat = stat2;
    last_stat_ts = GetNow_Steady();
  } else if (now_ts > last_stat_ts + 200) {
    cpu_occupy_t cur_stat;
    if (!get_cpuoccupy(&cur_stat))
      return -1.f;
    
    last_ratio = cal_cpuoccupy(&last_stat, &cur_stat);
    last_stat = cur_stat;
    last_stat_ts = now_ts;
  }

  return last_ratio;
#endif
}

float GetDiskUsageRatio(const char *path) {
  if (!path)
    return -1.f;
#if defined(_WIN32) || defined(_WIN64)
  // TODO
#elif defined(__linux__)
  struct statvfs svfs;
  if (0 == statvfs(path, &svfs)) {
    const double total = svfs.f_blocks * svfs.f_frsize;
    const double available = svfs.f_bavail * svfs.f_frsize;
    const double used = total - available;
    return used / total;
  }
#endif
  return -1.f;
}

float GetDiskSize_GB(const char *path) {
  if (!path)
    return -1.f;
#if defined(_WIN32) || defined(_WIN64)
  // TODO
#elif defined(__linux__)
  const double GB = 1024.0 * 1024.0 * 1024.0;
  struct statvfs svfs;
  if (0 == statvfs(path, &svfs)) {
    return svfs.f_blocks * svfs.f_frsize / GB;
  }
#endif
  return -1.f;
}

std::vector<float> GetCpuTemperature() {
  std::vector<float> temperatures;
  const std::string thermalPath = "/sys/class/thermal/";
    
  DIR* dir = opendir(thermalPath.c_str());
  if (!dir) {
    std::printf("GetCpuTemperature() can not open dir: /sys/class/thermal !\n");
    return temperatures;
  }
  
  float mean_temp = 0.0;
  dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name(entry->d_name);

    if (name.find("thermal_zone") == 0) {
      std::string tempPath = thermalPath + name + "/temp";
      
      std::ifstream tempFile(tempPath);
      int temp;
      tempFile >> temp;
      tempFile.close();
      
      // 转换为摄氏度（内核通常以毫摄氏度报告）
      temperatures.push_back(temp / 1000.0f);

      mean_temp += (temp / 1000.f);
    }
  }
  closedir(dir);

  if (temperatures.size() > 0) {
    mean_temp = mean_temp / temperatures.size();
    temperatures.push_back(mean_temp);
  }
  
  return temperatures;
}

std::vector<float> GetCpuFrequencies() {
  std::vector<float> frequencies;
  const std::string cpuPath = "/sys/devices/system/cpu/";
  
  DIR* dir = opendir(cpuPath.c_str());
  if (!dir) {
    std::printf("GetCpuFrequencies() can not open dir: /sys/devices/system/cpu/ \n");
    return frequencies;
  }
  
  float mean_freq = 0.f;
  dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name(entry->d_name);

    // 查找cpuX目录
    if (name.find("cpu") == 0 && name.size() > 3 && isdigit(name[3])) {
      std::string freqPath = cpuPath + name + "/cpufreq/scaling_cur_freq";
      
      std::ifstream freqFile(freqPath);
      if (freqFile) {
        int freqKHz;
        freqFile >> freqKHz;
        freqFile.close();
        
        // 转换为GHz
        frequencies.push_back(freqKHz / 1000.0f / 1000.0f);
        mean_freq += (freqKHz / 1000.0f / 1000.0f);
      }
    }
  }

  if (frequencies.size() > 0) {
    mean_freq = mean_freq / frequencies.size();
    frequencies.push_back(mean_freq);
  }
  
  closedir(dir);
  return frequencies;
}

// *************************************************************************************** //
// *********************************** process info ************************************** //
// *************************************************************************************** //

std::string GetProcessNameFromPid(unsigned int pid) {
  std::string pname;
#if defined(_WIN32) || defined(_WIN64)
  // TODO
#elif defined(__linux__)
  std::string fp = "/proc/"+std::to_string(pid)+"/comm";
  std::ifstream fin(fp);
  if (fin.is_open()) {
    fin >> pname;
  }
#endif
  return pname;
}

// *************************************************************************************** //
// ********************************* file / dir ****************************************** //
// *************************************************************************************** //

bool IsFileExisting(const char *fp) {
  struct stat info;
  if (stat(fp, &info) != 0 || (info.st_mode & S_IFDIR)) {
    return false;
  }
  return true;
}

int  DeleteFile(const char *fp) {
  return remove(fp);
}

bool IsDirExisting(const char *dir) {
  struct stat info;
  if (stat(dir, &info) != 0 || !(info.st_mode & S_IFDIR)) {
    return false;
  }
  return true;
}

bool CreateDir(const char *dir) {
#if defined(_WIN32) || defined(_WIN64)
  return (0 == _mkdir(dir) || EEXIST == errno);
#elif defined(__linux__)
  return (0 == mkdir(dir, S_IRWXU | S_IRWXG | S_IRWXO) || EEXIST == errno);
#endif
}

void GetAllFileName(const std::string &dir, 
    std::vector<std::string> &v_filename) {
  v_filename.resize(0);
#if defined(_WIN32) || defined(_WIN64)
  // TODO
#elif defined(__linux__)
  DIR *p_DIR = opendir(dir.c_str());
  if (p_DIR == nullptr) {
    printf("open dir error: dir = %s\n", dir.c_str());
    return;
  }
  dirent *p_dirent;
  while ((p_dirent=readdir(p_DIR)) != NULL) {
    if (strcmp(p_dirent->d_name, ".")==0 || strcmp(p_dirent->d_name, "..")==0) {
      continue;
    }
    if (p_dirent->d_type == 8) {
      v_filename.push_back(std::string(p_dirent->d_name));
    }
  }
  closedir(p_DIR);
  std::sort(v_filename.begin(), v_filename.end(), std::less<std::string>());
#endif
}

// *************************************************************************************** //
// ************************ system / local timestamp ************************************* //
// *************************************************************************************** //

long long GetNow_SysTime()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}

long long GetNow_Steady()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string GetTSText_MilSec(_fineTimeStamp fTS)
{
  char tstrbuf[64] = { '\0' };
  snprintf(tstrbuf, 64, "%04d/%02d/%02d-%02d:%02d:%02d.%03d", fTS.year, fTS.mon,
        fTS.day, fTS.hour, fTS.min, fTS.sec, fTS.milSec);
  return tstrbuf;
}

std::string GetCurTimeStamp_MilSec()
{
  return GetTSText_MilSec(GetCurTimeStamp());
}

std::string GetTSText_Sec(_fineTimeStamp fTS)
{
  char tstrbuf[32] = { '\0' };
  snprintf(tstrbuf, 32, "%04d%02d%02d_%02d%02d%02d",
    fTS.year, fTS.mon, fTS.day, fTS.hour, fTS.min, fTS.sec);
  return tstrbuf;
}

std::string GetCurTimeStamp_Sec()
{
  return GetTSText_Sec(GetCurTimeStamp());
}

std::string GetTSText_Minute(_fineTimeStamp fTS)
{
  char tstrbuf[32] = { '\0' };
  snprintf(tstrbuf, 32, "%04d%02d%02d_%02d%02d",
      fTS.year, fTS.mon, fTS.day, fTS.hour, fTS.min);
  return tstrbuf;
}

std::string GetCurTimeStamp_Minute()
{
  return GetTSText_Minute(GetCurTimeStamp());
}

_fineTimeStamp GetCurTimeStamp(bool bLocal)
{
  auto _utic = GetNow_SysTime();
  int ms = (int)(_utic % 1000);
  auto _usecs = _utic / 1000;
  tm bt;
  if (bLocal) {
#if defined(_WIN32) || defined(_WIN64)
    _localtime64_s(&bt, (const __time64_t*)&_usecs);
#elif defined(__linux__)
    localtime_r((const time_t*)&_usecs, &bt);
#endif
  } else {
#if defined(_WIN32) || defined(_WIN64)
    _gmtime64_s(&bt, (const __time64_t*)&_usecs);
#elif defined(__linux__)
    gmtime_r((const time_t*)&_usecs, &bt);
#endif
  }
  return _fineTimeStamp({ bt.tm_year + 1900, bt.tm_mon + 1, bt.tm_mday, bt.tm_hour, bt.tm_min, bt.tm_sec, ms });
}

_fineTimeStamp ConvertTimeStamp(long long msSinceEpoch)
{
  int ms = (int)(msSinceEpoch % 1000);
  auto _usecs = (time_t)msSinceEpoch / 1000;
  tm bt;
#if defined(_WIN32) || defined(_WIN64)
  _localtime64_s(&bt, (const __time64_t*)&_usecs);
#elif defined(__linux__)
  localtime_r((const time_t*)&_usecs, &bt);
#endif
  return _fineTimeStamp({ bt.tm_year + 1900, bt.tm_mon + 1, bt.tm_mday, bt.tm_hour, bt.tm_min, bt.tm_sec, ms });
}

long long ConvertTimeStamp(_fineTimeStamp fTS)
{
  tm ttmm;
  ttmm.tm_year = fTS.year - 1900;
  ttmm.tm_mon = fTS.mon - 1;
  ttmm.tm_mday = fTS.day;
  ttmm.tm_hour = fTS.hour;
  ttmm.tm_min = fTS.min;
  ttmm.tm_sec = fTS.sec;
  ttmm.tm_isdst = 0;
  return (mktime(&ttmm) * 1000 + fTS.milSec);
}

const std::string k_ts_ms = "2019/09/21-10:58:00.251";
const std::string k_ts_sec = "20190921_105801";
const std::string k_ts_min = "20190921_1058";
long long ConvertTSText_MilSec(const std::string &ts_str) {
  if (ts_str.size() != k_ts_ms.size())
    return -1;
  _fineTimeStamp fts;
  int ret = sscanf(ts_str.c_str(), "%4d/%2d/%2d-%2d:%2d:%2d.%3d", 
      &fts.year, &fts.mon, &fts.day, &fts.hour, &fts.min, &fts.sec, &fts.milSec);
  if (ret != 7)
    return -1;
  long long ts = ConvertTimeStamp(fts);
  if (-1 == ts) {
    std::printf("Error: ConvertTSText_MilSec() Failed\n");
    return -1;
  }
  return ts;
}

long long ConvertTSText_Sec(const std::string &ts_str) {
  if (ts_str.size() != k_ts_sec.size())
    return -1;
  _fineTimeStamp fts;
  int ret = sscanf(ts_str.c_str(), "%4d%2d%2d_%2d%2d%2d", 
      &fts.year, &fts.mon, &fts.day, &fts.hour, &fts.min, &fts.sec);
  if (ret != 6)
    return -1;
  fts.milSec = 0;
  long long ts = ConvertTimeStamp(fts);
  if (-1 == ts) {
    std::printf("Error: ConvertTSText_Sec() Failed\n");
    return -1;
  }
  return ts;
}

long long ConvertTSText_Minute(const std::string &ts_str) {
  if (ts_str.size() != k_ts_min.size())
    return -1;
  _fineTimeStamp fts;
  int ret = sscanf(ts_str.c_str(), "%4d%2d%2d_%2d%2d", 
      &fts.year, &fts.mon, &fts.day, &fts.hour, &fts.min);
  if (ret != 5)
    return -1;
  fts.sec = 0;
  fts.milSec = 0;
  long long ts = ConvertTimeStamp(fts);
  if (-1 == ts) {
    std::printf("Error: ConvertTSText_Minute() Failed\n");
    return -1;
  }
  return ts;
}

} // namespace utils