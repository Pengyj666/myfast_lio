#ifndef DROS_UTILS_COMMON_SYSUTILS_H
#define DROS_UTILS_COMMON_SYSUTILS_H

#include <string>
#include <vector>

namespace utils {

// *************************************************************************************** //
// **************************** system misc utilities ************************************ //
// *************************************************************************************** //

// ms: millisecond
void Sleep(int ms);

// exec the cmd and return the output of the cmd, read from stdout
// returned string's size is limited to max_size, default 4095
std::string ExecWithStdout(const std::string &cmd, const int &max_size = 4095);


// *************************************************************************************** //
// *************************** system resource status ************************************ //
// *************************************************************************************** //

// nic_name: the NetCard name, eg. 'eth0'
// return the mac address converted to string, eg. '00016C06A629'
// if getting failed, empty string returned, with 'error_info'.
std::string GetMacAddress(const std::string& nic_name, std::string& error_info);

// unit: Byte
long long GetTotalMemory();
// unit: GB
float GetTotalMemory_GB();

// 100.0%
// invalid if return < 0.f
float GetMemoryUsageRatio();

// 100.0%
// invalid if return < 0.f
float GetCpuUsageRatio();

// 100.0%, return the disk which the path on, eg: '/', '/home' for linux
// invalid if return < 0.f
float GetDiskUsageRatio(const char *path);

// unit: GB, return the disk which the path on, eg: '/', '/home' for linux
// invalid if return < 0.f
float GetDiskSize_GB(const char *path);

// vec[0...] temp of each core, vec.back() mean-temp
// unit: Celsius degree
// vec is empty means invalid
std::vector<float> GetCpuTemperature();

// vec[0...] freq of each core, vec.back() mean-freq
// unit: Ghz
// vec is empty means invalid
std::vector<float> GetCpuFrequencies();

// *************************************************************************************** //
// *********************************** process info ************************************** //
// *************************************************************************************** //

// read process name from '/proc/<pid>/comm, the name's length is limited to 15 char
std::string GetProcessNameFromPid(unsigned int pid);


// *************************************************************************************** //
// ********************************* file / dir ****************************************** //
// *************************************************************************************** //

bool IsFileExisting(const char *fp);
int  DeleteFile(const char *fp);

bool IsDirExisting(const char *dir);
bool CreateDir(const char *dir);

// filename NOT include the dir-path
void GetAllFileName(const std::string &dir, 
    std::vector<std::string> &v_filename);


// *************************************************************************************** //
// ************************ system / local timestamp ************************************* //
// *************************************************************************************** //

// get current time stamp (# of milliseconds since UNIX epoch time)
// ACCORING to OS's clock
long long GetNow_SysTime();

// get steady time stamp in milliseconds measured in CPU cycles
// so this is a steadily monotonic stop watch.
long long GetNow_Steady();

// get current time stamp in format like "2019/09/21-10:58:00.251"
std::string GetCurTimeStamp_MilSec();
// get current time stamp in format like "20190921_105801"
std::string GetCurTimeStamp_Sec();
// get current time stamp in format like "20190921_1058"
std::string GetCurTimeStamp_Minute();
struct _fineTimeStamp {
	int year; // 2019
	int mon; // 09
	int day; // 24
	int hour; // 09
	int min; // 39
	int sec; // 57
	int milSec; // 729
};
_fineTimeStamp GetCurTimeStamp(bool bLocal = true);
_fineTimeStamp ConvertTimeStamp(long long msSinceEpoch);
long long ConvertTimeStamp(_fineTimeStamp fTS);

// get current time stamp in format like "2019/09/21-10:58:00.251"
std::string GetTSText_MilSec(_fineTimeStamp fTS);
// get current time stamp in format like "20190921_105801"
std::string GetTSText_Sec(_fineTimeStamp fTS);
// get current time stamp in format like "20190921_1058"
std::string GetTSText_Minute(_fineTimeStamp fTS);

// ts_str must be in format like "2019/09/21-10:58:00.251"
// return -1 if conversion fails
long long ConvertTSText_MilSec(const std::string &ts_str);
// ts_str must be in format like "20190921_105801"
// return -1 if conversion fails
long long ConvertTSText_Sec(const std::string &ts_str);
// ts_str must be in format like "20190921_1058"
// return -1 if conversion fails
long long ConvertTSText_Minute(const std::string &ts_str);

} // namespace utils
#endif//DROS_UTILS_COMMON_SYSUTILS_H