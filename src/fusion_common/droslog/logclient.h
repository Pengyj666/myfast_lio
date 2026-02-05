#ifndef DROS_UTILS_DROSLOG_LOG_CLIENT_H
#define DROS_UTILS_DROSLOG_LOG_CLIENT_H

#include <cstdio>
#include <map>
#include <memory>
#include <string>

#include "droslog/logdef.h"

namespace utils {

// 用户write写入log, 内部实现:
// write() 仅做初步判断和解码, 然后放入缓存队列
// 内部的work线程处理缓存队列内的待写log:
//   1. STDOUT有效: 打印到stdout
//   2. STDERR有效: 打印到stdcerr
//   3. LOG_FILE有效: 写到文件
//   4. DROS_LOGGER有效: pub到dros_logger
class LogClient {
 public:
  enum Target : int {
    DISABLED = 1,
    STDOUT = 2,
    STDERR = 4,
    LOG_FILE = 8,
    DROS_LOGGER = 16,
  };

  enum Level : int {
    ALL = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5,
  };

  // a process only has one
  static std::shared_ptr<LogClient> Instance();
  virtual ~LogClient() {}
  virtual void Init() = 0;
  virtual void Quit() = 0;

  virtual bool write(Level level, const char *pMessage) = 0;

  virtual void setTarget(Target target) = 0;
  virtual void setLevel(Level level) = 0;

  // eg. LOGS
  virtual bool setFileDir(const std::string &fileDir) = 0;
  // eg. xxx_Logs
  virtual bool setFileSubDir(const std::string &fileSubDir) = 0;
  // eg. xxx_
  virtual bool setFilePrefix(const std::string &filePrefix) = 0;
  // interval_ms <= 0 to disable
  virtual bool setFileInterval(long long interval_ms) = 0;
  // keep_time_ms <= 0 to keep forever
  virtual bool setFileKeepTime(long long keep_time_ms) = 0;

  virtual const Level& getLevel() = 0;
  virtual Target getTarget() = 0;

  static std::string levelTostring(Level level);

  virtual void excludeTimestamp() = 0;
  virtual void includeTimestamp() = 0;
  virtual void excludeLogLevel() = 0;
  virtual void includeLogLevel() = 0;

  virtual bool SetDrosLogPubFunc(DrosLogPublishFunc log_pub_func, const std::string &log_name = "") = 0;
};

inline LogClient::Target operator&(LogClient::Target a, LogClient::Target b) {
	return static_cast<LogClient::Target>(static_cast<int>(a) & static_cast<int>(b));
}
inline LogClient::Target operator|(LogClient::Target a, LogClient::Target b) {
	return static_cast<LogClient::Target>(static_cast<int>(a) | static_cast<int>(b));
}

void LogClient_Log(int level, const char *format, ...);

struct LogClientConfig {
  bool display_timestamp = true;
  bool display_level = true;

  LogClient::Target target = LogClient::STDOUT | LogClient::LOG_FILE | LogClient::DROS_LOGGER;
  LogClient::Level level = LogClient::INFO;
  std::string log_root_dir = "LOGS";
  std::string log_sub_dir;
  std::string log_prefix;

  long long log_file_interval = 12 * 3600 * 1000;   // ms, 12 hours
  long long log_keep_time = 14 * 24 * 3600 * 1000;  // ms, 14 days
};
void LogClient_Init(const LogClientConfig &config);
void LogClient_Quit();

// void InitLogClient(const char *filename);
} // namespace utils
#endif//DROS_UTILS_DROSLOG_LOG_CLIENT_H