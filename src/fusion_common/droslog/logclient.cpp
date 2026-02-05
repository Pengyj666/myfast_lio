#include "droslog/logclient.h"

#if defined(_WIN32) || defined(_WIN64)
#define VSNPRINTF _vsnprintf_s
#else 
#define VSNPRINTF vsnprintf
#endif

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <queue>

#include "common/sysutils.h"

namespace utils {
namespace {

const std::map<LogClient::Level, std::string> k_LL2Str{
    {LogClient::Level::ALL,   "ALL   "},
    {LogClient::Level::DEBUG, "DEBUG "},
    {LogClient::Level::INFO,  "INFO  "},
    {LogClient::Level::WARN,  "WARN  "},
    {LogClient::Level::ERROR, "ERROR "},
    {LogClient::Level::FATAL, "FATAL "}
};


class LCImpl : public LogClient {
public:
  struct ALog {
    long long ts;  
    Level level;
    std::string msg;
  };

  LCImpl();
  ~LCImpl();

  void Init() override;
  void Quit() override;

  bool write(Level level, const char *pMessage) override;

  void setTarget(Target target) override {
    std::lock_guard<std::mutex> lock(config_mutex_);
    target_ = target;
  }
  void setLevel(Level level) override {
    std::lock_guard<std::mutex> lock(config_mutex_);
    level_ = level;
  }

  bool setFileDir(const std::string &fileDir) override {
    std::lock_guard<std::mutex> lock(config_mutex_);
    log_root_dir_ = fileDir;
    if (log_root_dir_.empty())
      log_root_dir_ = "./";
    if (log_root_dir_.back() != '/')
      log_root_dir_ += "/";
    if (!IsDirExisting(log_root_dir_.c_str())) {
      return CreateDir(log_root_dir_.c_str());
    }
    return true;
  }
  bool setFileSubDir(const std::string &fileSubDir) override {
    std::lock_guard<std::mutex> lock(config_mutex_);
    log_sub_dir_ = fileSubDir;
    if (log_sub_dir_.empty())
      log_sub_dir_ = "./";
    if (log_sub_dir_.back() != '/')
      log_sub_dir_ += "/";
    if (!IsDirExisting((log_root_dir_ + log_sub_dir_).c_str())) {
      return CreateDir((log_root_dir_ + log_sub_dir_).c_str());
    }
    return true;
  }
  bool setFilePrefix(const std::string &filePrefix) override {
    std::lock_guard<std::mutex> lock(config_mutex_);
    log_prefix_ = filePrefix;
    return true;
  }
  bool setFileInterval(long long interval_ms) override {
    std::lock_guard<std::mutex> lock(config_mutex_);
    log_file_interval_ = interval_ms;
    return true;
  }
  bool setFileKeepTime(long long keep_time_ms) override {
    std::lock_guard<std::mutex> lock(config_mutex_);
    log_keep_time_ = keep_time_ms;
    return true;
  }

  const Level& getLevel() override {
    return level_;
  }
  Target getTarget() override {
    return target_;
  }

  void excludeTimestamp() override {
    std::lock_guard<std::mutex> lock(config_mutex_);
    display_timestamp_ = false;
  }
  void includeTimestamp() override {
    std::lock_guard<std::mutex> lock(config_mutex_);
    display_timestamp_ = true;
  }
  void excludeLogLevel() override {
    std::lock_guard<std::mutex> lock(config_mutex_);
    display_level_ = false;
  }
  void includeLogLevel() override {
    std::lock_guard<std::mutex> lock(config_mutex_);
    display_level_ = true;
  }

  bool SetDrosLogPubFunc(DrosLogPublishFunc log_pub_func, const std::string &log_name = "") override {
    if (log_pub_func) {
      std::lock_guard<std::mutex> lock(config_mutex_); 
      log_pub_func_ = log_pub_func;
      dros_log_name_ = log_name;
      return true;
    }
    return false;
  }

private:
  void WorkThread();
  
  std::mutex config_mutex_;
  std::atomic_bool display_timestamp_;
  std::atomic_bool display_level_;

  Target target_;
  Level level_;

  std::string log_root_dir_;
  std::string log_sub_dir_;
  std::string log_prefix_;
  long long log_file_interval_; // ms
  long long log_keep_time_;     // ms

  std::thread work_thread_;
  std::atomic_bool to_stop_;

  std::mutex log_q_mutex_;
  std::queue<std::shared_ptr<ALog>> log_q_;

  DrosLogPublishFunc log_pub_func_;
  std::string dros_log_name_;
};

LCImpl::LCImpl() : display_timestamp_(true), display_level_(true),
    target_(Target::STDOUT), level_(Level::INFO), 
    log_root_dir_("./"), log_sub_dir_("./"), log_prefix_(""),
    log_file_interval_(0), log_keep_time_(0), to_stop_(true) {}

LCImpl::~LCImpl() {
  Quit();
}

void LCImpl::Init() {
  to_stop_ = false;
  work_thread_ = std::thread(&LCImpl::WorkThread, this);
}

void LCImpl::Quit() {
  to_stop_ = true;
  if (work_thread_.joinable())
    work_thread_.join();
}

bool LCImpl::write(Level level, const char *pMessage) {
  if (!pMessage || !(*pMessage)) {
    return false;
  }

  if ((target_ & Target::DISABLED) == Target::DISABLED) {
    return false;
  }

  if (level < level_)
    return false;
  
  auto sp_log = std::make_shared<ALog>();
  sp_log->ts = GetNow_SysTime();
  sp_log->level = level;
  sp_log->msg = pMessage;

  std::lock_guard<std::mutex> lock(log_q_mutex_);
  log_q_.push(sp_log);
  return true;
}

static const char s_U8BOMSig[] = { (char)(-17), (char)(-69), (char)(-65) };
void LCImpl::WorkThread() {
  std::shared_ptr<ALog> sp_log;
  int log_q_size = 0;
  std::vector<char> _buf, _fbuf;
  
  std::string log_root_dir;
  std::string log_sub_dir;
  std::string log_prefix, log_suffix = ".log";
  std::string log_fn;

  long long log_file_interval = 0;
  long long log_keep_time = 0;
  long long last_log_file_ts = 0;

  DrosLogPublishFunc pub_func;
  Target target;
  int file_io_cnt = 0;
  while (true) {
    Sleep(50);

    {
      std::lock_guard<std::mutex> lock(log_q_mutex_);
      log_q_size = log_q_.size();
      if (log_q_size > 0) {
        sp_log = log_q_.front();
        log_q_.pop();
      }
    }

    {
      std::lock_guard<std::mutex> lock(config_mutex_);
      // get info
      target = target_;
      pub_func = log_pub_func_;
      log_root_dir = log_root_dir_;
      log_sub_dir = log_sub_dir_;
      log_prefix = log_prefix_;
      log_file_interval = log_file_interval_;
      log_keep_time = log_keep_time_;
    }
    
    long long now_ts = GetNow_Steady();
    if ((target & Target::LOG_FILE) == Target::LOG_FILE) {
      // switch new log file
      if (last_log_file_ts <= 0 || now_ts > last_log_file_ts + log_file_interval) {
        last_log_file_ts = now_ts;
        // check dir
        if (!IsDirExisting(log_root_dir.c_str())) {
          CreateDir(log_root_dir.c_str());
        }
        if (!IsDirExisting((log_root_dir + log_sub_dir).c_str())) {
          CreateDir((log_root_dir + log_sub_dir).c_str());
        }

        // new log file name
        log_fn = log_root_dir + log_sub_dir + log_prefix +  
            GetTSText_Sec(ConvertTimeStamp(GetNow_SysTime())) + log_suffix;

        std::ofstream ofs(log_fn, std::ios::binary);
        ofs.write(s_U8BOMSig, sizeof(s_U8BOMSig));

        write(Level::INFO, ("LogClient::WorkThread() Switch new log file: " + log_fn).c_str());
      }
      
      // clear old log files
      std::vector<std::string> file_names;
      std::string log_dir = log_root_dir + log_sub_dir;
      std::string eg_fn = log_prefix + GetTSText_Sec(ConvertTimeStamp(GetNow_SysTime())) + log_suffix;
      GetAllFileName(log_dir, file_names);
      for (size_t i = 0; i < file_names.size(); ++i) {
        std::string fn = file_names[i];
        if (fn.size() == eg_fn.size()) {
          std::string fn_prefix(fn, 0, log_prefix.length());
          std::string fn_suffix(fn, fn.length()-log_suffix.length(), log_suffix.length());
          std::string fn_ts_str(fn, log_prefix.length(), fn.length()-log_prefix.length()-log_suffix.length());

          if (fn_prefix == log_prefix && fn_suffix == log_suffix) {
            long long fn_ts = ConvertTSText_Sec(fn_ts_str);
            if (fn_ts > 0 && fn_ts + log_keep_time < GetNow_SysTime()) {
              DeleteFile((log_dir+fn).c_str());
              write(Level::INFO, ("LogClient::WorkThread() Delete old log file: " + log_dir + fn).c_str());
            }
          }
        }
      }
    }

    bool to_file = (log_fn.length() > 0 && (target & Target::LOG_FILE) == Target::LOG_FILE);

    if (log_q_size > 0 && sp_log.get()) {
      // publish to dros logger
      if ((target & Target::DROS_LOGGER) == Target::DROS_LOGGER && pub_func) {
        pub_func(sp_log->ts, (int)sp_log->level, dros_log_name_, sp_log->msg);
      }

      // gen log msg
      std::string toLog;
      if (display_level_) {
        auto it = k_LL2Str.find(sp_log->level);
        if (it != k_LL2Str.end()) {
          toLog += it->second;
        }
      }
      if (display_timestamp_) {
        char fullstr[80];
        auto ts = ConvertTimeStamp(sp_log->ts);
        snprintf(fullstr, 80, "[%04d/%02d/%02d-%02d:%02d:%02d.%03d] ",
              ts.year, ts.mon, ts.day,
              ts.hour, ts.min, ts.sec, ts.milSec);
        toLog += fullstr;
      }

      toLog += sp_log->msg;
      if (toLog.length() == 0 || toLog.back() != 0x0A) {
        toLog += "\n";
      }

      // output to stdout
      std::cout << toLog;
      std::cout.flush();
      if ((target & Target::STDOUT) == Target::STDOUT) {
      } else if ((target & Target::STDERR) == Target::STDERR) {
        std::cerr << toLog;
        std::cerr.flush();
      }

      if (to_file) {
        _fbuf.insert(_fbuf.end(), toLog.begin(), toLog.end());
      }
    }

    bool to_finish = (to_stop_ && log_q_size == 0);

    // output to file
    if (to_file && (++file_io_cnt > 9 || to_finish)) {
      file_io_cnt = 0;
      if (!_fbuf.empty()) {
        std::ofstream ofs(log_fn, std::ios::app | std::ios::binary);
#if defined(_WIN32) || defined(_WIN64)
        _fbuf.emplace_back('\0');
        std::string str = DDRSys::sysStr_to_utf8(&_fbuf[0]);
        ofs.write(str.c_str(), str.length());
#elif defined(__linux__)
        ofs.write(&_fbuf[0], _fbuf.size());
#endif
      }
      _fbuf.resize(0);
    }      

    if (to_finish) {
      break;
    }
  }
}

} // namespace 


std::string LogClient::levelTostring(Level level) {
  auto it = k_LL2Str.find(level);
  if (it != k_LL2Str.end()) {
    return it->second;
  }
  return "";
}

std::shared_ptr<LogClient> LogClient::Instance() {
  static std::shared_ptr<LogClient> sp_log_client = std::make_shared<LCImpl>();
  return sp_log_client;
}

void LogClient_Log(int level, const char *format, ...) {
  if (level >= LogClient::Instance()->getLevel()) {
    const int CAP0 = 256, N_MAX_STR_LEN = 4096;
    char buff[CAP0];
    std::vector<char> bufVec;
    char *pBuf = buff;
    int nCap = CAP0;
    va_list args;
    va_start(args, format);
  #if defined(_WIN32) || defined(_WIN64)
    int nWritten = _vsnprintf_s(pBuf, nCap, nCap - 1, format, args);
  #elif defined(__linux__)
    int nWritten = vsnprintf(pBuf, nCap, format, args);
  #endif
    va_end(args);
    while (1) {
      if (nWritten > 0 && nWritten < nCap) {
        break;
      }
      if (nWritten <= 0) {
        nCap += (nCap >> 1) + 1;
      } else if (nWritten < N_MAX_STR_LEN) {
        nCap = nWritten + 1;
      } else {
        va_end(args);
        LogClient::Instance()->write((LogClient::Level)level, "ERROR - input texts too long!");
        return;
      }
      bufVec.resize(nCap);
      pBuf = &bufVec[0];
      va_start(args, format);
  #if defined(_WIN32) || defined(_WIN64)
      nWritten = _vsnprintf_s(pBuf, nCap, nCap - 1, format, args);
  #elif defined(__linux__)
      nWritten = vsnprintf(pBuf, nCap, format, args);
  #endif
      va_end(args);
    }
    LogClient::Instance()->write((LogClient::Level)level, pBuf);
  }
}

void LogClient_Init(const LogClientConfig &config) {
  if (config.display_level)
    LogClient::Instance()->includeLogLevel();
  else 
    LogClient::Instance()->excludeLogLevel();

  if (config.display_timestamp)
    LogClient::Instance()->includeTimestamp();
  else
    LogClient::Instance()->excludeTimestamp();
  
  LogClient::Instance()->setTarget(config.target);
  LogClient::Instance()->setLevel(config.level);

  LogClient::Instance()->setFileDir(config.log_root_dir);
  LogClient::Instance()->setFileSubDir(config.log_sub_dir);
  LogClient::Instance()->setFilePrefix(config.log_prefix);

  LogClient::Instance()->setFileInterval(config.log_file_interval);
  LogClient::Instance()->setFileKeepTime(config.log_keep_time);

  LogClient::Instance()->Init();
}

void LogClient_Quit() {
  LogClient::Instance()->Quit();
}

} // namespace utils