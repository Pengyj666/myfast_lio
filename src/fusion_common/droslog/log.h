#ifndef DROS_UTILS_DROSLOG_LOG_H
#define DROS_UTILS_DROSLOG_LOG_H

namespace utils {

enum LogLevel : int {
  ALL = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  FATAL = 5,
};

typedef void(* p_log_func)(int level, const char *format, ...);
extern p_log_func dros_log_func_ptr;
#define droslog(...) if (dros_log_func_ptr) (dros_log_func_ptr)(__VA_ARGS__);

} // namespace utils
#endif//DROS_UTILS_DROSLOG_LOG_H