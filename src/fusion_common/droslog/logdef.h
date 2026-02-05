#ifndef DROS_UTILS_DROSLOG_LOG_DEF_H
#define DROS_UTILS_DROSLOG_LOG_DEF_H

#include <functional>
#include <string>

namespace utils {

// log_file: log_root_dir/log_name/log_name_timestamp.log
// timestamp: ms
typedef std::function<bool(long long timestamp, int level, const std::string &log_name, const std::string &log_msg)> DrosLogPublishFunc;

} // namespace utils
#endif//DROS_UTILS_DROSLOG_LOG_DEF_H