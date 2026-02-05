#ifndef UTILS_COMMON_COMMON_DEF_H
#define UTILS_COMMON_COMMON_DEF_H

#include <string>

namespace common {
  
const std::string RTK_UNKNOWN = "UNKNOWN";
const std::string RTK_NARROW_INT = "NARROW_INT";
const std::string RTK_NARROW_FLOAT = "NARROW_FLOAT";
const std::string RTK_PSRDIFF = "PSRDIFF";
const std::string RTK_SINGLE = "SINGLE";
const std::string RTK_L1_INT = "L1_INT";
const std::string RTK_L1_FLOAT = "L1_FLOAT";
const std::string RTK_BASE_UNFIXED = "BASE_UNFIXED";  // 基站未固定
const std::string RTK_BASE_FIXED = "BASE_FIXED";      // 基站已固定

enum LocStage {
  IDLE = 0,
  INITING = 1,
  TRACKING = 2,
  LOST = 3,
  ERROR = 4,
};

const double k_pi = 3.14159265358979323846;
const float  k_epsilon = 1e-6f;

} // namespace common
#endif//UTILS_COMMON_COMMON_DEF_H
