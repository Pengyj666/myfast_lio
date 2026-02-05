#ifndef UTILS_COMMON_DATA_UTILS_H
#define UTILS_COMMON_DATA_UTILS_H

namespace common {

template<typename T>
struct Timed {
  double ts = -1.0;     // sec, >= 0.0 is valid
  T data;
};

template<typename T>
struct Validated {
  bool valid = false;
  T data;
};

template<typename T>
struct Scored {
  float score = 0.f;
  T data;
};

} // namespace common
#endif//UTILS_COMMON_DATA_UTILS_H
