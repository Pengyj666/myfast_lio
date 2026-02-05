cmake_minimum_required(VERSION 2.8.3)
project(lidar_localization_ros)

set(PROJECT_VERSION 1.0.0)

SET(CMAKE_BUILD_TYPE "Release")

ADD_COMPILE_OPTIONS(-std=c++17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")

# add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")  # 测试路径
add_definitions(-DROOT_DIR="/userdata/RobotData/map/")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -std=c++0x -std=c++17 -fexceptions")
#读盘核数 决定开多少线程
message("Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processor number:  ${N}")
  if(N GREATER 8)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=7)
    message("core for MP: 7")
  elseif(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_EN)
  add_definitions(-DMP_PROC_NUM=3)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  message_generation
  eigen_conversions
  mower_msgs
)
set(DEPENDS roscpp tf tf2_ros message_generation roslib geometry_msgs sensor_msgs nav_msgs mower_msgs GeographicLib SUITESPARSE G2O)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(G2O REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)

add_message_files(
  FILES
  Pose6D.msg
)

add_service_files(
  FILES
  Trigger.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

message(Eigen: ${EIGEN3_INCLUDE_DIR})
set(common_DIR ${PROJECT_SOURCE_DIR}/../fusion_common)
include_directories(
  ${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIRS}  
  ${common_DIR}         
  ${common_DIR}/common      
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/src/lio  
  include
  ${PROJECT_SOURCE_DIR}/include/ikd-Tree
)



catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy message_runtime std_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS include
  LIBRARIES fastlio_library
)

set(FAST_LIO_SOURCES
  ${common_DIR}/common/offset_timer.cpp
  ${common_DIR}/common/sysutils.cpp
  ${common_DIR}/droslog/logclient.cpp
  include/ikd-Tree/ikd_Tree.cpp 
  src/as_lio_node.cpp 
  src/lio/preprocess.cpp  
  src/lio/IMU_Processing.cpp 
  src/lio/lio_helper.cpp
  src/lio_code.cpp
  src/lio_node.cpp 
  src/lio_node_basic.cpp 
  src/lio/lio_registration.cpp
)


# 创建共享库而不是可执行文件
add_library(fastlio_library SHARED
  ${FAST_LIO_SOURCES}
)

# 消息生成依赖
add_dependencies(fastlio_library ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# 链接库
target_link_libraries(fastlio_library 
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES} 
)

# 设置库属性
set_target_properties(fastlio_library PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
)

add_executable(fastlio_mapping ${FAST_LIO_SOURCES})
add_dependencies(fastlio_mapping ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(fastlio_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} )
target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})
set_target_properties(fastlio_mapping PROPERTIES OUTPUT_NAME "fastlio_mapping")

add_executable(fast_lio_localization 
  src/as_lreloc_node.cpp
  src/lreloc/lreloc_function.cpp
  src/lreloc/icp_3d.cpp
  src/lreloc_node.cpp
  src/lreloc_node_basic.cpp
  src/lreloc_TF_basic.cpp
  src/lreloc_TF_fusion.cpp)
#连接库 需要IKD-Tree支持
target_link_libraries(fast_lio_localization ${catkin_LIBRARIES}  ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} fastlio_library)
target_include_directories(fast_lio_localization PRIVATE ${PYTHON_INCLUDE_DIRS})

# add_executable(transform_fusion src/transform_fusion.cpp)
# target_link_libraries(transform_fusion ${catkin_LIBRARIES})
# add_dependencies(transform_fusion ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})