cmake_minimum_required(VERSION 3.8)
project(fusion_localization_ros)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(catkin REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosbag
  nav_msgs
  tf  
  tf2_ros
  std_msgs 
  std_srvs
  nav_msgs
  geometry_msgs
  geographic_msgs
  sensor_msgs
  roslib
  message_generation
  mower_gps_msgs
  mower_msgs
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules "/usr/share/cmake/geographiclib/")
find_package(Eigen3 REQUIRED)
find_package(SUITESPARSE REQUIRED)
find_package(G2O REQUIRED)
find_package(GeographicLib REQUIRED)

set(DEPENDS roscpp tf tf2_ros message_generation roslib geometry_msgs sensor_msgs nav_msgs mower_msgs GeographicLib SUITESPARSE G2O)

catkin_package(
)

set(fusion_DIR ${PROJECT_SOURCE_DIR}/src)
set(common_DIR ${PROJECT_SOURCE_DIR}/../fusion_common)

#********include************* 
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${SUITESPARSE_INCLUDE_DIRS} 
  ${G2O_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}

  ${PROJECT_SOURCE_DIR}/src
  ${fusion_DIR}
  ${common_DIR}
)

set(OUTPUT_NAME "fusion_localization_node")
############# loc node ###################
add_executable(${OUTPUT_NAME} 
  src/loc_node.cpp
  src/loc_basic.cpp
  src/loc_core.cpp
  src/loc_pub.cpp
  src/loc_srv.cpp
  src/loc_sub_lidar.cpp
  src/loc_sub_vision.cpp
  src/loc_sub.cpp
  ${fusion_DIR}/locator/simple_vio_locator.cpp
  ${fusion_DIR}/common/dist_odometer.cpp
  ${fusion_DIR}/common/fusion_def.cpp
  ${fusion_DIR}/common/gnss_converter.cpp
  ${fusion_DIR}/common/gnss_initor.cpp
  ${fusion_DIR}/common/heading_estimator.cpp
  ${fusion_DIR}/common/loc_tracker.cpp
  ${fusion_DIR}/common/sensor_monitor.cpp
  ${fusion_DIR}/common/simple_tracker.cpp
  ${fusion_DIR}/common/vio_gnss_align.cpp
  ${fusion_DIR}/common/vio_tracker.cpp
  ${fusion_DIR}/common/vio_gnss_initor.cpp
  ${common_DIR}/common/data_type.cpp
  ${common_DIR}/common/sysutils.cpp
  ${common_DIR}/droslog/logclient.cpp
  ${common_DIR}/geo_utils/geo_utils.cpp
  ${common_DIR}/geo_utils/tf_helper.cpp
)

target_link_libraries(${OUTPUT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${GeographicLib_LIBRARIES}
  ${SUITESPARSE_LIBRARIES} 
  ${G2O_LIBRARIES}
)

add_executable(test_tool_node
  src/test/test_tool.cpp
  src/test/ctrl_line_test.cpp
  ${common_DIR}/common/data_type.cpp
  ${common_DIR}/common/sysutils.cpp
  ${common_DIR}/droslog/logclient.cpp
  ${common_DIR}/geo_utils/geo_utils.cpp
)

target_link_libraries(test_tool_node
  ${catkin_LIBRARIES}
)

if(DEFINED MOWER_TEST)
  message("MOWER_TEST is defined")
else()
  message("MOWER_TEST is not defined")
endif()