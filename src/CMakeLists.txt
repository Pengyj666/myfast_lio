# cmake_minimum_required(VERSION 2.8.3)
# project(fast_lio)

# SET(CMAKE_BUILD_TYPE "Debug")

# ADD_COMPILE_OPTIONS(-std=c++14 )
# ADD_COMPILE_OPTIONS(-std=c++14 )
# set( CMAKE_CXX_FLAGS "-std=c++14 -O3" )

# add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")

# message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
# if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
#   include(ProcessorCount)
#   ProcessorCount(N)
#   message("Processer number:  ${N}")
#   if(N GREATER 8)
#     add_definitions(-DMP_EN)
#     add_definitions(-DMP_PROC_NUM=7)
#     message("core for MP: 7")
#   elseif(N GREATER 4)
#     add_definitions(-DMP_EN)
#     add_definitions(-DMP_PROC_NUM=3)
#     message("core for MP: 3")
#   elseif(N GREATER 3)
#     add_definitions(-DMP_EN)
#     add_definitions(-DMP_PROC_NUM=2)
#     message("core for MP: 2")
#   else()
#     add_definitions(-DMP_PROC_NUM=1)
#   endif()
# else()
#   add_definitions(-DMP_EN)
#   add_definitions(-DMP_PROC_NUM=6)
# endif()

# find_package(OpenMP QUIET)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include ${CMAKE_CURRENT_SOURCE_DIR}/src/laserMapping_pch.h")

# find_package(PythonLibs REQUIRED)
# find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

# find_package(catkin REQUIRED COMPONENTS
#   geometry_msgs
#   nav_msgs
#   sensor_msgs
#   roscpp
#   rospy
#   std_msgs
#   pcl_ros
#   tf
#   livox_ros_driver
#   message_generation
#   eigen_conversions
# )

# find_package(Eigen3 REQUIRED)
# find_package(PCL 1.8 REQUIRED)

# message(Eigen: ${EIGEN3_INCLUDE_DIR})

# include_directories(
# 	${catkin_INCLUDE_DIRS} 
#   ${EIGEN3_INCLUDE_DIR}
#   ${PCL_INCLUDE_DIRS}
#   ${PYTHON_INCLUDE_DIRS}
#   include)

# add_message_files(
#   FILES
#   Pose6D.msg
# )

# generate_messages(
#  DEPENDENCIES
#  geometry_msgs
# )

# catkin_package(
#   CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
#   DEPENDS EIGEN3 PCL
#   INCLUDE_DIRS
# )

# add_executable(fastlio_mapping src/laserMapping.cpp src/preprocess.cpp src/laserMapping_mapping.cpp  include/ikd-Tree/ikd_Tree.cpp src/IMU_Processing.cpp src/laserMapping_ros.cpp src/laserMapping_help.cpp )
# set_source_files_properties(src/laserMapping.cpp PROPERTIES COMPILE_FLAGS "-std=c++14 -O1")
# add_dependencies(fastlio_mapping ${PROJECT_NAME}_generate_messages_cpp)
# target_link_libraries(fastlio_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
# target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})
#  SET(CMAKE_BUILD_TYPE Debug)



cmake_minimum_required(VERSION 2.8.3)
project(fast_lio)

set(PROJECT_VERSION 1.0.0)

SET(CMAKE_BUILD_TYPE "Release")

ADD_COMPILE_OPTIONS(-std=c++14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")

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
  add_definitions(-DMP_PROC_NUM=6)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include ${CMAKE_CURRENT_SOURCE_DIR}/src/laserMapping_pch.h")

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
  livox_ros_driver
  message_generation
  eigen_conversions
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)


add_message_files(
  FILES
  Pose6D.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
  ${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS include
  LIBRARIES fastlio_library
)

set(FAST_LIO_SOURCES
  src/laserMapping.cpp 
  src/preprocess.cpp 
  src/laserMapping_mapping.cpp  
  include/ikd-Tree/ikd_Tree.cpp 
  src/IMU_Processing.cpp 
  src/laserMapping_ros.cpp 
  src/laserMapping_help.cpp
)


# 创建共享库而不是可执行文件
add_library(fastlio_library SHARED
  ${FAST_LIO_SOURCES}
)

# 消息生成依赖
add_dependencies(fastlio_library ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

set_source_files_properties(src/laserMapping.cpp PROPERTIES COMPILE_FLAGS "-std=c++14 -O1")

# 链接库
target_link_libraries(fastlio_library 
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES} 
  ${PYTHON_LIBRARIES}
)

target_include_directories(fastlio_library PRIVATE ${PYTHON_INCLUDE_DIRS})

# 设置库属性
set_target_properties(fastlio_library PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
)

add_executable(fastlio_mapping ${FAST_LIO_SOURCES})
add_dependencies(fastlio_mapping ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(fastlio_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})
set_target_properties(fastlio_mapping PROPERTIES OUTPUT_NAME "fastlio_mapping")

# SET(CMAKE_BUILD_TYPE Debug)