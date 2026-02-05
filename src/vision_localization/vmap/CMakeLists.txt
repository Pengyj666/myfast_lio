cmake_minimum_required(VERSION 2.8.3)
project(vmap)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rosbag
    std_msgs
    nav_msgs
    as_camera_models
    cv_bridge
    roslib
    image_transport
    mower_msgs
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(SUITESPARSE REQUIRED)
find_package(G2O REQUIRED)
find_package(Eigen3)

set(common_DIR ${PROJECT_SOURCE_DIR}/../../fusion_common)

include_directories(
    ${catkin_INCLUDE_DIRS} 
    ${CERES_INCLUDE_DIRS}  
    ${EIGEN3_INCLUDE_DIR}
    ${SUITESPARSE_INCLUDE_DIRS} 
    ${G2O_INCLUDE_DIR}
    ${common_DIR}
)

catkin_package()

add_executable(as_vmap
    src/as_vmap.cpp
    src/simple_pose_graph.cpp
    src/keyframe.cpp
    src/spa_align.cpp
    src/vreloc_tracker.cpp
    src/cell.cpp
    src/submap.cpp
    src/spatial_map_manager.cpp
    src/submap_serializer.cpp
    src/spatial_meta.cpp
    src/submap_cache.cpp
    src/segment_optimizer.cpp
    src/disk_writer.cpp
    src/utility/map_drawer.cpp
    src/utility/CameraPoseVisualization.cpp
    src/ThirdParty/DBoW/BowVector.cpp
    src/ThirdParty/DBoW/FBrief.cpp
    src/ThirdParty/DBoW/FeatureVector.cpp
    src/ThirdParty/DBoW/QueryResults.cpp
    src/ThirdParty/DBoW/ScoringObject.cpp
    src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp
    src/ThirdParty/DVision/BRIEF.cpp
    src/ThirdParty/VocabularyBinary.cpp
    ${common_DIR}/common/sysutils.cpp
    ${common_DIR}/droslog/logclient.cpp
    ${common_DIR}/geo_utils/geo_utils.cpp
    ${common_DIR}/geo_utils/tf_helper.cpp
)
target_link_libraries(as_vmap 
    ${catkin_LIBRARIES}  
    ${OpenCV_LIBS} 
    ${CERES_LIBRARIES}
    ${SUITESPARSE_LIBRARIES} 
    ${G2O_LIBRARIES}
)
