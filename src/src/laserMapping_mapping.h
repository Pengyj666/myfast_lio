#ifndef LASERMAPPING_MAPPING_H
#define LASERMAPPING_MAPPING_H 

#include "laserMapping_help.h"
#include <std_srvs/SetBool.h> 
// #include <lidar_localization_ros/Trigger.h> 
#include "mower_msgs/Trigger.h"
#include <fstream>
#include <iomanip>
#include "common/sysutils.h"
using namespace utils;
bool loadExistingMap();
void exportStaticMapExample( );

bool save_map_cbk(mower_msgs::TriggerRequest &req,mower_msgs::TriggerResponse &res);
bool ctrl_mapping_cbk(mower_msgs::TriggerRequest &req,mower_msgs::TriggerResponse &res);

void save_map_accumulated_cloud();
void save_map_PclWaitSave();

#endif